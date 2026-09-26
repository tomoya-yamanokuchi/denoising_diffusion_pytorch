
# Suggested replacement for denoising_diffusion_pytorch/trainer/i2sb_conditional_image_trainer.py
# Core additions:
# - torchvision.utils import
# - periodic EMA reverse-sampling at checkpoint time
# - minimal logging set: target X0, condition C, mask, reconstruction
# - TensorBoard image logging
# - configurable sample NFE and number of log samples

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm

from denoising_diffusion_pytorch.models.helpers import (
    cycle,
    divisible_by,
    exists,
)
from denoising_diffusion_pytorch.version import __version__


class Trainer:
    def __init__(
        self,
        model,
        diffusion,
        dataset,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=2000,
        results_folder="./results",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        max_grad_norm=1.0,
        num_workers=8,
        num_samples=4,
        sample_nfe=20,
        calculate_fid=False,
        save_best_and_latest_only=False,
        **unused_kwargs,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else "no",
        )

        self.model = model
        self.diffusion = diffusion
        self.dataset = dataset

        self.channels = model.channels
        self.image_size = dataset.image_size

        self.batch_size = int(train_batch_size)
        self.gradient_accumulate_every = int(gradient_accumulate_every)
        self.train_num_steps = int(train_num_steps)
        self.save_and_sample_every = int(save_and_sample_every)
        self.max_grad_norm = float(max_grad_norm)

        # Periodic visualization settings
        self.num_samples = int(num_samples)
        self.sample_nfe = int(sample_nfe)

        if self.batch_size <= 0:
            raise ValueError("train_batch_size must be positive.")
        if self.gradient_accumulate_every <= 0:
            raise ValueError("gradient_accumulate_every must be positive.")
        if self.train_num_steps <= 0:
            raise ValueError("train_num_steps must be positive.")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if self.sample_nfe <= 0:
            raise ValueError("sample_nfe must be positive.")
        if len(dataset) < 2:
            raise ValueError(
                f"I2SB training requires at least 2 samples, got {len(dataset)}."
            )

        self.calculate_fid = calculate_fid
        self.save_best_and_latest_only = save_best_and_latest_only

        if self.calculate_fid:
            self.accelerator.print(
                "[I2SB] calculate_fid=True was provided, but FID evaluation "
                "is not connected in this trainer."
            )

        if unused_kwargs and self.accelerator.is_main_process:
            self.accelerator.print(
                "[I2SB] Unused trainer config keys:",
                sorted(unused_kwargs.keys()),
            )

        data_samples = len(dataset)
        train_size = int(data_samples * 0.9)
        val_size = data_samples - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
        )

        train_dl = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        # Keep validation deterministic in ordering, matching current I2SB code.
        val_dl = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
        )

        self.opt = Adam(
            self.model.parameters(),
            lr=train_lr,
            betas=adam_betas,
        )

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.sw_dir = self.results_folder / "sw_dir"
        self.sw_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.step = 0

        if self.accelerator.is_main_process:
            self.ema = EMA(
                self.model,
                beta=ema_decay,
                update_every=ema_update_every,
            )
            self.ema.to(self.device)

        self.model, self.opt, train_dl, val_dl = self.accelerator.prepare(
            self.model,
            self.opt,
            train_dl,
            val_dl,
        )

        self.train_dl = cycle(train_dl)
        self.val_dl = cycle(val_dl)

    @property
    def device(self):
        return self.accelerator.device

    # ==================================================================
    # Checkpoint I/O
    # ==================================================================

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": (
                self.accelerator.scaler.state_dict()
                if exists(self.accelerator.scaler)
                else None
            ),
            "version": __version__,
            "method": "i2sb",
        }

        torch.save(
            data,
            str(self.results_folder / f"model-{milestone}.pt"),
        )

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        checkpoint_path = self.results_folder / f"model-{milestone}.pt"

        data = torch.load(
            str(checkpoint_path),
            map_location=device,
        )

        model = accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])

        if accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if (
            exists(accelerator.scaler)
            and exists(data.get("scaler", None))
        ):
            accelerator.scaler.load_state_dict(
                data["scaler"]
            )

        if "version" in data:
            accelerator.print(
                f"loading from version {data['version']}"
            )

    # ==================================================================
    # I2SB training core
    # ==================================================================

    def compute_loss(
        self,
        x0,
        x1,
        cond,
        mask,
        step=None,
        ot_ode=False,
    ):
        if x0.shape != x1.shape:
            raise ValueError(
                f"x0 and x1 shape mismatch: {x0.shape} vs {x1.shape}"
            )

        if cond.shape != x0.shape:
            raise ValueError(
                f"cond and x0 shape mismatch: {cond.shape} vs {x0.shape}"
            )

        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(
                "mask must have shape [B, 1, H, W], "
                f"got {tuple(mask.shape)}"
            )

        batch_size = x0.shape[0]

        if step is None:
            step = torch.randint(
                low=0,
                high=len(self.diffusion.betas),
                size=(batch_size,),
                device=x0.device,
                dtype=torch.long,
            )
        else:
            step = step.to(
                device=x0.device,
                dtype=torch.long,
            )

        xt = self.diffusion.q_sample(
            step=step,
            x0=x0,
            x1=x1,
            ot_ode=ot_ode,
        )

        label = self.diffusion.compute_label(
            step=step,
            x0=x0,
            xt=xt,
        )

        pred = self.model(
            xt,
            step,
            None,
            cond,
        )

        if pred.shape != label.shape:
            raise RuntimeError(
                "Network prediction and I2SB target must have identical "
                f"shapes, got pred={tuple(pred.shape)}, "
                f"label={tuple(label.shape)}."
            )

        pred_masked = mask * pred
        label_masked = mask * label

        loss = F.mse_loss(
            pred_masked,
            label_masked,
        )

        diagnostics = {
            "step": step.detach(),
            "xt": xt.detach(),
            "label": label.detach(),
            "pred": pred.detach(),
        }

        return loss, diagnostics

    # ==================================================================
    # Periodic reverse-sampling log
    # ==================================================================

    def _make_sampling_steps(self):
        interval = len(self.diffusion.betas)

        steps = np.linspace(
            0,
            interval - 1,
            self.sample_nfe + 1,
            dtype=int,
        )

        steps = np.unique(steps)

        if steps[0] != 0:
            steps = np.insert(steps, 0, 0)

        if steps[-1] != interval - 1:
            steps = np.append(
                steps,
                interval - 1,
            )

        return steps.tolist()

    @staticmethod
    def _to_01(x):
        """
        Convert project image range [-1, 1] to torchvision-save range [0, 1].
        """
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

    @staticmethod
    def _mask_to_rgb(mask):
        """
        mask is [N,1,H,W], with:
            0 observed
            1 hidden

        Keep the raw repository convention in the saved mask image:
            black = observed
            white = hidden
        """
        if mask.shape[1] == 1:
            return mask.repeat(1, 3, 1, 1)
        return mask

    @torch.no_grad()
    def sample_and_log(self, milestone):
        """
        Generate a minimal, directly comparable progress log.

        Saved:
            sample-{milestone}_pred.png
            sample-{milestone}_mask.png
            sample-{milestone}_target.png
            sample-{milestone}_cond.png
            sample-{milestone}_x1.png

        The first three names intentionally match the existing Conditional
        Diffusion trainer convention. cond and x1 are I2SB-specific additions.
        """

        if not self.accelerator.is_main_process:
            return

        print(f"[I2SB] start eval process: {milestone}")

        self.ema.ema_model.eval()

        steps = self._make_sampling_steps()

        all_pred = []
        all_mask = []
        all_target = []
        all_cond = []
        all_x1 = []

        for _ in range(self.num_samples):
            data = next(self.val_dl)

            x0 = data["x0"].to(self.device)
            x1 = data["x1"].to(self.device)
            cond = data["cond"].to(self.device)
            mask = data["mask"].to(self.device)

            def pred_x0_fn(xt, step_value):
                step = torch.full(
                    (xt.shape[0],),
                    int(step_value),
                    device=self.device,
                    dtype=torch.long,
                )

                net_out = self.ema.ema_model(
                    xt,
                    step,
                    None,
                    cond,
                )

                return self.diffusion.compute_pred_x0(
                    step=step,
                    xt=xt,
                    net_out=net_out,
                    clip_denoise=True,
                )

            xs, _ = self.diffusion.ddpm_sampling(
                steps=steps,
                pred_x0_fn=pred_x0_fn,
                x1=x1,
                cond=cond,
                mask=mask,
                ot_ode=False,
                log_steps=[0],
                verbose=False,
            )

            # ddpm_sampling() returns increasing-time order.
            recon = xs[:, 0].to(self.device)

            # Exact endpoint data consistency for observed region.
            recon = (
                (1.0 - mask) * cond
                + mask * recon
            )

            all_pred.append(recon.detach().cpu())
            all_mask.append(mask.detach().cpu())
            all_target.append(x0.detach().cpu())
            all_cond.append(cond.detach().cpu())
            all_x1.append(x1.detach().cpu())

        all_pred = torch.cat(all_pred, dim=0)
        all_mask = torch.cat(all_mask, dim=0)
        all_target = torch.cat(all_target, dim=0)
        all_cond = torch.cat(all_cond, dim=0)
        all_x1 = torch.cat(all_x1, dim=0)

        # Existing repository commonly uses square grids, but do not force
        # num_samples to be a perfect square.
        nrow = max(1, int(math.sqrt(self.num_samples)))

        pred_vis = self._to_01(all_pred)
        target_vis = self._to_01(all_target)
        cond_vis = self._to_01(all_cond)
        x1_vis = self._to_01(all_x1)
        mask_vis = self._mask_to_rgb(all_mask).clamp(0.0, 1.0)

        # Match the main Conditional Diffusion filenames first.
        utils.save_image(
            pred_vis,
            str(self.results_folder / f"sample-{milestone}_pred.png"),
            nrow=nrow,
        )
        utils.save_image(
            mask_vis,
            str(self.results_folder / f"sample-{milestone}_mask.png"),
            nrow=nrow,
        )
        utils.save_image(
            target_vis,
            str(self.results_folder / f"sample-{milestone}_target.png"),
            nrow=nrow,
        )

        # I2SB-specific context, useful for debugging bridge behavior.
        utils.save_image(
            cond_vis,
            str(self.results_folder / f"sample-{milestone}_cond.png"),
            nrow=nrow,
        )
        utils.save_image(
            x1_vis,
            str(self.results_folder / f"sample-{milestone}_x1.png"),
            nrow=nrow,
        )

        # TensorBoard
        self.writer.add_images(
            "i2sb/pred",
            pred_vis,
            self.step,
            dataformats="NCHW",
        )
        self.writer.add_images(
            "i2sb/mask",
            mask_vis,
            self.step,
            dataformats="NCHW",
        )
        self.writer.add_images(
            "i2sb/target",
            target_vis,
            self.step,
            dataformats="NCHW",
        )
        self.writer.add_images(
            "i2sb/condition",
            cond_vis,
            self.step,
            dataformats="NCHW",
        )
        self.writer.add_images(
            "i2sb/x1",
            x1_vis,
            self.step,
            dataformats="NCHW",
        )

        self.ema.ema_model.train()

    # ==================================================================
    # Main training loop
    # ==================================================================

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        self.writer = SummaryWriter(
            log_dir=self.sw_dir
        )

        self.model.train()

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.0

                for _ in range(
                    self.gradient_accumulate_every
                ):
                    data = next(self.train_dl)

                    x0 = data["x0"].to(
                        device,
                        non_blocking=True,
                    )
                    x1 = data["x1"].to(
                        device,
                        non_blocking=True,
                    )
                    cond = data["cond"].to(
                        device,
                        non_blocking=True,
                    )
                    mask = data["mask"].to(
                        device,
                        non_blocking=True,
                    )

                    with accelerator.autocast():
                        loss, _ = self.compute_loss(
                            x0=x0,
                            x1=x1,
                            cond=cond,
                            mask=mask,
                            ot_ode=False,
                        )

                        loss = (
                            loss
                            / self.gradient_accumulate_every
                        )

                        total_loss += loss.item()

                    accelerator.backward(loss)

                accelerator.wait_for_everyone()

                accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1

                if accelerator.is_main_process:
                    self.ema.update()

                    self.writer.add_scalar(
                        "Train_loss",
                        total_loss,
                        self.step,
                    )

                    if (
                        self.step != 0
                        and divisible_by(
                            self.step,
                            self.save_and_sample_every,
                        )
                    ):
                        milestone = self.step

                        # 1) Generate validation images with EMA model.
                        self.sample_and_log(milestone)

                        # 2) Save checkpoint at the same milestone.
                        self.save(milestone)

                pbar.set_description(
                    f"loss: {total_loss:.4f}"
                )
                pbar.update(1)

        self.writer.close()

        accelerator.print(
            "I2SB training complete"
        )
