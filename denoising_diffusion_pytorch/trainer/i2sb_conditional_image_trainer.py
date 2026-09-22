"""
I2SB conditional image trainer.

This trainer is intentionally separate from
`diffusion_conditional_image_trainer.py` so that the existing Conditional DDPM /
VoxelDiffusionCut training pipeline remains untouched.

Training follows the official NVlabs/I2SB core procedure:

    1. sample paired boundaries (x0, x1)
    2. sample timestep t
    3. sample xt ~ q(xt | x0, x1)
    4. compute Eq. (12) target: (xt - x0) / sigma_t
    5. predict the target with the existing conditional UNet / DiT
    6. compute MSE on the hidden region
    7. backpropagate and update EMA

For this project:
    x0   = full slice image (exterior + internal structure)
    x1   = exterior-only slice image
    cond = partially observed x0
    mask = binary mask, 0=observed and 1=hidden

Inference / I2SB reverse sampling is intentionally NOT implemented here yet.
That will be added after the training core is validated.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
        # Kept for config compatibility with the existing trainer.
        num_samples=25,
        calculate_fid=False,
        save_best_and_latest_only=False,
        **unused_kwargs,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # Accelerator
        # ------------------------------------------------------------
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else "no",
        )

        # ------------------------------------------------------------
        # Core I2SB components
        # ------------------------------------------------------------
        self.model = model
        self.diffusion = diffusion
        self.dataset = dataset

        self.channels = model.channels
        self.image_size = dataset.image_size

        # ------------------------------------------------------------
        # Training configuration
        # ------------------------------------------------------------
        self.batch_size = int(train_batch_size)
        self.gradient_accumulate_every = int(gradient_accumulate_every)
        self.train_num_steps = int(train_num_steps)
        self.save_and_sample_every = int(save_and_sample_every)
        self.max_grad_norm = float(max_grad_norm)

        if self.batch_size <= 0:
            raise ValueError("train_batch_size must be positive.")

        if self.gradient_accumulate_every <= 0:
            raise ValueError("gradient_accumulate_every must be positive.")

        if self.train_num_steps <= 0:
            raise ValueError("train_num_steps must be positive.")

        if len(dataset) < 2:
            raise ValueError(
                f"I2SB training requires at least 2 samples, got {len(dataset)}."
            )

        # These are currently not used because reverse I2SB sampling / FID
        # evaluation has not yet been connected.
        self.num_samples = num_samples
        self.calculate_fid = calculate_fid
        self.save_best_and_latest_only = save_best_and_latest_only

        if self.calculate_fid:
            self.accelerator.print(
                "[I2SB] calculate_fid=True was provided, but FID evaluation "
                "is deferred until the I2SB reverse sampler is implemented."
            )

        if unused_kwargs and self.accelerator.is_main_process:
            self.accelerator.print(
                "[I2SB] Unused trainer config keys:",
                sorted(unused_kwargs.keys()),
            )

        # ------------------------------------------------------------
        # Dataset split / dataloaders
        #
        # Keep the existing repository convention: random 90/10 split.
        # ------------------------------------------------------------
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

        val_dl = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
        )

        # ------------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------------
        self.opt = Adam(
            self.model.parameters(),
            lr=train_lr,
            betas=adam_betas,
        )

        # ------------------------------------------------------------
        # Output directories
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # EMA
        #
        # Match the existing repository's EMA behavior.
        # ------------------------------------------------------------
        if self.accelerator.is_main_process:
            self.ema = EMA(
                self.model,
                beta=ema_decay,
                update_every=ema_update_every,
            )
            self.ema.to(self.device)

        # ------------------------------------------------------------
        # Accelerate preparation
        # ------------------------------------------------------------
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
        """
        Compute one I2SB training loss.

        Parameters
        ----------
        x0 : Tensor [B, 3, H, W]
            Full target slice.

        x1 : Tensor [B, 3, H, W]
            Exterior-only bridge endpoint.

        cond : Tensor [B, 3, H, W]
            Partial observation supplied to the conditional network.

        mask : Tensor [B, 1, H, W]
            Binary mask:
                0 = observed / known
                1 = hidden / prediction region

        step : Optional Tensor [B]
            If omitted, timesteps are sampled uniformly.

        ot_ode : bool
            If False (default), use stochastic I2SB bridge samples.
            This matches standard I2SB training rather than OT-ODE.

        Returns
        -------
        loss, diagnostics
        """
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

        # ----------------------------------------------------------
        # Official I2SB:
        #   xt ~ q(xt | x0, x1)
        # ----------------------------------------------------------
        xt = self.diffusion.q_sample(
            step=step,
            x0=x0,
            x1=x1,
            ot_ode=ot_ode,
        )

        # ----------------------------------------------------------
        # Eq. (12):
        #   target = (xt - x0) / sigma_t
        # ----------------------------------------------------------
        label = self.diffusion.compute_label(
            step=step,
            x0=x0,
            xt=xt,
        )

        # ----------------------------------------------------------
        # Existing conditional UNet / DiT interface:
        #
        #   x           = xt
        #   time        = step
        #   x_self_cond = None
        #   mask_cond   = cond
        #   binary_mask = mask
        # ----------------------------------------------------------
        pred = self.model(
            xt,
            step,
            None,
            cond,
        )
        # import ipdb; ipdb.set_trace()

        if pred.shape != label.shape:
            raise RuntimeError(
                "Network prediction and I2SB target must have identical "
                f"shapes, got pred={tuple(pred.shape)}, "
                f"label={tuple(label.shape)}."
            )

        # ----------------------------------------------------------
        # Match the official I2SB inpainting-style masked objective:
        # optimize only the hidden / prediction region.
        #
        # In this repository:
        #   mask == 1 -> hidden
        #   mask == 0 -> observed
        # ----------------------------------------------------------
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
                        # Reverse I2SB sampling is not connected yet.
                        # For now, save a training checkpoint only.
                        self.save(self.step)

                pbar.set_description(
                    f"loss: {total_loss:.4f}"
                )
                pbar.update(1)

        self.writer.close()

        accelerator.print(
            "I2SB training complete"
        )
