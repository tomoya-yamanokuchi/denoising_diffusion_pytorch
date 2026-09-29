"""
I2SB conditional image trainer using the shared ProductAwareLogger.

Expected shared utility:
    denoising_diffusion_pytorch/utils/product_aware_logger.py

Logging layout:
    samples/
      sheetsander/
        static/
          target.png
          cond.png
          mask.png
          x1.png
        pred/
          step_005000.png
          step_010000.png
          ...
      polisher/
        ...
      powercutter/
        ...

The shared ProductAwareLogger is responsible for:
- selecting fixed validation samples per product
- caching them once
- saving static images once
- saving dynamic images per checkpoint
- TensorBoard product grouping

This trainer remains responsible for:
- I2SB training loss
- I2SB reverse sampling
- converting I2SB outputs into ProductLogOutput
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
from denoising_diffusion_pytorch.utils.product_aware_logger import (
    ProductAwareLogger,
    ProductLogOutput,
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
        # kept for backward compatibility with older configs
        num_samples=4,
        # product-aware logging
        samples_per_product=1,
        # I2SB-specific reverse-sampling NFE
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
        self.gradient_accumulate_every = int(
            gradient_accumulate_every
        )
        self.train_num_steps = int(
            train_num_steps
        )
        self.save_and_sample_every = int(
            save_and_sample_every
        )
        self.max_grad_norm = float(
            max_grad_norm
        )

        # compatibility only; ProductAwareLogger uses samples_per_product
        self.num_samples = int(num_samples)
        self.samples_per_product = int(
            samples_per_product
        )
        self.sample_nfe = int(
            sample_nfe
        )

        if self.batch_size <= 0:
            raise ValueError(
                "train_batch_size must be positive."
            )

        if self.gradient_accumulate_every <= 0:
            raise ValueError(
                "gradient_accumulate_every must be positive."
            )

        if self.train_num_steps <= 0:
            raise ValueError(
                "train_num_steps must be positive."
            )

        if self.samples_per_product <= 0:
            raise ValueError(
                "samples_per_product must be positive."
            )

        if self.sample_nfe <= 0:
            raise ValueError(
                "sample_nfe must be positive."
            )

        if len(dataset) < 2:
            raise ValueError(
                "I2SB training requires at least 2 samples, "
                f"got {len(dataset)}."
            )

        self.calculate_fid = (
            calculate_fid
        )

        self.save_best_and_latest_only = (
            save_best_and_latest_only
        )

        if self.calculate_fid:
            self.accelerator.print(
                "[I2SB] calculate_fid=True was provided, "
                "but FID evaluation is not connected "
                "in this trainer."
            )

        if (
            unused_kwargs
            and self.accelerator.is_main_process
        ):
            self.accelerator.print(
                "[I2SB] Unused trainer config keys:",
                sorted(
                    unused_kwargs.keys()
                ),
            )

        # ==============================================================
        # Dataset split
        # ==============================================================

        data_samples = len(
            dataset
        )

        train_size = int(
            data_samples * 0.9
        )

        val_size = (
            data_samples
            - train_size
        )

        train_dataset, val_dataset = (
            torch.utils.data.random_split(
                dataset,
                [
                    train_size,
                    val_size,
                ],
            )
        )

        # ==============================================================
        # Shared product-aware logger
        #
        # IMPORTANT:
        # Construct this BEFORE wrapping dataloaders with Accelerate.
        # It needs the original val_dataset.indices and dataset metadata.
        # ==============================================================

        self.product_logger = (
            ProductAwareLogger(
                dataset=dataset,
                val_dataset=val_dataset,
                results_folder=results_folder,
                samples_per_product=
                    self.samples_per_product,
                tensorboard_prefix="i2sb",
            )
        )

        train_dl = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        # This validation loader is no longer used for product-aware
        # image logging, but is retained for compatibility / future use.
        val_dl = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
        )

        # ==============================================================
        # Optimizer / output directories
        # ==============================================================

        self.opt = Adam(
            self.model.parameters(),
            lr=train_lr,
            betas=adam_betas,
        )

        self.results_folder = Path(
            results_folder
        )

        self.results_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.sw_dir = (
            self.results_folder
            / "sw_dir"
        )

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

            self.ema.to(
                self.device
            )

        self.model, self.opt, train_dl, val_dl = (
            self.accelerator.prepare(
                self.model,
                self.opt,
                train_dl,
                val_dl,
            )
        )

        self.train_dl = cycle(
            train_dl
        )

        self.val_dl = cycle(
            val_dl
        )

        if self.accelerator.is_main_process:
            self.product_logger.print_summary(
                self.accelerator.print
            )

    @property
    def device(
        self,
    ):
        return self.accelerator.device

    # ==================================================================
    # Checkpoint I/O
    # ==================================================================

    def save(
        self,
        milestone,
    ):
        if not (
            self.accelerator
            .is_local_main_process
        ):
            return

        data = {
            "step":
                self.step,
            "model":
                self.accelerator
                .get_state_dict(
                    self.model
                ),
            "opt":
                self.opt.state_dict(),
            "ema":
                self.ema.state_dict(),
            "scaler":
                (
                    self.accelerator
                    .scaler
                    .state_dict()
                    if exists(
                        self.accelerator.scaler
                    )
                    else None
                ),
            "version":
                __version__,
            "method":
                "i2sb",
        }

        torch.save(
            data,
            str(
                self.results_folder
                / f"model-{milestone}.pt"
            ),
        )

    def load(
        self,
        milestone,
    ):
        accelerator = (
            self.accelerator
        )

        device = (
            accelerator.device
        )

        checkpoint_path = (
            self.results_folder
            / f"model-{milestone}.pt"
        )

        data = torch.load(
            str(
                checkpoint_path
            ),
            map_location=device,
        )

        model = (
            accelerator
            .unwrap_model(
                self.model
            )
        )

        model.load_state_dict(
            data["model"]
        )

        self.step = data[
            "step"
        ]

        self.opt.load_state_dict(
            data["opt"]
        )

        if (
            accelerator
            .is_main_process
        ):
            self.ema.load_state_dict(
                data["ema"]
            )

        if (
            exists(
                accelerator.scaler
            )
            and exists(
                data.get(
                    "scaler",
                    None,
                )
            )
        ):
            accelerator.scaler.load_state_dict(
                data["scaler"]
            )

        if "version" in data:
            accelerator.print(
                "loading from version "
                f"{data['version']}"
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
        if (
            x0.shape
            != x1.shape
        ):
            raise ValueError(
                "x0 and x1 shape mismatch: "
                f"{x0.shape} vs {x1.shape}"
            )

        if (
            cond.shape
            != x0.shape
        ):
            raise ValueError(
                "cond and x0 shape mismatch: "
                f"{cond.shape} vs {x0.shape}"
            )

        if (
            mask.ndim != 4
            or mask.shape[1] != 1
        ):
            raise ValueError(
                "mask must have shape "
                "[B, 1, H, W], "
                f"got {tuple(mask.shape)}"
            )

        batch_size = (
            x0.shape[0]
        )

        if step is None:
            step = torch.randint(
                low=0,
                high=len(
                    self.diffusion.betas
                ),
                size=(
                    batch_size,
                ),
                device=x0.device,
                dtype=torch.long,
            )
        else:
            step = step.to(
                device=x0.device,
                dtype=torch.long,
            )

        xt = (
            self.diffusion
            .q_sample(
                step=step,
                x0=x0,
                x1=x1,
                ot_ode=ot_ode,
            )
        )

        label = (
            self.diffusion
            .compute_label(
                step=step,
                x0=x0,
                xt=xt,
            )
        )

        pred = self.model(
            xt,
            step,
            None,
            cond,
        )

        if (
            pred.shape
            != label.shape
        ):
            raise RuntimeError(
                "Network prediction and I2SB target must have "
                "identical shapes, got "
                f"pred={tuple(pred.shape)}, "
                f"label={tuple(label.shape)}."
            )

        pred_masked = (
            mask
            * pred
        )

        label_masked = (
            mask
            * label
        )

        loss = F.mse_loss(
            pred_masked,
            label_masked,
        )

        diagnostics = {
            "step":
                step.detach(),
            "xt":
                xt.detach(),
            "label":
                label.detach(),
            "pred":
                pred.detach(),
        }

        return (
            loss,
            diagnostics,
        )

    # ==================================================================
    # I2SB-specific reverse-sampling helpers
    # ==================================================================

    def _make_sampling_steps(
        self,
    ):
        interval = len(
            self.diffusion.betas
        )

        steps = np.linspace(
            0,
            interval - 1,
            self.sample_nfe + 1,
            dtype=int,
        )

        steps = np.unique(
            steps
        )

        if steps[0] != 0:
            steps = np.insert(
                steps,
                0,
                0,
            )

        if (
            steps[-1]
            != interval - 1
        ):
            steps = np.append(
                steps,
                interval - 1,
            )

        return steps.tolist()

    @staticmethod
    def _to_01(
        x,
    ):
        return (
            (
                x + 1.0
            )
            / 2.0
        ).clamp(
            0.0,
            1.0,
        )

    @staticmethod
    def _mask_to_rgb(
        mask,
    ):
        if mask.shape[1] == 1:
            return mask.repeat(
                1,
                3,
                1,
                1,
            )

        return mask

    # ==================================================================
    # Adapter: I2SB -> shared ProductAwareLogger
    # ==================================================================

    @torch.no_grad()
    def _make_product_log_output(
        self,
        fixed_sample,
    ):
        """
        Model-specific adapter.

        The shared ProductAwareLogger knows nothing about:
        - I2SB bridge states
        - NFE
        - compute_pred_x0()
        - ddpm_sampling()

        It only receives visualization-ready tensors in [0, 1].
        """

        item = fixed_sample[
            "item"
        ]

        x0 = (
            item["x0"]
            .unsqueeze(0)
            .to(self.device)
        )

        x1 = (
            item["x1"]
            .unsqueeze(0)
            .to(self.device)
        )

        cond = (
            item["cond"]
            .unsqueeze(0)
            .to(self.device)
        )

        mask = (
            item["mask"]
            .unsqueeze(0)
            .to(self.device)
        )

        steps = (
            self._make_sampling_steps()
        )

        def pred_x0_fn(
            xt,
            step_value,
        ):
            step = torch.full(
                (
                    xt.shape[0],
                ),
                int(
                    step_value
                ),
                device=self.device,
                dtype=torch.long,
            )

            net_out = (
                self.ema
                .ema_model(
                    xt,
                    step,
                    None,
                    cond,
                )
            )

            return (
                self.diffusion
                .compute_pred_x0(
                    step=step,
                    xt=xt,
                    net_out=net_out,
                    clip_denoise=True,
                )
            )

        xs, _ = (
            self.diffusion
            .ddpm_sampling(
                steps=steps,
                pred_x0_fn=
                    pred_x0_fn,
                x1=x1,
                cond=cond,
                mask=mask,
                ot_ode=False,
                log_steps=[
                    0
                ],
                verbose=False,
            )
        )

        # ddpm_sampling() returns increasing-time order.
        recon = (
            xs[:, 0]
            .to(self.device)
        )

        # Exact endpoint consistency:
        # observed region comes directly from condition.
        recon = (
            (
                1.0
                - mask
            )
            * cond
            + mask
            * recon
        )

        return ProductLogOutput(
            product_name=
                fixed_sample[
                    "product_name"
                ],
            static={
                "target":
                    self._to_01(
                        x0
                    ),
                "cond":
                    self._to_01(
                        cond
                    ),
                "mask":
                    self._mask_to_rgb(
                        mask
                    ).clamp(
                        0.0,
                        1.0,
                    ),
                "x1":
                    self._to_01(
                        x1
                    ),
            },
            dynamic={
                "pred":
                    self._to_01(
                        recon
                    ),
            },
        )

    # ==================================================================
    # Product-aware periodic reverse-sampling log
    # ==================================================================

    @torch.no_grad()
    def sample_and_log(
        self,
        milestone,
    ):
        if not (
            self.accelerator
            .is_main_process
        ):
            return

        self.accelerator.print(
            "[I2SB] start product-aware "
            f"eval process: {milestone}"
        )

        self.ema.ema_model.eval()

        self.product_logger.log(
            step=milestone,
            sample_fn=
                self._make_product_log_output,
            writer=self.writer,
        )

        self.ema.ema_model.train()

        self.accelerator.print(
            "[I2SB] product-aware samples saved under "
            f"{self.product_logger.samples_dir}"
        )

    # ==================================================================
    # Main training loop
    # ==================================================================

    def train(
        self,
    ):
        accelerator = (
            self.accelerator
        )

        device = (
            accelerator.device
        )

        self.writer = (
            SummaryWriter(
                log_dir=self.sw_dir
            )
        )

        self.model.train()

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not (
                accelerator
                .is_main_process
            ),
        ) as pbar:

            while (
                self.step
                < self.train_num_steps
            ):

                total_loss = 0.0

                for _ in range(
                    self.gradient_accumulate_every
                ):
                    data = next(
                        self.train_dl
                    )

                    x0 = data[
                        "x0"
                    ].to(
                        device,
                        non_blocking=True,
                    )

                    x1 = data[
                        "x1"
                    ].to(
                        device,
                        non_blocking=True,
                    )

                    cond = data[
                        "cond"
                    ].to(
                        device,
                        non_blocking=True,
                    )

                    mask = data[
                        "mask"
                    ].to(
                        device,
                        non_blocking=True,
                    )

                    with (
                        accelerator
                        .autocast()
                    ):
                        loss, _ = (
                            self.compute_loss(
                                x0=x0,
                                x1=x1,
                                cond=cond,
                                mask=mask,
                                ot_ode=False,
                            )
                        )

                        loss = (
                            loss
                            / self.gradient_accumulate_every
                        )

                        total_loss += (
                            loss.item()
                        )

                    accelerator.backward(
                        loss
                    )

                accelerator.wait_for_everyone()

                accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1

                if (
                    accelerator
                    .is_main_process
                ):
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
                        milestone = (
                            self.step
                        )

                        # 1) Product-aware EMA reverse sampling.
                        self.sample_and_log(
                            milestone
                        )

                        # 2) Checkpoint at the same milestone.
                        self.save(
                            milestone
                        )

                pbar.set_description(
                    f"loss: {total_loss:.4f}"
                )

                pbar.update(
                    1
                )

        self.writer.close()

        accelerator.print(
            "I2SB training complete"
        )
