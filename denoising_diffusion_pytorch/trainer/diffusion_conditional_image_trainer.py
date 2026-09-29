"""
Conditional Diffusion trainer using the shared ProductAwareLogger.

Expected shared utility:
    denoising_diffusion_pytorch/utils/product_aware_logger.py

Expected dataset item format:
    {
        "image":        Tensor[3,H,W] in [-1,1],   # full target
        "observed":     Tensor[3,H,W] in [-1,1],   # conditional image
        "mask":         Tensor[1,H,W] in {0,1},    # 0 observed / 1 hidden
        "product_name": str,
        "image_path":   str,
    }

Product-aware logging layout:
    samples/
      sheetsander/
        static/
          target.png
          cond.png
          mask.png
        pred/
          step_005000.png
          step_010000.png
          ...
      polisher/
        ...
      powercutter/
        ...

Responsibilities of the shared ProductAwareLogger:
- select fixed validation samples per product
- cache each fixed sample once
- save static images once
- save dynamic images at every checkpoint
- organize TensorBoard entries per product

Responsibilities kept in this trainer:
- Conditional Diffusion training
- Conditional Diffusion sampling
- adapting method-specific tensors into ProductLogOutput
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm

from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation
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
        diffusion_model,
        dataset,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_horizontal_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=2000,
        # Kept for backward compatibility with existing configs.
        num_samples=25,
        # Shared product-aware logging.
        samples_per_product=1,
        results_folder="./results",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        convert_image_to=None,
        calculate_fid=True,
        inception_block_idx=2048,
        max_grad_norm=1.0,
        num_fid_samples=50000,
        save_best_and_latest_only=False,
        num_workers=8,
        **unused_kwargs,
    ):
        super().__init__()

        # ==============================================================
        # Accelerator
        # ==============================================================

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=(
                mixed_precision_type
                if amp
                else "no"
            ),
        )

        # ==============================================================
        # Model
        # ==============================================================

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.image_size = diffusion_model.image_size

        is_ddim_sampling = (
            diffusion_model.is_ddim_sampling
        )

        # Retained only for compatibility with the old trainer.
        if not exists(
            convert_image_to
        ):
            convert_image_to = {
                1: "L",
                3: "RGB",
                4: "RGBA",
            }.get(
                self.channels
            )

        self.convert_image_to = (
            convert_image_to
        )

        # ==============================================================
        # Training / logging settings
        # ==============================================================

        self.num_samples = int(
            num_samples
        )

        self.samples_per_product = int(
            samples_per_product
        )

        self.save_and_sample_every = int(
            save_and_sample_every
        )

        self.batch_size = int(
            train_batch_size
        )

        self.gradient_accumulate_every = int(
            gradient_accumulate_every
        )

        self.train_num_steps = int(
            train_num_steps
        )

        self.max_grad_norm = float(
            max_grad_norm
        )

        if self.batch_size <= 0:
            raise ValueError(
                "train_batch_size must be positive."
            )

        if self.gradient_accumulate_every <= 0:
            raise ValueError(
                "gradient_accumulate_every must be positive."
            )

        if (
            self.batch_size
            * self.gradient_accumulate_every
            < 16
        ):
            raise ValueError(
                "effective batch size "
                "(train_batch_size x gradient_accumulate_every) "
                "should be at least 16."
            )

        if self.train_num_steps <= 0:
            raise ValueError(
                "train_num_steps must be positive."
            )

        if self.samples_per_product <= 0:
            raise ValueError(
                "samples_per_product must be positive."
            )

        if len(dataset) < 100:
            raise ValueError(
                "Conditional Diffusion training requires "
                "at least 100 dataset samples."
            )

        if (
            unused_kwargs
            and self.accelerator.is_main_process
        ):
            self.accelerator.print(
                "[ConditionalDiffusion] "
                "Unused trainer config keys:",
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
            data_samples
            * 0.9
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
        # Construct BEFORE wrapping any dataloader with Accelerate.
        # ProductAwareLogger needs the original Subset.indices and
        # dataset.samples metadata.
        # ==============================================================

        self.product_logger = (
            ProductAwareLogger(
                dataset=dataset,
                val_dataset=val_dataset,
                results_folder=results_folder,
                samples_per_product=
                    self.samples_per_product,
                tensorboard_prefix=
                    "conditional_diffusion",
            )
        )

        # ==============================================================
        # Dataloaders
        # ==============================================================

        train_dl = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        # Retained for compatibility / possible FID usage.
        val_dl = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
        )

        # ==============================================================
        # Optimizer
        # ==============================================================

        self.opt = Adam(
            diffusion_model.parameters(),
            lr=train_lr,
            betas=adam_betas,
        )

        # ==============================================================
        # Output directories
        # ==============================================================

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

        # ==============================================================
        # EMA
        # ==============================================================

        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model,
                beta=ema_decay,
                update_every=ema_update_every,
            )

            self.ema.to(
                self.device
            )

        # ==============================================================
        # Accelerate preparation
        # ==============================================================

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

        # ==============================================================
        # Product-aware log summary
        # ==============================================================

        if self.accelerator.is_main_process:
            self.product_logger.print_summary(
                self.accelerator.print
            )

        # ==============================================================
        # FID
        # ==============================================================

        self.calculate_fid = (
            calculate_fid
            and self.accelerator.is_main_process
        )

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires "
                    "many generated samples and may be slow. "
                    "Consider DDIM sampling."
                )

            self.fid_scorer = (
                FIDEvaluation(
                    batch_size=self.batch_size,
                    dl=self.train_dl,
                    sampler=self.ema.ema_model,
                    channels=self.channels,
                    accelerator=self.accelerator,
                    stats_dir=results_folder,
                    device=self.device,
                    num_fid_samples=
                        num_fid_samples,
                    inception_block_idx=
                        inception_block_idx,
                )
            )

        if save_best_and_latest_only:
            if not calculate_fid:
                raise ValueError(
                    "`calculate_fid` must be True when "
                    "`save_best_and_latest_only=True`."
                )

            self.best_fid = 1e10

        self.save_best_and_latest_only = (
            save_best_and_latest_only
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
                "conditional_diffusion",
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
            self.accelerator
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
            self.accelerator
            .is_main_process
        ):
            self.ema.load_state_dict(
                data["ema"]
            )

        if "version" in data:
            self.accelerator.print(
                "loading from version "
                f"{data['version']}"
            )

        if (
            exists(
                self.accelerator.scaler
            )
            and exists(
                data.get(
                    "scaler",
                    None,
                )
            )
        ):
            self.accelerator.scaler.load_state_dict(
                data["scaler"]
            )

    # ==================================================================
    # Visualization helpers
    # ==================================================================

    @staticmethod
    def _to_01(
        x,
    ):
        """
        Project image convention:
            [-1,1] -> [0,1]
        """
        return (
            (
                x
                + 1.0
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
        """
        Binary mask convention:
            0 = observed
            1 = hidden

        Visualization:
            black = observed
            white = hidden
        """
        if (
            mask.shape[1]
            == 1
        ):
            return mask.repeat(
                1,
                3,
                1,
                1,
            )

        return mask

    # ==================================================================
    # Adapter: Conditional Diffusion -> ProductAwareLogger
    # ==================================================================

    @torch.no_grad()
    def _make_product_log_output(
        self,
        fixed_sample,
    ):
        """
        Method-specific adapter.

        The shared logger does not know anything about:
        - DDPM / DDIM
        - classifier-free guidance
        - GaussianDiffusion.sample()

        It only receives visualization-ready tensors in [0,1].
        """

        item = fixed_sample[
            "item"
        ]

        target = (
            item["image"]
            .unsqueeze(0)
            .to(self.device)
        )

        cond = (
            item["observed"]
            .unsqueeze(0)
            .to(self.device)
        )

        binary_mask = (
            item["mask"]
            .unsqueeze(0)
            .to(self.device)
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # In the current project implementation of GaussianDiffusion,
        # sample(..., mask=...) expects the CONDITIONAL IMAGE rather than
        # the binary 0/1 mask. This matches the legacy trainer:
        #
        #     mask = data["observed"]
        #     images = ema_model.sample(batch_size=1, mask=mask)
        #
        # Keep this behavior unchanged here.
        # --------------------------------------------------------------

        pred = (
            self.ema
            .ema_model
            .sample(
                batch_size=1,
                mask=cond,
            )
        )

        # GaussianDiffusion.sample() currently calls self.unnormalize()
        # before returning, so prediction is already [0,1].
        pred_vis = (
            pred.clamp(
                0.0,
                1.0,
            )
        )

        target_vis = (
            self._to_01(
                target
            )
        )

        cond_vis = (
            self._to_01(
                cond
            )
        )

        mask_vis = (
            self._mask_to_rgb(
                binary_mask
            )
            .clamp(
                0.0,
                1.0,
            )
        )

        return ProductLogOutput(
            product_name=
                fixed_sample[
                    "product_name"
                ],
            static={
                "target":
                    target_vis,
                "cond":
                    cond_vis,
                "mask":
                    mask_vis,
            },
            dynamic={
                "pred":
                    pred_vis,
            },
        )

    # ==================================================================
    # Product-aware sampling log
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
            "[ConditionalDiffusion] "
            "start product-aware "
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
            "[ConditionalDiffusion] "
            "product-aware samples saved under "
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

                    img = (
                        data["image"]
                        .to(
                            device,
                            non_blocking=True,
                        )
                    )

                    cond = (
                        data["observed"]
                        .to(
                            device,
                            non_blocking=True,
                        )
                    )

                    with (
                        self.accelerator
                        .autocast()
                    ):
                        loss = self.model(
                            img,
                            mask=cond,
                        )

                        loss = (
                            loss
                            / self.gradient_accumulate_every
                        )

                        total_loss += (
                            loss.item()
                        )

                    self.accelerator.backward(
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

                        # --------------------------------------------------
                        # 1) Product-aware generation / visualization.
                        # --------------------------------------------------
                        self.sample_and_log(
                            milestone
                        )

                        # --------------------------------------------------
                        # 2) Optional FID.
                        # --------------------------------------------------
                        fid_score = None

                        if self.calculate_fid:
                            fid_score = (
                                self.fid_scorer
                                .fid_score()
                            )

                            accelerator.print(
                                "fid_score: "
                                f"{fid_score}"
                            )

                        # --------------------------------------------------
                        # 3) Checkpoint.
                        # --------------------------------------------------
                        if (
                            self.save_best_and_latest_only
                        ):
                            # Constructor guarantees calculate_fid=True.
                            if (
                                self.best_fid
                                > fid_score
                            ):
                                self.best_fid = (
                                    fid_score
                                )

                                self.save(
                                    "best"
                                )

                            self.save(
                                "latest"
                            )

                        else:
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
            "Conditional Diffusion training complete"
        )
