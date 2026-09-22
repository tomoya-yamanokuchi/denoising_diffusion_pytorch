"""
I2SB method builder.

This builder keeps the existing training architecture intact and adds I2SB as
a parallel training method. The existing ConditionalImageDiffusionBuilder is
reused for network construction (UNet / DiT), while I2SB-specific behavior is
isolated to the dataset, diffusion process, and trainer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from app.wiring.method.ConditionalImageDiffusionBuilder import (
    ConditionalImageDiffusionBuilder,
)


def make_beta_schedule(
    n_timestep: int = 1000,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
) -> np.ndarray:
    """
    Beta schedule used by the official NVlabs/I2SB implementation.
    """
    betas = (
        torch.linspace(
            linear_start ** 0.5,
            linear_end ** 0.5,
            n_timestep,
            dtype=torch.float64,
        )
        ** 2
    )
    return betas.cpu().numpy()


class I2SBImageDiffusionBuilder(ConditionalImageDiffusionBuilder):
    """
    Builder for Image-to-Image Schroedinger Bridge training.

    Reuses the parent class for:
        - build_unet()
        - build_dit()
        - build_model()
        - _maybe_to_device()

    Overrides only:
        - build_dataset()
        - build_method()
        - build_trainer()
    """

    def __init__(
        self,
        cfg: DictConfig,
        artifact_static_root: str,
    ):
        super().__init__(
            cfg=cfg,
            artifact_static_root=artifact_static_root,
        )

    def build_dataset(self) -> Any:
        """
        Build the I2SB-specific paired dataset.

        Each sample contains:
            x0   : full slice image
            x1   : exterior-only slice image
            cond : masked x0 used as network conditioning
            mask : binary observation mask
        """
        from denoising_diffusion_pytorch.data_loader.i2sb_cond_image_data_loader import (
            I2SBCondImageDataset,
        )

        self.dataset = I2SBCondImageDataset(
            cfg=self.cfg,
            image_size=self.cfg.dataset.image_size,
        )

        return self.dataset


    def build_dit(self):
        from denoising_diffusion_pytorch.models.experimental.dit_i2sb import DiT

        model = DiT(
            dim=self.cfg.inferencer.network.dim,
            depth=self.cfg.inferencer.network.depth,
            heads=self.cfg.inferencer.network.heads,
            dim_head=self.cfg.inferencer.network.dim_head,
            patch_size=self.cfg.inferencer.network.patch_size,
        )

        self.model = self._maybe_to_device(model)
        # import ipdb; ipdb.set_trace()
        return self.model


    def build_method(self) -> Any:
        """
        Build the I2SB diffusion process using the same symmetric beta
        schedule construction as the official NVlabs/I2SB implementation.
        """
        from denoising_diffusion_pytorch.models.i2sb_diffusion import (
            I2SBDiffusion,
        )

        diffusion_cfg = self.cfg.inferencer.diffusion

        interval = int(diffusion_cfg.interval)
        beta_max = float(diffusion_cfg.beta_max)

        if interval <= 1:
            raise ValueError(
                f"I2SB `interval` must be > 1, but got {interval}."
            )

        if interval % 2 != 0:
            raise ValueError(
                "The current I2SB implementation uses the official symmetric "
                f"schedule and expects an even `interval`, got {interval}."
            )

        if beta_max <= 0.0:
            raise ValueError(
                f"I2SB `beta_max` must be positive, but got {beta_max}."
            )

        betas = make_beta_schedule(
            n_timestep=interval,
            linear_end=beta_max / interval,
        )

        betas = np.concatenate(
            [
                betas[: interval // 2],
                np.flip(betas[: interval // 2]),
            ]
        )

        if len(betas) != interval:
            raise RuntimeError(
                f"Unexpected beta schedule length: {len(betas)} != {interval}"
            )

        self.method = I2SBDiffusion(
            betas=betas,
            device=str(self.cfg.device),
        )

        return self.method

    def build_trainer(self) -> Any:
        """
        Build the I2SB-specific trainer.

        Expected trainer interface:

            Trainer(
                model=...,
                diffusion=...,
                dataset=...,
                results_folder=...,
                **trainer_cfg,
            )
        """
        from denoising_diffusion_pytorch.trainer.i2sb_conditional_image_trainer import (
            Trainer,
        )

        self.trainer = Trainer(
            model=self.model,
            diffusion=self.method,
            dataset=self.dataset,
            results_folder=str(self.artifact_static_root),
            **self.cfg.inferencer.trainer,
        )

        return self.trainer
