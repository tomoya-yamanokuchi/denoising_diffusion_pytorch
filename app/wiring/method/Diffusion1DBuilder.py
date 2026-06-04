# app/wiring/method/Diffusion1DBuilder.py
from __future__ import annotations

from typing import Any
from omegaconf import DictConfig


class Diffusion1DBuilder:
    def __init__(
        self,
        cfg: DictConfig,
        artifact_static_root: str,
    ):
        self.cfg = cfg
        self.artifact_static_root = artifact_static_root

    def build_dataset(self) -> Any:
        from denoising_diffusion_pytorch.data_loader.image_data_loader import Dataset1D

        self.dataset = Dataset1D(
            folder                  = self.cfg.dataset.path,
            image_size              = self.cfg.dataset.image_size,
            grid_3dim               = self.cfg.dataset.grid_3dim,
            is_shuffle              = self.cfg.dataset.is_shuffle,
            augment_horizontal_flip = self.cfg.dataset.horizontal_flip,
            convert_image_to        = self.cfg.dataset.convert_image_to,
        )
        return self.dataset

    def build_model(self) -> Any:
        from denoising_diffusion_pytorch.models.unet_1d import Unet1D

        channels, seq_len = self.dataset[0].shape
        self.seq_len = seq_len

        model = Unet1D(
            dim            = self.cfg.inferencer.network.dim,
            dim_mults      = self.cfg.inferencer.network.dim_mults,
            channels       = channels,
            out_dim         = channels,
            self_condition = self.cfg.inferencer.network.self_condition,
        )
        self.model = self._maybe_to_device(model)
        return self.model

    def build_method(self) -> Any:
        from denoising_diffusion_pytorch.models.diffusion_1d import GaussianDiffusion1D

        method = GaussianDiffusion1D(
            model      = self.model,
            seq_length = self.seq_len,
            **self.cfg.inferencer.diffusion,
        )
        self.method = self._maybe_to_device(method)
        return self.method

    def build_trainer(self) -> Any:
        from denoising_diffusion_pytorch.trainer.diffusion_1d_trainer import Trainer1D

        self.trainer = Trainer1D(
            diffusion_model = self.method,
            dataset         = self.dataset,
            results_folder  = str(self.artifact_static_root),
            **self.cfg.inferencer.trainer,
        )
        return self.trainer

    def _maybe_to_device(self, obj: Any) -> Any:
        dev = str(self.cfg.device)
        if dev and hasattr(obj, "to"):
            return obj.to(dev)
        return obj
