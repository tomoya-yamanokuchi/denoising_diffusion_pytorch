# app/wiring/method/VAEACBuilder.py
from __future__ import annotations

from typing import Any
from omegaconf import DictConfig


class VAEACBuilder:
    def __init__(
        self,
        cfg: DictConfig,
        artifact_static_root: str,
    ):
        self.cfg = cfg
        self.artifact_static_root = artifact_static_root

    def build_dataset(self) -> Any:
        from denoising_diffusion_pytorch.data_loader.vaeac_data_loader import (
            VAEAC_dataloader,
        )

        self.dataset = VAEAC_dataloader(
            cfg        = self.cfg,
            image_size = self.cfg.dataset.image_size,
        )
        return self.dataset

    def build_model(self) -> Any:
        from denoising_diffusion_pytorch.models.vaeac.vaeac import EncoderDecoder

        model = EncoderDecoder(cfg=self.cfg.inferencer)
        self.model = self._maybe_to_device(model)
        return self.model

    def build_method(self) -> Any:
        # VAEACは diffusion wrapper を持たず、生modelがmethod相当
        self.method = self.model
        return self.method

    def build_trainer(self) -> Any:
        from denoising_diffusion_pytorch.trainer.vaeac_trainer import Trainer

        self.trainer = Trainer(
            model=self.model,
            dataset=self.dataset,
            cfg=self.cfg.inferencer,
            dataset_cfg=self.cfg.dataset,
            savepath=str(self.artifact_static_root),
        )
        return self.trainer

    def _maybe_to_device(self, obj: Any) -> Any:
        dev = str(self.cfg.device)
        if dev and hasattr(obj, "to"):
            return obj.to(dev)
        return obj
