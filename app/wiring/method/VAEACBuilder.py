# app/wiring/components.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hydra
# from omegaconf import DictConfig
# from .. builder.train_builder import TrainBuilder

class VAEACBuilder:
    def __init__(self, train_builder):
        self.builder = train_builder
        self.cfg     = train_builder.cfg

    def build_dataset(self) -> Any:
        from denoising_diffusion_pytorch.data_loader.cond_image_data_loader import Cond_image_dataloader
        self.dataset = Cond_image_dataloader(
            cfg        = self.cfg,
            image_size = self.cfg.dataset.image_size,
        )

    def build_model(self):
        from denoising_diffusion_pytorch.models.vaeac.vaeac import EncoderDecoder
        model = EncoderDecoder(
            cfg = self.cfg,
        )
        self.model = self._maybe_to_device(model)
        return self.model


    def build_method(self) -> Any:
        return self.model


    def build_trainer(self) -> Any:
        from denoising_diffusion_pytorch.trainer.vaeac_trainer import Trainer
        self.trainer = Trainer(
            model   = self.model,
            dataset = self.dataset,
            cfg     = self.cfg,
            savepath = self.builder.run_dir,
        )
        return self.trainer

    def build_evaluator(self, algorithm: Any, dataset: Any) -> Any:
        from denoising_diffusion_pytorch.trainer.vaeac_trainer import Trainer
        self.trainer = Trainer(
            model   = self.model,
            dataset = self.dataset,
            cfg     = self.cfg,
            savepath = self.builder.run_dir,
        )
        return self.trainer

    def _maybe_to_device(self, obj: Any) -> Any:
        dev = str(self.cfg.device) if "device" in self.cfg else None
        if dev and hasattr(obj, "to"):
            return obj.to(dev)
        return obj

