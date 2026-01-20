# app/wiring/components.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hydra
from omegaconf import DictConfig
from ..builder import Builder


class ConditionalImageDiffusionBuilder:
    def __init__(self, builder: Builder):
        self.builder = builder
        self.cfg     = builder.cfg


    def build_model(self):
        from denoising_diffusion_pytorch.models.unet_2d_simple_devel2 import Unet
        model = Unet(
            dim            = self.cfg.model.dim,
            dim_mults      = self.cfg.model.dim_mults,
            mask_dim       = self.cfg.dataset.image_size,
            flash_attn     = self.cfg.model.flash_attn,
            self_condition = self.cfg.model.self_condition,
        )
        self.model = self._maybe_to_device(model)
        return self.model


    def build_method(self) -> Any:
        from denoising_diffusion_pytorch.models.conditional_image_diffusion_cfg_devel2 import GaussianDiffusion
        method = GaussianDiffusion(
            model      = self.model,
            image_size = self.cfg.dataset.image_size,
            **self.cfg.diffusion,
        )
        self.method = self._maybe_to_device(method)
        return self.method


    def build_trainer(self) -> Any:
        from denoising_diffusion_pytorch.trainer.diffusion_conditional_image_trainer import Trainer

        self.trainer = Trainer(
            diffusion_model = self.method,
            dataset = self.builder.dataset,
            **self.cfg.trainer,
        )

        return self.trainer

    # def build_evaluator(self, algorithm: Any, dataset: Any) -> Any:
    #     return hydra.utils.instantiate(
    #         self.cfg.evaluator,
    #         algorithm=algorithm,
    #         dataset=dataset,
    #         results_folder=self.ctx.run_dir,
    #     )

    def _maybe_to_device(self, obj: Any) -> Any:
        dev = str(self.cfg.device) if "device" in self.cfg else None
        if dev and hasattr(obj, "to"):
            return obj.to(dev)
        return obj

