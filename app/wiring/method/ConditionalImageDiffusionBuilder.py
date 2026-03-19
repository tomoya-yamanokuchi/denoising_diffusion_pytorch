# app/wiring/components.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hydra
from omegaconf import DictConfig
# from ..builder.train_builder import TrainBuilder


class ConditionalImageDiffusionBuilder:
    def __init__(self, cfg: DictConfig, artifact_static_root: str):
        self.cfg                  = cfg
        self.artifact_static_root = artifact_static_root

    def build_dataset(self) -> Any:
        from denoising_diffusion_pytorch.data_loader.cond_image_data_loader import Cond_image_dataloader
        self.dataset = Cond_image_dataloader(
            cfg        = self.cfg,
            image_size = self.cfg.dataset.image_size,
        )

    def build_model(self):
        from denoising_diffusion_pytorch.models.unet_2d_simple_devel2 import Unet
        model = Unet(
            dim            = self.cfg.inferencer.network.dim,
            dim_mults      = self.cfg.inferencer.network.dim_mults,
            mask_dim       = self.cfg.dataset.image_size,
            flash_attn     = self.cfg.inferencer.network.flash_attn,
            self_condition = self.cfg.inferencer.network.self_condition,
        )
        self.model = self._maybe_to_device(model)
        return self.model


    def build_method(self) -> Any:
        from denoising_diffusion_pytorch.models.conditional_image_diffusion_cfg_devel2 import GaussianDiffusion
        method = GaussianDiffusion(
            model      = self.model,
            image_size = self.cfg.dataset.image_size,
            **self.cfg.inferencer.diffusion,
        )
        self.method = self._maybe_to_device(method)
        return self.method


    def build_trainer(self) -> Any:
        from denoising_diffusion_pytorch.trainer.diffusion_conditional_image_trainer import Trainer
        # ---
        self.trainer = Trainer(
            diffusion_model = self.method,
            dataset         = self.dataset,
            results_folder  = str(self.artifact_static_root),
            **self.cfg.inferencer.trainer,
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
        # dev = str(self.cfg.device) if "device" in self.cfg else None
        dev = str(self.cfg.device)

        if dev and hasattr(obj, "to"):
            return obj.to(dev)
        return obj

