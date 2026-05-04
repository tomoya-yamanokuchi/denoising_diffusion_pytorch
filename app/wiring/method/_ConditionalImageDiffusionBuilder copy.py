# app/wiring/components.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hydra
from omegaconf import DictConfig
# from ..builder.train_builder import TrainBuilder


class ConditionalImageDiffusionBuilder:
    def __init__(self,
            cfg_root            : DictConfig,
            cfg_usecase         : DictConfig,
            artifact_static_root: str
        ):
        self.cfg_root             = cfg_root
        self.cfg_usecase          = cfg_usecase
        self.artifact_static_root = artifact_static_root

    def build_dataset(self) -> Any:
        from denoising_diffusion_pytorch.data_loader.cond_image_data_loader import Cond_image_dataloader
        self.dataset = Cond_image_dataloader(
            cfg        = self.cfg_usecase,
            image_size = self.cfg_usecase.dataset.image_size,
        )

    def build_unet(self):
        from denoising_diffusion_pytorch.models.unet_2d_simple_devel2 import Unet
        model = Unet(
            dim            = self.cfg_root.network.dim,
            dim_mults      = self.cfg_root.network.dim_mults,
            mask_dim       = self.cfg_usecase.dataset.image_size,
            flash_attn     = self.cfg_root.network.flash_attn,
            self_condition = self.cfg_root.network.self_condition,
        )
        self.model = self._maybe_to_device(model)
        return self.model


    def build_dit(self):
        from denoising_diffusion_pytorch.models.experimental.dit import DiT
        model = DiT(
            dim            = self.cfg_root.network.dim,
            depth          = self.cfg_root.network.depth,
            heads          = self.cfg_root.network.heads,
            dim_head       = self.cfg_root.network.dim_head,
            patch_size     = self.cfg_root.network.patch_size,
        )
        self.model = self._maybe_to_device(model)
        return self.model


    def build_model(self) -> Any:
        # import ipdb; ipdb.set_trace()
        if self.cfg_root.network.name == "unet":
            return self.build_unet()
        elif self.cfg_root.network.name == "dit":
            return self.build_dit()
        else:
            raise ValueError(f"unknown architecture: {self.cfg_root.network.name}")

    def build_method(self) -> Any:
        from denoising_diffusion_pytorch.models.conditional_image_diffusion_cfg_devel2 import GaussianDiffusion
        method = GaussianDiffusion(
            model      = self.model,
            image_size = self.cfg_usecase.dataset.image_size,
            **self.cfg_usecase.inferencer.diffusion,
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
            **self.cfg_usecase.inferencer.trainer,
        )
        return self.trainer


    def _maybe_to_device(self, obj: Any) -> Any:
        # dev = str(self.cfg.device) if "device" in self.cfg else None
        dev = str(self.cfg_usecase.device)

        if dev and hasattr(obj, "to"):
            return obj.to(dev)
        return obj

