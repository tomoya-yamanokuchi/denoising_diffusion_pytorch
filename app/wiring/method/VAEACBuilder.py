# app/wiring/components.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hydra
from omegaconf import DictConfig
from ..builder import Builder


class VAEACBuilder:
    def __init__(self, builder: Builder):
        self.builder = builder
        self.cfg     = builder.cfg

    def build_dataset(self) -> Any:
        from denoising_diffusion_pytorch.data_loader.cond_image_data_loader import Cond_image_dataloader
        self.dataset = Cond_image_dataloader(
            cfg        = self.cfg,
            image_size = self.builder.image_size,
        )

    def build_model(self):
        from denoising_diffusion_pytorch.models.vaeac.vaeac import EncoderDecoder
        model = EncoderDecoder(
            cfg = self.cfg,
        )
        self.model = self._maybe_to_device(model)
        return self.model


    def build_method(self) -> Any:
        # from denoising_diffusion_pytorch.models.conditional_image_diffusion_cfg_devel2 import GaussianDiffusion
        # method = GaussianDiffusion(
        #     model      = self.model,
        #     image_size = self.cfg.dataset.image_size,
        #     # **self.cfg.diffusion,
        # )
        # self.method = self._maybe_to_device(method)
        # self.method = self.model
        # return self.method
        return


    def build_trainer(self) -> Any:
        from denoising_diffusion_pytorch.trainer.vaeac_trainer import Trainer

        self.trainer = Trainer(
            model   = self.model,
            dataset = self.dataset,
            cfg     = self.cfg,
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

