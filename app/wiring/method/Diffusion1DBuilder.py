# app/wiring/components.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hydra
from omegaconf import DictConfig



class Diffusion1DBuilder:
    def build_dataset(self):
        Dataset1D(
            folder=self.cfg.dataset.path,
            image_size=self.cfg.dataset.image_size,
            grid_3dim=self.cfg.dataset.grid_3dim,
            is_shuffle=self.cfg.dataset.is_shuffle,
            augment_horizontal_flip=self.cfg.dataset.horizontal_flip,
            convert_image_to=self.cfg.dataset.convert_image_to,
        )

    def build_model(self):
        channels, seq_len = self.dataset[0].shape
        Unet1D(
            dim=self.cfg.inferencer.network.dim,
            dim_mults=self.cfg.inferencer.network.dim_mults,
            channels=channels,
            out_dim=channels,
            self_condition=self.cfg.inferencer.network.self_condition,
        )

    def build_method(self):
        GaussianDiffusion1D(
            model=self.model,
            seq_length=seq_len,
            **self.cfg.inferencer.diffusion,
        )

    def build_trainer(self):
        Trainer1D(
            diffusion_model=self.method,
            dataset=self.dataset,
            results_folder=str(self.artifact_static_root),
            **self.cfg.inferencer.trainer,
        )
