# app/wiring/builder/train_builder.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type
from omegaconf import DictConfig, OmegaConf
from app.wiring.method.protocols import TrainMethodBuilder
from denoising_diffusion_pytorch.utils.omega_config_util import select_str
from app.wiring.method.ConditionalImageDiffusionBuilder import ConditionalImageDiffusionBuilder
from app.wiring.method.VAEACBuilder import VAEACBuilder
from app.wiring.services.run_dir_manager import RunDirManager
from app.wiring.services.config_validator import ConfigValidator


_METHOD_BUILDERS: Dict[str, Type[TrainMethodBuilder]] = {
    "conditional_image_diffusion": ConditionalImageDiffusionBuilder,
    "vaeac"                      : VAEACBuilder,
}


@dataclass
class TrainBuilder:
    """
    学習用の構築を担当。
    build_* の結果を self.xxx に保持して、Director が Components を作れるようにする。
    """
    # --- input ----
    cfg       : DictConfig
    # --- output ----
    run_dir   : Optional[Path] = None
    exp_name  : Optional[str]  = None
    dataset   : Any            = None
    model     : Any            = None
    method    : Any            = None
    trainer   : Any            = None
    image_size: Optional[int]  = None

    def build_config_validator(self):
        self.validator = ConfigValidator()

    def validate_config(self) -> None:
        # train に最低限必要なキー（trainerなどが重要）
        self.validator.require_keys(self.cfg, ["device", "dataset", "model", "trainer"])

    def set_important_params(self):
        self.image_size = int(OmegaConf.select(self.cfg, "dataset.image_size") or 0)
        self.device     = str(OmegaConf.select(self.cfg, "device") or "cuda:0")

    def build_run_dir_manager(self):
        from denoising_diffusion_pytorch.utils.RunDirPlanner import RunDirPlanner
        from denoising_diffusion_pytorch.utils.RunDirInitializer import RunDirInitializer
        self.run_dir_mgr = RunDirManager(
            planner     = RunDirPlanner.from_cfg(self.cfg),
            initializer = RunDirInitializer(),
        )

    def plan_and_initialize_run_dir(self) -> None:
        self.run_dir, self.exp_name = self.run_dir_mgr.plan(self.cfg)
        self.run_dir_mgr.init(self.cfg, self.run_dir, self.exp_name)

    def build_method(self) -> None:
        """
        - Sub builder に差分を閉じ込める。
        - Sub builder には最低限、build_dataset/build_model/build_method/build_trainer というメソッドがある想定。
        """
        name = select_str(self.cfg, "name", default="")
        if name not in _METHOD_BUILDERS:
            raise ValueError(f"Unknown train method: {name}. Known: {list(_METHOD_BUILDERS.keys())}")

        Sub = _METHOD_BUILDERS[name]
        sub: TrainMethodBuilder = Sub(self)  # IntelliSenseが効く

        self.dataset    = sub.build_dataset()
        self.model      = sub.build_model()
        self.method     = sub.build_method()
        self.trainer    = sub.build_trainer()


    def build_all(self) -> None:
        self.build_config_validator()
        self.validate_config()
        self.set_important_params()
        self.build_run_dir_manager()
        self.plan_and_initialize_run_dir()
        self.build_method()
