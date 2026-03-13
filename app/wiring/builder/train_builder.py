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
    # # --- output ----
    # run_dir   : Optional[Path] = None
    # exp_name  : Optional[str]  = None
    # dataset   : Any            = None
    # model     : Any            = None
    # method    : Any            = None
    # trainer   : Any            = None
    # image_size: Optional[int]  = None

    # --------------------------------------------------
    # 1. config validation
    # --------------------------------------------------
    def validate_config_top(self) -> None:
        from app.wiring.services.validate_key_config import validate_key_config
        validate_key_config(self.cfg, ["usecase"])

    def set_config_root_as_usecase_root(self):
        self.usecase = self.cfg.usecase.name
        self.cfg     = self.cfg.usecase

    def validate_config_usecase(self) -> None:
        from app.wiring.services.validate_key_config import validate_key_config
        validate_key_config(self.cfg, ["watch", "method", "dataset"])

    def build_config_artifact_writer(self) -> None:
        from app.wiring.services.train_config_artifact_writer import TrainConfigArtifactWriter
        self.config_artifact_writer = TrainConfigArtifactWriter()

    def save_train_config_artifacts(self) -> None:
        self.config_artifact_writer.write(
            cfg=self.cfg,
            artifact_static_root=self.artifact_static_root,
        )

    # --------------------------------------------------
    # 2. directory management
    # --------------------------------------------------
    def build_run_dir_manager(self) -> None:
        from app.wiring.services.run_dir_manager import RunDirManager
        from denoising_diffusion_pytorch.utils.RunDirPlanner import RunDirPlanner
        from denoising_diffusion_pytorch.utils.RunDirInitializer import RunDirInitializer
        self.run_dir_mgr = RunDirManager(
            planner     = RunDirPlanner.from_cfg(self.cfg),
            initializer = RunDirInitializer(),
        )

    def build_run_dir(self) -> None:
        run_dir, _exp_name = self.run_dir_mgr.plan(self.cfg)
        self.run_dir_mgr.init(self.cfg, run_dir, _exp_name)
        self.artifact_static_root = run_dir
        import ipdb; ipdb.set_trace()



    def build_method(self) -> None:
        """
        - Sub builder に差分を閉じ込める。
        - Sub builder には最低限、build_dataset/build_model/build_method/build_trainer というメソッドがある想定。
        """
        name = self.cfg.method.name
        if name not in _METHOD_BUILDERS:
            raise ValueError(f"Unknown train method: {name}. Known: {list(_METHOD_BUILDERS.keys())}")

        Sub = _METHOD_BUILDERS[name]
        sub: TrainMethodBuilder = Sub(
            cfg = self.cfg,
            artifact_static_root = self.artifact_static_root,
        )  # IntelliSenseが効く

        self.dataset    = sub.build_dataset()
        self.model      = sub.build_model()
        self.method     = sub.build_method()
        self.trainer    = sub.build_trainer()

    def build_orchestrator(self):
        from app.usecases.train.train_orchestrator import TrianOrchestrator
        self.train_orchestrator = TrianOrchestrator(self)


    def build_all(self) -> "TrainContext":
        # --- load config ---
        self.validate_config_top()
        self.set_config_root_as_usecase_root()
        self.validate_config_usecase()

        # --- plan dir ---
        self.build_run_dir_manager()
        self.build_run_dir()

        # --- save config ---
        self.build_config_artifact_writer()
        self.save_train_config_artifacts()

        self.build_method()
        self.build_orchestrator()

        return self
