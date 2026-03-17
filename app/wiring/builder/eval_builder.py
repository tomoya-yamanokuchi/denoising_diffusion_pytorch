from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig

# --- eval/ 以下で作った部品 ---
# 既存ユーティリティ（あなたのプロジェクト側に合わせて import を調整）

from app.wiring.services.config_validator import ConfigValidator


@dataclass
class EvalBuilder:
    """
    Eval 実行に必要な依存（runner / evaluator / env factory 等）を組み立てる。
    ここでは “インスタンス化” だけに寄せて、評価手順そのものは Evaluator に寄せる。
    """
    cfg: DictConfig

    # run_dir: Optional[Path] = None
    # # build results
    # evaluator        : Any                 = None
    # # episode_runner   : EpisodeRunner       = None
    # env_factory      : EnvFactory | None   = None
    # image_writer     : Any                 = None
    # config_validator : ConfigValidator      = None
    # run_dir_mgr      : RunDirManager       = None

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
        validate_key_config(self.cfg, ["watch", "inferencer", "eval", "env"])

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

    # def build_saved_train_run_dir_resolver(self) -> None:
    #     from app.wiring.loaders.saved_train_run_dir_resolver import SavedTrainRunDirResolver
    #     self.saved_train_run_dir_resolver = SavedTrainRunDirResolver()

    # --------------------------------------------------
    # infra (IO / env)
    # --------------------------------------------------

    def build_eval_cases(self) -> None:
        from app.wiring.services.eval_case_loader import load_eval_case_configs
        self.eval_cases = load_eval_case_configs(self.cfg.eval)

    def build_mesh_components_factory(self) -> None:
        from app.wiring.factories.mesh_component_factory import MeshComponentFactory
        self.mesh_factory = MeshComponentFactory()

    def build_env_factory(self) -> None:
        from app.wiring.factories.env_factory import EnvFactory
        self.env_factory = EnvFactory(grid_config=self.cfg.env.grid)


    def build_obs_model_factory(self):
        from app.wiring.factories.obs_model_factory import VoxelObsModelFactory
        # ---
        self.obs_model_factory = VoxelObsModelFactory(grid_config=self.cfg.env.grid)

    def build_case_context_factory(self):
        from app.wiring.factories.case_context_factory import CaseContextFactory
        self.case_context_factory = CaseContextFactory(
            env_factory       = self.env_factory,
            obs_model_factory = self.obs_model_factory,
        )

    def build_episode_context_factory(self):
        from app.wiring.factories.episode_context_factory import EpisodeContextFactory
        self.episode_context_factory = EpisodeContextFactory(
            grid_config          = self.cfg.env.grid,
            task_step            = self.cfg.eval.task_step,
            ctrl_mode            = self.cfg.eval.policy_config.ctrl_mode,
            artifact_static_root = self.artifact_static_root,
        )

    def build_episode_runner(self):
        from denoising_diffusion_pytorch.eval.episode_runner import EpisodeRunner
        self.episode_runner = EpisodeRunner()


    def build_trained_model_assets(self) -> None:
        from app.wiring.loaders.saved_run_config_loader import SavedRunConfigLoader
        from app.wiring.loaders.checkpoint_path_resolver import CheckpointPathResolver
        from app.wiring.loaders.trained_model_assets_loader import TrainedModelAssetsLoader
        # ---- build loader ----
        loader = TrainedModelAssetsLoader(
            config_loader            = SavedRunConfigLoader(),
            checkpoint_path_resolver = CheckpointPathResolver(),
        )
        # ---- load data ----
        self.trained_model_assets = loader.load(
            run_dir      = self.cfg.eval.train_run_dir,
            epoch        = getattr(self.cfg.eval, "epoch", "latest"),
            device       = str(self.cfg.device),
        )

    def build_policy_assets(self) -> None:
        from ..types.policy_assets import PolicyAssets
        self.policy_assets = PolicyAssets(
            trained_assets = self.trained_model_assets,
            policy_config  = self.cfg.eval.policy_config,
        )

    def build_policy_factory(self):
        from ..factories.policy_factory import PolicyFactory
        self.policy_factory = PolicyFactory(assets = self.policy_assets)

    def build_orchestrator(self):
        from app.usecases.eval.eval_orchestrator import EvalOrchestrator
        self.eval_orchestrator = EvalOrchestrator(self)

    # --------------------------------------------------
    def build_all(self) -> "EvalContext":
        # --- config ---
        self.validate_config_top()
        self.set_config_root_as_usecase_root()
        self.validate_config_usecase()

        # --- dir ---
        self.build_run_dir_manager()
        self.build_run_dir()

        # ---- inferencer ----
        self.build_trained_model_assets()
        self.build_policy_assets()
        self.build_policy_factory()

        self.build_eval_cases()
        self.build_mesh_components_factory()
        self.build_env_factory()
        self.build_obs_model_factory()

        self.build_case_context_factory()

        # --- episode ---
        self.build_episode_context_factory()
        self.build_episode_runner()

        self.build_orchestrator()

        return self
