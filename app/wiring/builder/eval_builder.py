from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig



@dataclass
class EvalBuilder:
    """
    Eval 実行に必要な依存（runner / evaluator / env factory 等）を組み立てる。
    ここでは “インスタンス化” だけに寄せて、評価手順そのものは Evaluator に寄せる。
    """
    cfg: DictConfig

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
        # import ipdb; ipdb.set_trace()

    # --------------------------------------------------
    # infra (IO / env)
    # --------------------------------------------------
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
            ctrl_mode            = self.cfg.eval.policy.control.mode,
            artifact_static_root = self.artifact_static_root,
        )


    def build_episode_result_writer(self):
        from app.usecases.eval.episode.episode_result_writer import EpisodeResultWriter
        self.episode_result_writer = EpisodeResultWriter()


    def build_action_executer(self):
        from denoising_diffusion_pytorch.policy.planning.action_executor import ActionExecutor
        self.action_executer = ActionExecutor()

    def build_step_observer(self):
        from app.usecases.eval.episode.episode_step_observer import EpisodeStepObserver
        self.step_observer = EpisodeStepObserver(verbose = True)

    def build_episode_runner(self):
        from app.usecases.eval.episode.episode_runner import EpisodeRunner
        self.episode_runner = EpisodeRunner(
            action_executor = self.action_executer,
            step_observer   = self.step_observer,
        )


    def build_trained_model_assets_loader_factory(self) -> None:
        from app.wiring.loaders.saved_run_config_loader import SavedRunConfigLoader
        from app.wiring.loaders.checkpoint_path_resolver import CheckpointPathResolver
        from app.wiring.loaders.trained_model_assets_loader_factory import (
            TrainedModelAssetsLoaderFactory,
        )

        self.trained_model_assets_loader_factory = TrainedModelAssetsLoaderFactory(
            config_loader            = SavedRunConfigLoader(),
            checkpoint_path_resolver = CheckpointPathResolver(),
        )


    def build_trained_model_assets(self) -> None:
        # ----
        infer_model = str(self.cfg.eval.policy.infer_model)
        loader      = self.trained_model_assets_loader_factory.create(infer_model)
        # -----
        self.trained_model_assets = loader.load(
            run_dir     = self.cfg.eval.train_run_dir,
            epoch       = getattr(self.cfg.eval, "epoch", "latest"),
            device      = str(self.cfg.device),
            infer_model = infer_model,
        )


    def build_policy_config(self) -> None:
        from app.wiring.mappers.policy_config_mapper import build_policy_config
        self.policy_config = build_policy_config(
            cfg_policy             = self.cfg.eval.policy,
            voxel_grid_side_length = self.cfg.env.grid.side_length,
        )


    def build_policy_assets(self) -> None:
        from ..types.policy_assets import PolicyAssets
        self.policy_assets = PolicyAssets(
            trained_assets = self.trained_model_assets,
            policy_config  = self.policy_config
        )

    def build_policy_factory(self):
        from ..factories.policy_factory import PolicyFactory
        self.policy_factory = PolicyFactory(assets = self.policy_assets)

    def build_action_planner_factory(self):
        from ..factories.action_planner_factory import ActionPlannerFactory
        from denoising_diffusion_pytorch.policy.planning.initial_action_provider import InitialActionProvider
        self.action_planner_factory = ActionPlannerFactory(
            initial_action_provider = InitialActionProvider(
                voxel_grid_side_length = self.cfg.env.grid.side_length
            ),
        )

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
        self.build_trained_model_assets_loader_factory()
        self.build_trained_model_assets()
        self.build_policy_config()
        self.build_policy_assets()
        self.build_policy_factory()

        self.build_mesh_components_factory()
        self.build_env_factory()
        self.build_obs_model_factory()
        self.build_case_context_factory()

        self.build_action_executer()
        self.build_action_planner_factory()
        self.build_step_observer()

        # --- episode ---
        self.build_episode_context_factory()
        self.build_episode_result_writer()
        self.build_episode_runner()

        self.build_orchestrator()

        return self
