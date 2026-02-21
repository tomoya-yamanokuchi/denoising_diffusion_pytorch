from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig

# --- eval/ 以下で作った部品 ---
from denoising_diffusion_pytorch.eval.types import Envs

from denoising_diffusion_pytorch.eval.observers import ImageObserver, NullObserver
# from denoising_diffusion_pytorch.eval.oracle_updater import DefaultOracleUpdater
# from denoising_diffusion_pytorch.eval.step_executor import DefaultStepExecutor
from denoising_diffusion_pytorch.eval.next_action import DefaultNextActionPolicy
from denoising_diffusion_pytorch.eval.strategies import make_action_init_strategy

# 既存ユーティリティ（あなたのプロジェクト側に合わせて import を調整）
from denoising_diffusion_pytorch.utils.RunDirPlanner import RunDirPlanner
from denoising_diffusion_pytorch.utils.RunDirInitializer import RunDirInitializer

from app.wiring.services.run_dir_manager import RunDirManager
from app.wiring.services.config_validator import ConfigValidator

# env / policy / eval orchestration は既存に合わせる（例）


# from denoising_diffusion_pytorch.eval.evaluator import Evaluator      # 後述の Evaluator を想定


# from denoising_diffusion_pytorch.eval.episode_runner import EpisodeRunner

@dataclass
class EvalBuilder:
    """
    Eval 実行に必要な依存（runner / evaluator / env factory 等）を組み立てる。
    ここでは “インスタンス化” だけに寄せて、評価手順そのものは Evaluator に寄せる。
    """
    cfg: DictConfig

    run_dir: Optional[Path] = None

    # build results
    evaluator        : Any                 = None
    # episode_runner   : EpisodeRunner       = None
    env_factory      : EnvFactory | None   = None
    image_writer     : Any                 = None
    config_validator : ConfigValidator      = None
    run_dir_mgr      : RunDirManager       = None

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
        validate_key_config(self.cfg, ["watch", "method", "eval", "env"])

    # --------------------------------------------------
    # 2. directory management
    # --------------------------------------------------
    def build_run_dir_manager(self) -> None:
        self.run_dir_mgr = RunDirManager(
            planner     = RunDirPlanner.from_cfg(self.cfg),
            initializer = RunDirInitializer(),
        )

    def build_run_dir(self) -> None:
        self.run_dir, _exp_name = self.run_dir_mgr.plan(self.cfg)
        self.run_dir_mgr.init(self.cfg, self.run_dir, _exp_name)

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
            grid_config = self.cfg.env.grid,
            task_step   = self.cfg.eval.task_step,
            ctrl_mode   = self.cfg.eval.policy_config.ctrl_mode,
        )

    def build_episode_runner(self):
        from denoising_diffusion_pytorch.eval.episode_runner import EpisodeRunner
        self.episode_runner = EpisodeRunner()

    def build_policy(self):
        self.policy = None

    def build_orchestrator(self):
        from app.usecases.eval_orchestrator import EvalOrchestrator
        self.eval_orchestrator = EvalOrchestrator(self)

    # def build_context(self):
    #     from app.wiring.types.eval_context import EvalContext
    #     return EvalContext(
    #         cfg                 = self.cfg,
    #         run_dir             = self.run_dir,
    #         eval_cases          = self.eval_cases,
    #         mesh_factory        = self.mesh_factory,
    #         case_ctx_factory    = self.case_ctx_factory,
    #         episode_ctx_factory = self.episode_ctx_factory,
    #         episode_runner      = self.episode_runner,
    # )

    # --------------------------------------------------
    # method / evaluator (project-specific)
    # --------------------------------------------------
    # def build_method(self) -> None:
    #     """
    #     ここはあなたの既存 Builder 群（VAEACBuilder 等）に寄せてOK。
    #     ここでは「policy を作って cfg.eval.policy_config を注入した状態」を返せればよい。
    #     """
    #     # 例：既存の method builder を呼ぶ（あなたの実装に合わせて）
    #     # self.method = self.method_builder.build(self.cfg.method, device=str(self.cfg.device), ...)
    #     from app.wiring.method_factory import build_method  # 仮：あなたのプロジェクトに合わせて
    #     self.method = build_method(self.cfg.method, device=str(self.cfg.device))

    # --------------------------------------------------
    def build_all(self) -> "EvalContext":
        self.validate_config_top()
        self.set_config_root_as_usecase_root()
        self.validate_config_usecase()

        self.build_run_dir_manager()
        self.build_run_dir()

        self.build_eval_cases()
        self.build_mesh_components_factory()
        self.build_env_factory()
        self.build_obs_model_factory()

        self.build_case_context_factory()
        self.build_episode_context_factory()
        self.build_episode_runner()

        self.build_policy()

        self.build_orchestrator()

        return self
