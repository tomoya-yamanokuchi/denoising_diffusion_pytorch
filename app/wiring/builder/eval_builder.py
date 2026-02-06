from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig

# --- eval/ 以下で作った部品 ---
from denoising_diffusion_pytorch.eval.types import Envs
from denoising_diffusion_pytorch.eval.episode_runner import EpisodeRunner
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
    episode_runner   : EpisodeRunner | None = None
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
        self.env_factory = EnvFactory()


    # def build_image_writer(self) -> None:
    #     from denoising_diffusion_pytorch.eval. import ImageWriter  # 既存の ImageWriter を想定
    #     self.image_writer = ImageWriter()

    def build_episode_runner(self) -> None:
        ctrl_mode   = str(self.cfg.eval.policy_config.ctrl_mode)
        init_action = make_action_init_strategy(ctrl_mode)

        # 保存する/しないは Observer の差し替えで吸収（runner に if を持ち込まない）
        save_images = bool(getattr(self.cfg.eval, "save_images", True))
        observer = ImageObserver(self.image_writer) if save_images else NullObserver()

        self.episode_runner = EpisodeRunner(
            observer           = observer,
            # oracle_updater     = DefaultOracleUpdater(),
            # step_executor      = DefaultStepExecutor(),
            init_action        = init_action,
            next_action_policy = DefaultNextActionPolicy(),
        )

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

    def build_evaluator(self) -> None:
        """
        Evaluator は「ケース列挙→episode_runner.run(ctx)」を回すだけのオブジェクト。
        """
        # eval 設定を evaluator に渡す
        self.evaluator = Evaluator(
            cfg_eval=self.cfg.eval,
            run_dir=self.run_dir,
            env_factory=self.env_factory,
            episode_runner=self.episode_runner,
            policy=self.method,  # method=policy として扱う
            device=str(self.cfg.device),
        )

    # --------------------------------------------------
    def build_all(self) -> None:
        # ---- config root / validation
        self.validate_config_top()
        self.set_config_root_as_usecase_root()
        self.validate_config_usecase()

        # ---- run dir
        self.build_run_dir_manager()
        self.build_run_dir()

        # ---- cases / factories
        self.build_eval_cases()
        self.build_env_factory()
        # ----
        # self.build_image_writer()
        # self.build_method()
        self.build_episode_runner()
        self.build_evaluator()
