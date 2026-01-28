# denoising_diffusion_pytorch/eval/types.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Envs:
    eval  : Any  # dismantling_env (evaluation usage)
    policy: Any  # dismantling_env (policy internal usage)


@dataclass(frozen=True)
class ActionArtifacts:
    """
    action 推論時に付随して返る「保存・可視化・デバッグ向けの成果物」。
    - EpisodeRunnerは中身を解釈しない（Observerに渡すだけ）
    """
    ensemble_image: Optional[Dict[str, Any]] = None
    # 将来増える例：
    # candidates : Optional[List[Any]] = None
    # scores     : Optional[List[float]] = None
    # heatmap    : Optional[Any] = None


@dataclass(frozen=True)
class EpisodeContext:
    policy     : Any
    envs       : Envs
    episode_dir: Path

    grid_config : Dict[str, Any]
    start_action: Any
    task_step   : int
    ctrl_mode   : str


@dataclass(frozen=True)
class EpisodeResult:
    actions             : List[Any]
    rewards             : List[float]
    infos               : List[Any]
    removal_performance : List[float]
    intermediate_actions: List[Any]
    last_info           : Optional[Dict[str, Any]] = None
