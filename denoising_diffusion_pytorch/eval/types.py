# denoising_diffusion_pytorch/eval/types.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Envs:
    eval  : Any  # dismantling_env (evaluation usage with full observation)
    policy: Any  # dismantling_env (partial observation for policy)

@dataclass(frozen=True)
class CaseContext:
    name            : str
    dataset_dir     : str
    start_action_idx: List[int]
    mesh_components : Any
    envs            : Envs
    obs_model       : Any

@dataclass(frozen=True)
class EpisodeContext:
    case           : CaseContext
    policy         : Any
    grid_config    : Dict[str, Any]
    task_step      : int
    ctrl_mode      : str

@dataclass(frozen=True)
class EpisodeResult:
    actions             : List[Any]
    rewards             : List[float]
    infos               : List[Any]
    removal_performance : List[float]
    intermediate_actions: List[Any]
    last_info           : Optional[Dict[str, Any]] = None


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
