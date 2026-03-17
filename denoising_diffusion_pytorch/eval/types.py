# denoising_diffusion_pytorch/eval/types.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


from ..env.voxel_cut_sim_v1 import dismantling_env
from .episode_paths import EpisodePaths
from .episode_image_writer import EpisodeImageWriter
from .episode_artifact_manager import EpisodeArtifactManager

# from ..policy.cutting_surface_planner_v9 import cutting_surface_planner
from ..action_plan.action_planner import ActionPlanner

@dataclass(frozen=True)
class Envs:
    eval  : dismantling_env  # dismantling_env (evaluation usage with full observation)
    policy: dismantling_env  # dismantling_env (partial observation for policy)

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
    case            : CaseContext
    action_planner  : ActionPlanner
    grid_config     : Dict[str, Any]
    task_step       : int
    ctrl_mode       : str
    episode_idx     : int
    path            : EpisodePaths
    artifact_manager: EpisodeArtifactManager
    image_writer    : EpisodeImageWriter


@dataclass(frozen=True)
class EpisodeResult:
    actions             : List[Any]
    rewards             : List[float]
    infos               : List[Any]
    removal_performance : List[float]
    intermediate_actions: List[Any]
    last_info           : Optional[Dict[str, Any]] = None


from typing import List, Tuple
import numpy as np
@dataclass(frozen=True)
class StepOutcome:
    macro_action       : tuple[int, ...]  # intermediate_action_l に相当
    last_action        : int              # action_l に相当
    reward             : float            # reward_l
    obs_z              : np.ndarray       # obs_l
    target_removal_rate: float            # info_l
    removal_performance: float            # removal_pref_l

@dataclass(frozen=True)
class EpisodeRolloutSnapshot:
    steps: Tuple[StepOutcome, ...]  # 不変で「結果」感を出す


