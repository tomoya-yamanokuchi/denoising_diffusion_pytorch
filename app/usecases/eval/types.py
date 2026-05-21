# denoising_diffusion_pytorch/eval/types.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


from .episode.episode_paths import EpisodePaths
from .episode.episode_image_writer import EpisodeImageWriter
from .episode.episode_artifact_manager import EpisodeArtifactManager

from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import dismantling_env
from denoising_diffusion_pytorch.policy.planning.action_planner import ActionPlanner
from denoising_diffusion_pytorch.policy.planning.action_definition.action_candidates import ActionCandidates


@dataclass(frozen=True)
class Envs:
    eval  : dismantling_env  # dismantling_env (evaluation usage with full observation)
    policy: dismantling_env  # dismantling_env (partial observation for policy)

@dataclass(frozen=True)
class InitialGlobalActionRange:
    start: int
    stop : int

@dataclass(frozen=True)
class CaseContext:
    name                         : str
    dataset_dir                  : str
    initial_global_action_range  : InitialGlobalActionRange
    mesh_components              : Any
    envs                         : Envs
    obs_model                    : Any

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
    actions              : List[Any]
    observations         : List[Any]
    cutting_error_volumes: List[float]
    infos                : List[Any]
    removal_performance  : List[float]
    intermediate_actions : List[Any]
    last_info            : Optional[Dict[str, Any]] = None

    # ---- execution error logging ----
    planned_actions              : Any = None
    executed_actions             : Any = None
    planned_intermediate_actions : Any = None
    executed_intermediate_actions: Any = None
    execution_error_infos        : Any = None



@dataclass(frozen=True)
class EpisodeRolloutSnapshot:
    steps: Tuple[StepOutcome, ...]  # 不変で「結果」感を出す


from denoising_diffusion_pytorch.env.types import DismantlingStepResult
@dataclass(frozen=True)
class StepOutcome:
    planned_action_candidates : ActionCandidates
    executed_action_candidates: ActionCandidates
    execution_error_info      : Any
    env_result                : DismantlingStepResult

    @property
    def cutting_error_volume(self) -> float:
        return self.env_result.cutting_error_volume

    @property
    def observation(self):
        return self.env_result.observation

    @property
    def info(self):
        return self.env_result.info

    @property
    def obs_z(self):
        return self.env_result.observation.axis_images.z

    @property
    def observation_history(self):
        return self.env_result.observation.observation_history

    @property
    def oracle_obs(self):
        return self.env_result.info.oracle_axis_images

    @property
    def target_removal_rate(self) -> float:
        return self.env_result.info.target_removal_rate

    @property
    def removal_performance(self) -> float:
        return self.env_result.info.removal_performance

    @property
    def last_executed_global_index(self) -> int:
        return self.executed_action_candidates.last.global_index
