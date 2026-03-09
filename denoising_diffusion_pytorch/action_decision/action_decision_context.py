from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


from denoising_diffusion_pytorch.policy.cutting_surface_planner_v9 import cutting_surface_planner

@dataclass(frozen=True)
class ActionDecisionContext:
    step_idx      : int
    policy        : cutting_surface_planner
    env_for_policy: Any

    obs_z              : Optional[Any]
    observation_history: dict

    last_executed_action: Optional[int]


    initial_action: list[int]
    grid_config   : dict
