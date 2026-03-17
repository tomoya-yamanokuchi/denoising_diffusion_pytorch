from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .types import MacroAction
from ..eval.types import StepOutcome
from ..env.voxel_cut_sim_v1 import dismantling_env


@dataclass
class MacroActionExecutor:
    def execute(self,
            env         : dismantling_env,
            macro_action: MacroAction
        ) -> tuple[StepOutcome, dict, dict]:

        cut_cost  = 0.0
        last_obs  = None
        last_info = None

        for atomic in macro_action.indices:
            obs, reward, done, info = env.step(action_idx=atomic)
            cut_cost += reward
            last_obs  = obs
            last_info = info

        outcome = StepOutcome(
            macro_action        = macro_action.indices,
            last_action         = macro_action.last_atomic,
            reward              = cut_cost,
            obs_z               = last_obs["sequential_obs"]["z"],
            target_removal_rate = last_info["target_removal_rate"],
            removal_performance = last_info["removal_performance"],
        )
        return outcome, last_obs, last_info
