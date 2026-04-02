from __future__ import annotations

from dataclasses import dataclass

from ...env.types import DismantlingStepResult
from ...env.voxel_cut_sim_v1 import dismantling_env
from ..types import ActionCandidates
from ...eval.types import StepOutcome


@dataclass
class ActionExecutor:
    def execute(self,
            env              : dismantling_env,
            action_candidates: ActionCandidates,
        ) -> StepOutcome:

        cut_cost  = 0.0

        for action_index in action_candidates:
            step_result = env.step(
                action_idx = action_index.global_index
            )
            cut_cost += step_result.reward

        return StepOutcome(
            executed_action_candidates = action_candidates,
            env_result = DismantlingStepResult(
                observation = step_result.observation,
                reward      = cut_cost,
                done        = step_result.done,
                info        = step_result.info,
            )
        )
