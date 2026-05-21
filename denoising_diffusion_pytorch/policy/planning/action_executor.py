from __future__ import annotations

from dataclasses import dataclass

from ...env.types import DismantlingStepResult
from ...env.voxel_cut_sim_v1 import dismantling_env
from ..types import ActionCandidates
from app.usecases.eval.types import StepOutcome

from .execution_error import (
    ActionExecutionErrorModel,
    NoActionExecutionErrorModel,
)


@dataclass
class ActionExecutor:
    execution_error_model: ActionExecutionErrorModel | None = None

    def __post_init__(self):
        if self.execution_error_model is None:
            self.execution_error_model = NoActionExecutionErrorModel()

    def execute(
        self,
        env: dismantling_env,
        action_candidates: ActionCandidates,
    ) -> StepOutcome:

        planned_candidates = action_candidates

        executed_candidates, execution_error_info = self.execution_error_model.apply(
            planned_candidates
        )

        step_result = None

        macro_cutting_error_volume = 0.0

        for action_index in executed_candidates:
            step_result = env.step(
                action_idx=action_index.global_index
            )
            macro_cutting_error_volume += step_result.cutting_error_volume


        if step_result is None:
            raise RuntimeError("No action was executed.")

        return StepOutcome(
            planned_action_candidates=planned_candidates,
            executed_action_candidates=executed_candidates,
            execution_error_info=execution_error_info,
            env_result = DismantlingStepResult(
                observation          = step_result.observation,
                cutting_error_volume = macro_cutting_error_volume,
                done                 = step_result.done,
                info                 = step_result.info,
            )
        )
