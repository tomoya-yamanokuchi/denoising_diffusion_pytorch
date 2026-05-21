# denoising_diffusion_pytorch/policy/planning/execution_error/no_error_model.py
from __future__ import annotations

from ..action_definition.action_candidates import ActionCandidates
from .types import ExecutionErrorInfo


class NoActionExecutionErrorModel:
    """
    No-op execution error model.

    Used for the original evaluation setting.
    planned_candidates == executed_candidates.
    """

    def apply(
        self,
        planned_candidates: ActionCandidates,
    ) -> tuple[ActionCandidates, ExecutionErrorInfo]:
        boundary = planned_candidates.last

        info = ExecutionErrorInfo(
            enabled       = False,
            mode          = "none",
            max_abs_shift = 0,
            axis          = planned_candidates.axis,
            sampled_shift = 0,
            applied_shift = 0,
            # ---
            planned_boundary_local_index   = boundary.local_index,
            executed_boundary_local_index  = boundary.local_index,
            planned_boundary_global_index  = boundary.global_index,
            executed_boundary_global_index = boundary.global_index,
            # ---
            planned_global_indices         = planned_candidates.global_indices,
            executed_global_indices        = planned_candidates.global_indices,
        )

        return planned_candidates, info
