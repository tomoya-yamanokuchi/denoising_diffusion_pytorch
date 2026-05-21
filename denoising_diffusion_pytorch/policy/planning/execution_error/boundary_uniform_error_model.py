# denoising_diffusion_pytorch/policy/planning/execution_error/boundary_uniform_error_model.py
from __future__ import annotations

import numpy as np

from ..action_definition.action_candidates import ActionCandidates
from .types import ExecutionErrorInfo


class BoundaryUniformExecutionErrorModel:
    """
    Boundary-shift execution error model.

    The planner returns an ordered ActionCandidates sequence from outside to inside.
    This model shifts only the inner boundary candidate by a random integer offset:

        executed_boundary = planned_boundary + delta
        delta ~ Uniform{-max_abs_shift, ..., +max_abs_shift}

    Then it reconstructs the action-candidate sequence from the original outside-side
    anchor to the shifted boundary.
    """

    def __init__(
        self,
        max_abs_shift: int,
        seed: int | None = None,
    ):
        if max_abs_shift < 0:
            raise ValueError(f"max_abs_shift must be non-negative: {max_abs_shift}")

        self.max_abs_shift = int(max_abs_shift)
        self._rng = np.random.default_rng(seed)

    def apply(
        self,
        planned_candidates: ActionCandidates,
    ) -> tuple[ActionCandidates, ExecutionErrorInfo]:
        sampled_shift = self._sample_shift()

        executed_candidates = self._build_shifted_candidates(
            planned_candidates=planned_candidates,
            sampled_shift=sampled_shift,
        )

        planned_boundary = planned_candidates.last
        executed_boundary = executed_candidates.last

        info = ExecutionErrorInfo(
            enabled=True,
            mode="boundary_uniform",
            max_abs_shift=self.max_abs_shift,
            axis=planned_candidates.axis,
            sampled_shift=int(sampled_shift),
            applied_shift=int(
                executed_boundary.local_index - planned_boundary.local_index
            ),
            planned_boundary_local_index=planned_boundary.local_index,
            executed_boundary_local_index=executed_boundary.local_index,
            planned_boundary_global_index=planned_boundary.global_index,
            executed_boundary_global_index=executed_boundary.global_index,
            planned_global_indices=planned_candidates.global_indices,
            executed_global_indices=executed_candidates.global_indices,
        )

        return executed_candidates, info

    def _sample_shift(self) -> int:
        if self.max_abs_shift == 0:
            return 0

        return int(
            self._rng.integers(
                low  = -self.max_abs_shift,
                high = self.max_abs_shift,
                endpoint = True,
            )
        )

    def _build_shifted_candidates(
        self,
        planned_candidates: ActionCandidates,
        sampled_shift     : int,
    ) -> ActionCandidates:
        axis        = planned_candidates.axis
        side_length = planned_candidates.side_length

        outside_anchor   = planned_candidates.first.local_index
        planned_boundary = planned_candidates.last.local_index

        direction = self._infer_direction(planned_candidates)

        shifted_boundary = planned_boundary + int(sampled_shift)
        clipped_boundary = int(np.clip(shifted_boundary, 0, side_length - 1))

        if direction > 0:
            executed_boundary = max(outside_anchor, clipped_boundary)
            local_indices = tuple(range(outside_anchor, executed_boundary + 1))
        else:
            executed_boundary = min(outside_anchor, clipped_boundary)
            local_indices = tuple(range(outside_anchor, executed_boundary - 1, -1))

        executed_candidates = ActionCandidates.from_local_indices(
            axis          = axis,
            local_indices = local_indices,
            side_length   = side_length,
        )

        if executed_candidates is None:
            raise RuntimeError(
                "Failed to build executed ActionCandidates. "
                f"axis={axis}, local_indices={local_indices}, side_length={side_length}"
            )

        return executed_candidates

    def _infer_direction(self, candidates: ActionCandidates) -> int:
        first = candidates.first.local_index
        last  = candidates.last.local_index

        if last > first:
            return +1

        if last < first:
            return -1

        midpoint = (candidates.side_length - 1) / 2.0
        return +1 if first <= midpoint else -1
