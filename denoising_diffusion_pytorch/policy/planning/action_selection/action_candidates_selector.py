from __future__ import annotations

from denoising_diffusion_pytorch.policy.planning.action_definition.action_candidates import ActionCandidates
from denoising_diffusion_pytorch.policy.types import (
    AxisCostSet,
    SliceSelectionResult,
)
from denoising_diffusion_pytorch.policy.planning.candidate_building.candidate_coordinator import (
    CandidateCoordinator,
)
from .selection_policy import SelectionPolicy


class ActionCandidatesSelector:
    """
    Coordinate candidate building and final candidate selection.
    """

    def __init__(
        self,
        candidate_coordinator: CandidateCoordinator,
        selection_policy: SelectionPolicy,
    ):
        self.candidate_coordinator = candidate_coordinator
        self.selection_policy = selection_policy

    def select(
        self,
        axis_costs: AxisCostSet | None,
        observation_history: dict[int, dict],
    ) -> SliceSelectionResult:
        slice_range_candidates_across_axes = self.candidate_coordinator.build(
            axis_costs=axis_costs,
            observation_history=observation_history,
        )

        optimal_selected_slice_range = self.selection_policy.choose(
            slice_range_candidates_across_axes
        )

        if optimal_selected_slice_range is None:
            optimal_selected_slice_range = self._build_fallback_candidates(
                side_length=self._infer_side_length(slice_range_candidates_across_axes),
            )

        return SliceSelectionResult(
            optimal_selected_slice_range=optimal_selected_slice_range,
            slice_range_candidates_across_axes=slice_range_candidates_across_axes,
        )


    def _build_fallback_candidates(
        self,
        side_length: int,
    ) -> ActionCandidates:
        fallback = ActionCandidates.from_global_indices(
            global_indices = [0], # fallback global index。旧実装との整合性のため、0固定。
            side_length   = side_length,
        )
        if fallback is None:
            raise RuntimeError("Failed to construct legacy fallback ActionCandidates.")
        return fallback
