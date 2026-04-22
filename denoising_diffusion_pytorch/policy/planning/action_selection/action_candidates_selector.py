from __future__ import annotations

from denoising_diffusion_pytorch.policy.planning.action_definition.action_candidates import ActionCandidates
import numpy as np
from typing import Dict

from .selection_policy import SelectionPolicy
from ...types import (
    AxisCostSet,
    SliceSelectionResult,
)
from ..candidate_building.action_candidate_building_coordinator import ActionCandidateBuildingCoordinator


class ActionCandidatesSelector:
    """
    Coordinate axis-wise candidate building and final slice-range selection.
    """

    def __init__(
        self,
        candidate_coordinator: ActionCandidateBuildingCoordinator,
        selection_policy     : SelectionPolicy,
    ):
        self.candidate_coordinator = candidate_coordinator
        self.selection_policy      = selection_policy

    def select(
        self,
        axis_costs         : AxisCostSet,
        observation_history: Dict[int, dict],
    ) -> SliceSelectionResult:

        slice_range_candidates_across_axes = self.candidate_coordinator.build(
            axis_costs          = axis_costs,
            observation_history = observation_history,
        )

        optimal_selected_slice_range = self.selection_policy.choose(slice_range_candidates_across_axes)

        if optimal_selected_slice_range is None:
            optimal_selected_slice_range = self._build_fallback_candidates(
                side_length = len(axis_costs.x),
            )

        return SliceSelectionResult(
            optimal_selected_slice_range       = optimal_selected_slice_range,
            slice_range_candidates_across_axes = slice_range_candidates_across_axes,
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
