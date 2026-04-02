from __future__ import annotations

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

        slice_candidates = self.candidate_coordinator.build(
            axis_costs          = axis_costs,
            observation_history = observation_history,
        )

        slice_range = self.selection_policy.choose(slice_candidates)

        return SliceSelectionResult(
            slice_range      = slice_range,
            slice_candidates = slice_candidates,
        )
