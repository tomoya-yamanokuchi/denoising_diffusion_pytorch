from __future__ import annotations

import numpy as np
from typing import Dict

from .slice_range_selection_policy import SliceRangeSelectionPolicy
# from .split_observation_update_factory import SplitObservationUpdateFactory
from .types import (
    AxisCostSet,
    SliceSelectionResult,
)

from .axis_slice_candidate_coordinator import AxisSliceCandidateCoordinator


class AxisSliceRangeSelector:
    """
    Coordinate axis-wise candidate building and final slice-range selection.

    Responsibility
    --------------
    - wrap x/y/z costs into AxisCostVector
    - build out-to-in slice indices for each axis
    - choose the final slice range among x/y/z
    - create split_obs_config update payload
    """

    def __init__(
        self,
        candidate_coordinator: AxisSliceCandidateCoordinator,
        selection_policy     : SliceRangeSelectionPolicy,
        # split_update_factory: SplitObservationUpdateFactory,
    ):
        self.candidate_coordinator = candidate_coordinator
        self.selection_policy      = selection_policy
        # self.split_update_factory = split_update_factory

    def select(
        self,
        axis_costs         : AxisCostSet,
        observation_history: Dict[int, dict],
    ) -> SliceSelectionResult:

        slice_candidates = self.candidate_coordinator.build(
            axis_costs          = axis_costs,
            observation_history = observation_history,
        )

        slice_range       = self.selection_policy.choose(slice_candidates)
        # split_obs_update = self.split_update_factory.build(slice_range)

        import ipdb; ipdb.set_trace()

        return SliceSelectionResult(
            slice_range      = slice_range,
            slice_candidates = slice_candidates,
            # split_obs_update = split_obs_update,
        )
