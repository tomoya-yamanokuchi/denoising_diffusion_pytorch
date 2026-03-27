from __future__ import annotations

import numpy as np

from .axis_candidate_range_builder import AxisCandidateRangeBuilder
from .slice_range_selection_policy import SliceRangeSelectionPolicy
# from .split_observation_update_factory import SplitObservationUpdateFactory
from .types import (
    AxisCostVector,
    SliceCandidates,
    SliceSelectionResult,
)


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
        candidate_builder   : AxisCandidateRangeBuilder,
        selection_policy    : SliceRangeSelectionPolicy,
        # split_update_factory: SplitObservationUpdateFactory,
        expected_side_length: int,
    ):
        self.candidate_builder    = candidate_builder
        self.selection_policy     = selection_policy
        # self.split_update_factory = split_update_factory
        self.expected_side_length = int(expected_side_length)

    def select(
        self,
        *,
        cost_x: np.ndarray,
        cost_y: np.ndarray,
        cost_z: np.ndarray,
        observation_history: dict[int, dict],
    ) -> SliceSelectionResult:
        axis_cost_x = AxisCostVector(
            axis="x",
            values=cost_x,
            expected_side_length=self.expected_side_length,
        )
        axis_cost_y = AxisCostVector(
            axis="y",
            values=cost_y,
            expected_side_length=self.expected_side_length,
        )
        axis_cost_z = AxisCostVector(
            axis="z",
            values=cost_z,
            expected_side_length=self.expected_side_length,
        )

        slice_candidates = SliceCandidates(
            x = self.candidate_builder.build(
                axis_cost=axis_cost_x,
                observation_history=observation_history,
            ),
            y = self.candidate_builder.build(
                axis_cost=axis_cost_y,
                observation_history=observation_history,
            ),
            z = self.candidate_builder.build(
                axis_cost=axis_cost_z,
                observation_history=observation_history,
            ),
        )

        slice_range      = self.selection_policy.choose(slice_candidates)
        # split_obs_update = self.split_update_factory.build(slice_range)

        import ipdb; ipdb.set_trace()

        return SliceSelectionResult(
            slice_range      = slice_range,
            slice_candidates = slice_candidates,
            split_obs_update = split_obs_update,
        )
