from typing import Dict
from .axis_candidate_range_builder import AxisCandidateRangeBuilder
from ...types import AxisCostVector, AxisCostSet, SliceCandidates


class ActionCandidateBuildingCoordinator:
    def __init__(
        self,
        candidate_builder   : AxisCandidateRangeBuilder,
        expected_side_length: int,
    ):
        self.candidate_builder    = candidate_builder
        self.expected_side_length = int(expected_side_length)

    def build(
        self,
        axis_costs         : AxisCostSet,
        observation_history: Dict[int, dict],
    ) -> SliceCandidates:
        built = {}
        for axis, cost in axis_costs.items():
            axis_cost = AxisCostVector(
                axis                 = axis,
                values               = cost,
                expected_side_length = self.expected_side_length,
            )

            built[axis] = self.candidate_builder.build(
                axis_cost           = axis_cost,
                observation_history = observation_history,
            )

        return SliceCandidates(
            x = built["x"],
            y = built["y"],
            z = built["z"],
        )
