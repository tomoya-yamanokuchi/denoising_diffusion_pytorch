from __future__ import annotations

from denoising_diffusion_pytorch.policy.types import AxisCostVector, AxisCostSet, SliceCandidates
from .candidate_coordinator import CandidateCoordinator


class ActionCandidateBuildingCoordinator(CandidateCoordinator):
    def __init__(
        self,
        candidate_builder,
        expected_side_length: int,
    ):
        self.candidate_builder    = candidate_builder
        self.expected_side_length = expected_side_length

    def build(
        self,
        axis_costs: AxisCostSet | None,
        observation_history: dict[int, dict],
    ) -> SliceCandidates:
        if axis_costs is None:
            raise ValueError(
                "ActionCandidateBuildingCoordinator requires axis_costs, "
                "but got None."
            )

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
            z=built["z"],
            x=built["x"],
            y=built["y"],
        )
