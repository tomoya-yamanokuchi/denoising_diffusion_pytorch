from __future__ import annotations

from typing import Dict

from denoising_diffusion_pytorch.policy.types import AxisCostSet, SliceCandidates
from .candidate_coordinator import CandidateCoordinator
from ..action_definition.action_candidates import ActionCandidates


class FullActionSpaceCandidateCoordinator(CandidateCoordinator):
    """
    Build candidates from the full voxel-indexed action space.

    This coordinator ignores axis_costs and uses only observation_history
    to remove already executed actions.
    """

    def __init__(self, expected_side_length: int):
        self.expected_side_length = expected_side_length

    def build(
        self,
        axis_costs: AxisCostSet | None,
        observation_history: Dict[int, dict],
    ) -> SliceCandidates:
        return SliceCandidates(
            z=self._build_axis_candidates("z", observation_history),
            x=self._build_axis_candidates("x", observation_history),
            y=self._build_axis_candidates("y", observation_history),
        )

    def _build_axis_candidates(
        self,
        axis: str,
        observation_history: Dict[int, dict],
    ) -> ActionCandidates | None:
        candidates = ActionCandidates.from_local_indices(
            axis=axis,
            local_indices=tuple(range(self.expected_side_length)),
            side_length=self.expected_side_length,
        )

        if candidates is None:
            return None

        return candidates.prune_by_observation_history(observation_history)
