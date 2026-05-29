# denoising_diffusion_pytorch/policy/planning/candidate_building/candidate_coordinator.py
from __future__ import annotations

from typing import Protocol

from denoising_diffusion_pytorch.policy.types import AxisCostSet, SliceCandidates


class CandidateCoordinator(Protocol):
    """
    Interface for building axis-wise action candidates.

    Implementations may build candidates from:
      - diffusion-based cost maps
      - full action space
      - oracle cost maps
      - other future planning sources
    """

    def build(
        self,
        axis_costs: AxisCostSet | None,
        observation_history: dict[int, dict],
    ) -> SliceCandidates:
        ...
