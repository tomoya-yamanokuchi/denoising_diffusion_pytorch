from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class AxisImages:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def for_axis(self, axis: str) -> np.ndarray:
        if axis == "x":
            return self.x
        if axis == "y":
            return self.y
        if axis == "z":
            return self.z
        raise ValueError(f"Unsupported axis: {axis}")


@dataclass(frozen=True)
class DismantlingObservation:
    axis_images        : AxisImages
    observation_history: dict[int, dict]


@dataclass(frozen=True)
class DismantlingInfo:
    oracle_axis_images  : AxisImages
    observation_history : dict[int, dict]
    action_table        : dict[int, dict[str, Any]]
    target_removal_rate : float
    removal_performance : float
    remaining_vol       : float
    target_remaining_vol: float


@dataclass(frozen=True)
class DismantlingStepResult:
    observation: DismantlingObservation
    reward     : float
    done       : bool
    info       : DismantlingInfo
