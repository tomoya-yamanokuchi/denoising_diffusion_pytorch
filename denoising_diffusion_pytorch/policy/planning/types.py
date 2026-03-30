from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np



@dataclass(frozen=True)
class AxisCostVector:
    axis                : str
    values              : np.ndarray
    expected_side_length: int

    def __post_init__(self) -> None:
        values = np.asarray(self.values).reshape(-1)

        if self.axis not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported axis: {self.axis}")

        if len(values) != self.expected_side_length:
            raise ValueError(
                f"Cost length must equal expected_side_length. "
                f"Got len(values)={len(values)}, "
                f"expected_side_length={self.expected_side_length}."
            )

        object.__setattr__(self, "values", values)

    @property
    def side_length(self) -> int:
        return len(self.values)



@dataclass(frozen=True)
class AxisCostSet:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def items(self):
        return (
            ("z", self.z),
            ("x", self.x),
            ("y", self.y),
        )


@dataclass(frozen=True)
class ActiveRange:
    start_index: int
    end_index  : int


@dataclass(frozen=True)
class LocalAxisCandidates:
    top   : Tuple[int, ...]
    bottom: Tuple[int, ...]

@dataclass(frozen=True)
class GlobalAxisCandidates:
    top   : Tuple[int, ...]
    bottom: Tuple[int, ...]


@dataclass(frozen=True)
class SliceCandidates:
    x: Tuple[int, ...]
    y: Tuple[int, ...]
    z: Tuple[int, ...]


@dataclass(frozen=True)
class SliceSelectionResult:
    slice_range     : Tuple[int, ...]
    slice_candidates: SliceCandidates
    split_obs_update: Optional[dict]


@dataclass(frozen=True)
class OutToInSliceIndices:
    values: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.values)

    def to_list(self) -> list[int]:
        return list(self.values)

    def is_empty(self) -> bool:
        return len(self.values) == 0




