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
    x: ActionCandidates
    y: ActionCandidates
    z: ActionCandidates


@dataclass(frozen=True)
class SliceSelectionResult:
    optimal_selected_slice_range      : ActionCandidates
    slice_range_candidates_across_axes: SliceCandidates


@dataclass(frozen=True)
class OutToInSliceIndices:
    values: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.values)

    def to_list(self) -> list[int]:
        return list(self.values)

    def is_empty(self) -> bool:
        return len(self.values) == 0


from .planning.action_definition.action_candidates import ActionCandidates
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ActionArtifacts:
    ensemble_image: Optional[dict[str, Any]] = None
    debug_info    : Optional[dict[str, Any]] = None

@dataclass(frozen=True)
class ActionPlan:
    action_candidates: ActionCandidates
    artifacts        : ActionArtifacts

'''
    config
'''

@dataclass
class ControlConfig:
    mode: str

@dataclass
class InferenceConfig:
    model           : str
    guidance_scale  : float
    sample_image_num: int

@dataclass
class ColorMaskConfig:
    target_mask   : list[float]
    target_mask_lb: list[float]
    target_mask_ub: list[float]

@dataclass
class SegmentationConfig:
    blue  : ColorMaskConfig
    red   : ColorMaskConfig
    yellow: ColorMaskConfig

@dataclass
class DecisionParamConfig:
    ucb_lb: float = 0.5

@dataclass
class DecisionConfig:
    mode : str
    param: DecisionParamConfig

@dataclass
class PolicyConfig:
    control               : ControlConfig
    inference             : InferenceConfig
    segmentation          : SegmentationConfig
    decision              : DecisionConfig
    voxel_grid_side_length: int


import torch


@dataclass(frozen=True)
class PlanningPolicyInput:
    normalized_cond: torch.Tensor | None
