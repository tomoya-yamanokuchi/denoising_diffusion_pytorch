# denoising_diffusion_pytorch/eval/types.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple



@dataclass(frozen=True)
class MacroAction:
    indices: Tuple[int, ...]

    @property
    def last_atomic(self) -> int:
        return self.indices[-1]

@dataclass(frozen=True)
class ActionArtifacts:
    ensemble_image: Optional[dict[str, Any]] = None
    debug_info    : Optional[dict[str, Any]] = None

@dataclass(frozen=True)
class ActionPlan:
    action   : MacroAction
    artifacts: ActionArtifacts



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
    control     : ControlConfig
    inference   : InferenceConfig
    segmentation: SegmentationConfig
    decision    : DecisionConfig



import numpy as np

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AxisCost:
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray


@dataclass(frozen=True)
class SegmentationCost:
    blue: AxisCost
    red: AxisCost
    yellow: AxisCost


@dataclass(frozen=True)
class AxisCostEnsemble:
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray


@dataclass(frozen=True)
class SegmentationCostEnsemble:
    blue: AxisCostEnsemble
    red: AxisCostEnsemble
    yellow: AxisCostEnsemble
