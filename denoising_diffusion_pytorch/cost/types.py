# denoising_diffusion_pytorch/eval/types.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class AxisCost:
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray


@dataclass(frozen=True)
class SegmentationCost:
    blue  : AxisCost
    red   : AxisCost
    yellow: AxisCost


@dataclass(frozen=True)
class AxisCostEnsemble:
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray


@dataclass(frozen=True)
class SegmentationCostEnsemble:
    blue  : AxisCostEnsemble
    red   : AxisCostEnsemble
    yellow: AxisCostEnsemble


@dataclass(frozen=True)
class AxisDecisionCost:
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray


@dataclass(frozen=True)
class SegmentationDecisionCost:
    blue  : AxisDecisionCost
    red   : AxisDecisionCost
    yellow: AxisDecisionCost
