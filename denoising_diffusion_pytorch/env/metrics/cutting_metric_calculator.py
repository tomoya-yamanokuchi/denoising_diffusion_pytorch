# denoising_diffusion_pytorch/env/metrics/cutting_metric_calculator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from denoising_diffusion_pytorch.env.metrics.target_color_segmenter import (
    TargetColorSegmenter,
)
from denoising_diffusion_pytorch.env.types import (
    AxisImages,
    DismantlingInfo,
)
from denoising_diffusion_pytorch.utils.pil_utils import color_range_mask


@dataclass(frozen=True)
class DismantlingMetricValues:
    """
    Intermediate metric values for one environment state.

    These values are separated from DismantlingInfo so that the pure metric
    calculation can be tested independently from environment-specific metadata
    such as action_table and observation_history.
    """

    target_removal_rate: float
    part_remaining_rate: float
    part_occupancy_rate: float
    remaining_vol: float
    target_remaining_vol: float
    oracle_target_shape_vol: float


@dataclass
class CuttingMetricCalculator:
    """
    Calculate cutting-related task metrics.

    Responsibilities:
      - count target-colored pixels/voxels on a cut surface
      - calculate step-level normalized cutting error rate
      - calculate state-level target remaining / occupancy metrics

    This class intentionally does not own environment state.
    It only receives images and scalar volumes, then returns metric values.
    """

    target_segmenter: TargetColorSegmenter
    remaining_volume_epsilon: float = 1e-6

    @classmethod
    def create_default(cls) -> "CuttingMetricCalculator":
        """
        Create a metric calculator for the current task setting.

        Current task:
          target part = blue component
        """
        return cls(
            target_segmenter=TargetColorSegmenter.create_default_blue_segmenter()
        )

    def calculate_cutting_error_volume(
        self,
        cut_surface_image: np.ndarray,
    ) -> float:
        """
        Calculate Cutting Error Volume for one cut surface.

        This corresponds to the number of target-colored pixels/voxels included
        in the executed cut surface.
        """
        return self.target_segmenter.count_target_pixels(cut_surface_image)

    def calculate_step_normalized_cutting_error_rate(
        self,
        cutting_error_volume: float,
        oracle_target_shape_vol: float,
    ) -> float:
        """
        Calculate step-level Normalized Cutting Error Rate [%].

        Definition:
            step_normalized_cutting_error_rate
              = cutting_error_volume / oracle_target_shape_vol * 100
        """
        oracle_target_shape_vol = float(oracle_target_shape_vol)

        if oracle_target_shape_vol <= 0:
            return 0.0

        return float(cutting_error_volume) / oracle_target_shape_vol * 100.0

    def calculate_state_metrics(
        self,
        sequential_observation_z: np.ndarray,
        oracle_target_shape_vol: float,
    ) -> DismantlingMetricValues:
        """
        Calculate state-level dismantling metrics from the current z-axis
        sequential observation.

        This reproduces the previous logic in dismantling_env.get_info():

            current_target_removal_vol
              = target-colored volume in current sequential z image

            target_removal_rate
              = current_target_removal_vol / oracle_target_shape_vol * 100

            remaining_vol
              = black/unobserved volume in current sequential z image + epsilon

            target_remaining_vol
              = oracle_target_shape_vol - current_target_removal_vol

            part_occupancy_rate
              = target_remaining_vol / remaining_vol * 100

            part_remaining_rate
              = target_remaining_vol / oracle_target_shape_vol * 100
        """
        oracle_target_shape_vol = float(oracle_target_shape_vol)

        current_target_removal_vol = self.calculate_cutting_error_volume(
            sequential_observation_z
        )

        target_removal_rate = self._safe_percent(
            numerator=current_target_removal_vol,
            denominator=oracle_target_shape_vol,
        )

        remaining_vol = self.calculate_remaining_volume(
            sequential_observation_z
        )

        target_remaining_vol = (
            oracle_target_shape_vol - current_target_removal_vol
        )

        part_occupancy_rate = self._safe_percent(
            numerator=target_remaining_vol,
            denominator=remaining_vol,
        )

        part_remaining_rate = self._safe_percent(
            numerator=target_remaining_vol,
            denominator=oracle_target_shape_vol,
        )

        return DismantlingMetricValues(
            target_removal_rate=target_removal_rate,
            part_remaining_rate=part_remaining_rate,
            part_occupancy_rate=part_occupancy_rate,
            remaining_vol=remaining_vol,
            target_remaining_vol=target_remaining_vol,
            oracle_target_shape_vol=oracle_target_shape_vol,
        )

    def calculate_remaining_volume(
        self,
        sequential_observation_z: np.ndarray,
    ) -> float:
        """
        Count unobserved / remaining region volume.

        In the existing implementation, black pixels are treated as remaining
        / unobserved region.

        Previous implementation:
            target_mask = np.asarray([0.0, 0.0, 0.0])
            mask_image = color_range_mask(...)
            remaining_vol = mask_image.mean(2).sum() + 1e-6
        """
        black_mask_config = self._black_mask_config()

        mask_image = color_range_mask(
            sequential_observation_z,
            black_mask_config,
        )

        if mask_image.ndim == 2:
            remaining_vol = float(mask_image.sum())
        elif mask_image.ndim == 3:
            remaining_vol = float(mask_image.mean(axis=2).sum())
        else:
            raise ValueError(
                "Remaining mask must be either 2D or 3D, "
                f"but got shape: {mask_image.shape}"
            )

        return remaining_vol + self.remaining_volume_epsilon

    def build_dismantling_info(
        self,
        *,
        oracle_axis_images: AxisImages,
        sequential_observation_z: np.ndarray,
        oracle_target_shape_vol: float,
        observation_history: dict[int, dict],
        action_table: dict[int, dict[str, Any]],
    ) -> DismantlingInfo:
        """
        Build DismantlingInfo from images and environment metadata.

        This method keeps the environment-facing return type unchanged while
        moving metric calculations out of dismantling_env.
        """
        metric_values = self.calculate_state_metrics(
            sequential_observation_z=sequential_observation_z,
            oracle_target_shape_vol=oracle_target_shape_vol,
        )

        return DismantlingInfo(
            oracle_axis_images=oracle_axis_images,
            observation_history=observation_history,
            action_table=action_table,

            target_removal_rate=metric_values.target_removal_rate,
            part_remaining_rate=metric_values.part_remaining_rate,
            part_occupancy_rate=metric_values.part_occupancy_rate,

            remaining_vol=metric_values.remaining_vol,
            target_remaining_vol=metric_values.target_remaining_vol,
            oracle_target_shape_vol=metric_values.oracle_target_shape_vol,
        )

    def _black_mask_config(self) -> dict[str, np.ndarray]:
        target_mask = np.asarray([0.0, 0.0, 0.0], dtype=float)

        return {
            "target_mask": target_mask,
            "target_mask_lb": target_mask - 0.0,
            "target_mask_ub": target_mask + 0.0,
        }

    def _safe_percent(
        self,
        numerator: float,
        denominator: float,
    ) -> float:
        denominator = float(denominator)

        if denominator <= 0:
            return 0.0

        return float(numerator) / denominator * 100.0
