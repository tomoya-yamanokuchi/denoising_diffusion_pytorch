# denoising_diffusion_pytorch/env/metrics/target_color_segmenter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from denoising_diffusion_pytorch.utils.pil_utils import color_range_mask


@dataclass(frozen=True)
class TargetColorRange:
    """
    Color range used to segment the target part.

    The current task treats the blue component as the target part.
    """

    target_mask: np.ndarray
    target_mask_lb: np.ndarray
    target_mask_ub: np.ndarray

    def to_legacy_config(self) -> dict[str, np.ndarray]:
        """
        Convert to the config format expected by color_range_mask().
        """
        return {
            "target_mask": self.target_mask,
            "target_mask_lb": self.target_mask_lb,
            "target_mask_ub": self.target_mask_ub,
        }


@dataclass(frozen=True)
class TargetColorSegmenter:
    """
    Segment target-colored pixels/voxels from a 2D slice image.

    In the current dismantling task, the blue component is treated as
    the target part. This class centralizes the color thresholding logic
    that was previously embedded in dismantling_env.calculate_cutting_error_volume().
    """

    color_range: TargetColorRange

    @classmethod
    def create_default_blue_segmenter(cls) -> "TargetColorSegmenter":
        """
        Create the default segmenter for the current task setting.

        This reproduces the previous implementation:

            target_mask_b = np.asarray([0.2, 0.8, 0.8])
            lb = target_mask_b - np.asarray([0.1, 0.1, 0.1])
            ub = target_mask_b + np.asarray([0.7, 0.2, 0.2])
        """
        target_mask = np.asarray([0.2, 0.8, 0.8], dtype=float)

        return cls(
            color_range=TargetColorRange(
                target_mask=target_mask,
                target_mask_lb=target_mask - np.asarray([0.1, 0.1, 0.1], dtype=float),
                target_mask_ub=target_mask + np.asarray([0.7, 0.2, 0.2], dtype=float),
            )
        )

    @classmethod
    def from_legacy_config(
        cls,
        config: dict[str, Any],
    ) -> "TargetColorSegmenter":
        """
        Create a segmenter from the legacy config format.

        Expected keys:
            target_mask
            target_mask_lb
            target_mask_ub
        """
        return cls(
            color_range=TargetColorRange(
                target_mask=np.asarray(config["target_mask"], dtype=float),
                target_mask_lb=np.asarray(config["target_mask_lb"], dtype=float),
                target_mask_ub=np.asarray(config["target_mask_ub"], dtype=float),
            )
        )

    def build_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Build a target mask image.

        Args:
            image:
                2D RGB-like slice image.
                Expected shape: (H, W, C), usually C=3.

        Returns:
            np.ndarray:
                Mask image returned by color_range_mask().
                In the current legacy implementation this is treated as
                a 3-channel mask, and target pixels are counted by
                mask.mean(axis=2).sum().
        """
        self._validate_image(image)

        return color_range_mask(
            image,
            self.color_range.to_legacy_config(),
        )

    def count_target_pixels(self, image: np.ndarray) -> float:
        """
        Count target-colored pixels/voxels in the given slice image.

        This reproduces the previous logic:

            mask_image = color_range_mask(...)
            target_volume = mask_image.mean(2).sum()

        Args:
            image:
                2D RGB-like slice image.

        Returns:
            float:
                Number of target-colored pixels/voxels in the slice.
        """
        mask_image = self.build_mask(image)

        if mask_image.ndim == 2:
            return float(mask_image.sum())

        if mask_image.ndim == 3:
            return float(mask_image.mean(axis=2).sum())

        raise ValueError(
            "Target mask must be either 2D or 3D, "
            f"but got shape: {mask_image.shape}"
        )

    def _validate_image(self, image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "image must be a numpy.ndarray, "
                f"but got {type(image)}"
            )

        if image.ndim != 3:
            raise ValueError(
                "image must have shape (H, W, C), "
                f"but got shape: {image.shape}"
            )

        if image.shape[2] != 3:
            raise ValueError(
                "image must have 3 channels, "
                f"but got shape: {image.shape}"
            )
