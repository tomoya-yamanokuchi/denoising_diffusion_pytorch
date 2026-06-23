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
    Segment target-colored pixels/voxels from RGB-like arrays.

    In the current dismantling task, the blue component is treated as
    the target part. This class centralizes the color thresholding logic
    that was previously embedded in dismantling_env.calculate_cutting_error_volume().
    """

    color_range: TargetColorRange

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

    def build_bool_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Build a boolean target mask from an RGB-like array.

        This method is intended for downstream volume visualization/logging where
        a true boolean mask is easier to store and compose than the legacy
        3-channel mask returned by color_range_mask(). It accepts any array whose
        last dimension is RGB, for example:

            - 2D slice image:        (H, W, 3)
            - 3D voxel color grid:   (D, H, W, 3)
            - flattened voxel color: (N, 3)

        Returns:
            np.ndarray:
                Boolean mask with shape image.shape[:-1].
        """
        self._validate_rgb_array(image)

        rgb = np.asarray(image, dtype=float)
        lb = np.asarray(self.color_range.target_mask_lb, dtype=float)
        ub = np.asarray(self.color_range.target_mask_ub, dtype=float)

        return np.all((rgb >= lb) & (rgb <= ub), axis=-1)

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

    def _validate_rgb_array(self, image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "image must be a numpy.ndarray, "
                f"but got {type(image)}"
            )

        if image.ndim < 2:
            raise ValueError(
                "image must have at least 2 dimensions with RGB channels on the last axis, "
                f"but got shape: {image.shape}"
            )

        if image.shape[-1] != 3:
            raise ValueError(
                "image must have 3 RGB channels on the last axis, "
                f"but got shape: {image.shape}"
            )
