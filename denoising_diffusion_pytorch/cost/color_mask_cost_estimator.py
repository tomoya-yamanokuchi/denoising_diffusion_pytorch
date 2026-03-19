# denoising_diffusion_pytorch/policy/color_mask_cost_estimator.py

from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import numpy as np

from denoising_diffusion_pytorch.action_plan.types import ColorMaskConfig, AxisCost
from ..action_plan.types import SegmentationConfig, SegmentationCost
from denoising_diffusion_pytorch.utils.pil_utils import color_range_mask
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler


class ColorMaskCostEstimator:
    """
    policy が保持する segmentation 設定に基づいて、
    2D image から色別の slice cost を推定する。
    """

    def __init__(self, obs_model: voxel_cut_handler, segmentation: SegmentationConfig):
        self.obs_model    = obs_model
        self.segmentation = segmentation

    def estimate_all(self, image: np.ndarray) -> SegmentationCost:
        cast_images = self._cast_to_axis_images(image)

        return SegmentationCost(
            blue   = self._estimate_from_cast_images(cast_images, self.segmentation.blue),
            red    = self._estimate_from_cast_images(cast_images, self.segmentation.red),
            yellow = self._estimate_from_cast_images(cast_images, self.segmentation.yellow),
        )


    def _estimate_from_cast_images(self,
        cast_images: Dict[str, np.ndarray],
        mask_config: ColorMaskConfig,
    ) -> AxisCost:
        mask_images = self._build_color_mask_images(cast_images, mask_config)
        return self._aggregate_slice_costs(mask_images)


    def _resolve_mask_config(self, color_name: str) -> ColorMaskConfig:
        if color_name == "blue":
            return self.segmentation.blue
        if color_name == "red":
            return self.segmentation.red
        if color_name == "yellow":
            return self.segmentation.yellow
        raise ValueError(f"Unknown color_name: {color_name}")

    def _cast_to_axis_images(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        self.obs_model.cast_2d_image_to_box_color(img=image, config={"axis": "z"})

        return {
            "image_x": self.obs_model.get_2d_image(axis="x"),
            "image_y": self.obs_model.get_2d_image(axis="y"),
            "image_z": self.obs_model.get_2d_image(axis="z"),
        }


    def _build_color_mask_images(
        self,
        images: Dict[str, np.ndarray],
        mask_config: ColorMaskConfig,
    ) -> Dict[str, np.ndarray]:
        mask_config_dict = asdict(mask_config)
        return {
            axis_name: color_range_mask(axis_image, mask_config_dict)
            for axis_name, axis_image in images.items()
        }

    def _aggregate_slice_costs(self, mask_images: Dict[str, np.ndarray]) -> AxisCost:
        voxel_handler = self.obs_model.voxel_hander

        cost_x = (
            voxel_handler
            .get_2d_image_to_mini_batch_image(mask_images["image_x"], permute="z")
            .sum(3).sum(1).sum(1)
        )
        cost_y = (
            voxel_handler
            .get_2d_image_to_mini_batch_image(mask_images["image_y"], permute="z")
            .sum(3).sum(1).sum(1)
        )
        cost_z = (
            voxel_handler
            .get_2d_image_to_mini_batch_image(mask_images["image_z"], permute="z")
            .sum(3).sum(1).sum(1)
        )

        return AxisCost(
            x_axis = cost_x,
            y_axis = cost_y,
            z_axis = cost_z,
        )
