# app/wiring/mappers/policy_config_mapper.py

from omegaconf import DictConfig

from denoising_diffusion_pytorch.policy.types import (
    ColorMaskConfig,
    ControlConfig,
    DecisionConfig,
    DecisionParamConfig,
    InferenceConfig,
    PolicyConfig,
    SegmentationConfig,
)


def _build_color_mask_config(cfg_mask: DictConfig) -> ColorMaskConfig:
    return ColorMaskConfig(
        target_mask    = list(cfg_mask.target_mask),
        target_mask_lb = list(cfg_mask.target_mask_lb),
        target_mask_ub = list(cfg_mask.target_mask_ub),
    )


def build_policy_config(
        cfg_policy            : DictConfig,
        voxel_grid_side_length: int,
    ) -> PolicyConfig:
    return PolicyConfig(
        control=ControlConfig(
            mode = str(cfg_policy.control.mode),
        ),
        inference=InferenceConfig(
            model            = str(cfg_policy.inference.model),
            guidance_scale   = float(cfg_policy.inference.guidance_scale),
            sample_image_num = int(cfg_policy.inference.sample_image_num),
        ),
        segmentation=SegmentationConfig(
            blue   = _build_color_mask_config(cfg_policy.segmentation.blue),
            red    = _build_color_mask_config(cfg_policy.segmentation.red),
            yellow = _build_color_mask_config(cfg_policy.segmentation.yellow),
        ),
        decision=DecisionConfig(
            mode  = str(cfg_policy.decision.mode),
            param = DecisionParamConfig(
                ucb_lb = float(cfg_policy.decision.param.ucb_lb),
            ),
        ),
        voxel_grid_side_length = voxel_grid_side_length,
    )
