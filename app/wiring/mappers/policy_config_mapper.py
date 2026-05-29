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
    SelectionConfig,
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


    selection_cfg = getattr(cfg_policy, "selection", None)

    return PolicyConfig(
        control=ControlConfig(
            mode = str(cfg_policy.control.mode),
        ),
        inference=InferenceConfig(
            model              = str(cfg_policy.inference.model),
            guidance_scale     = float(cfg_policy.inference.guidance_scale),
            sample_image_num   = int(cfg_policy.inference.sample_image_num),
            sampling_timesteps = int(cfg_policy.inference.sampling_timesteps),
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
        selection = SelectionConfig(
            mode=str(getattr(selection_cfg, "mode", "longest")),
            seed=(
                None
                if selection_cfg is None or getattr(selection_cfg, "seed", None) is None
                else int(selection_cfg.seed)
            ),
        ),
        voxel_grid_side_length = voxel_grid_side_length,
    )
