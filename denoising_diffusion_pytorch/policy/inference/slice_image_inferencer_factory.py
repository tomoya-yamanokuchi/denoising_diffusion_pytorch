from __future__ import annotations

from denoising_diffusion_pytorch.policy.cutting_surface_planner import PolicyConfig


from .conditional_diffusion_slice_image_inferencer import ConditionalDiffusionSliceImageInferencer
from denoising_diffusion_pytorch.models.conditional_image_diffusion_cfg_devel2 import GaussianDiffusion


from .vaeac_slice_image_inferencer import VaeacSliceImageInferencer
from denoising_diffusion_pytorch.models.vaeac.vaeac import EncoderDecoder

class SliceImageInferencerFactory:
    @staticmethod
    def build(
        inferencer      : EncoderDecoder | GaussianDiffusion,
        policy_config   : PolicyConfig,
    ):
        model_name = policy_config.inference.model

        if model_name == "vaeac":
            return VaeacSliceImageInferencer(
                inferencer       = inferencer,
                sample_image_num = policy_config.inference.sample_image_num,
                control_mode     = policy_config.control.mode,
            )

        if model_name == "conditional_diffusion":
            return ConditionalDiffusionSliceImageInferencer(
                inferencer       = inferencer,
                sample_image_num = policy_config.inference.sample_image_num,
                control_mode     = policy_config.control.mode,
                guidance_scale   = policy_config.inference.guidance_scale,
            )

        raise ValueError(f"Unsupported inference model: {model_name}")
