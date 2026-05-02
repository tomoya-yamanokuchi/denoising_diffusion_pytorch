from __future__ import annotations

import numpy as np
import torch

from denoising_diffusion_pytorch.policy.types import PlanningPolicyInput

from ..resize_utils import _crop_images_to_hw, _pad_cond_to_model_size
from .slice_image_inferencer import SliceImageInferencer
from denoising_diffusion_pytorch.models.conditional_image_diffusion_cfg_devel2 import GaussianDiffusion


class ConditionalDiffusionSliceImageInferencer(SliceImageInferencer):
    def __init__(
        self,
        inferencer      : GaussianDiffusion,
        sample_image_num: int,
        control_mode    : str,
        guidance_scale  : float,
    ):
        self.inferencer       = inferencer
        self.sample_image_num = int(sample_image_num)
        self.control_mode     = control_mode
        self.guidance_scale   = float(guidance_scale)


    def predict(self, planning_input: PlanningPolicyInput) -> np.ndarray:
        normalized_cond = planning_input.normalized_cond
        if normalized_cond is None:
            raise ValueError(
                "normalized_cond must not be None for conditional diffusion inference."
            )
        # --- prepare cond and mask ---
        model_size = int(self.inferencer.ema_model.image_size)
        normalized_cond, original_hw = _pad_cond_to_model_size(
            cond       = normalized_cond,
            model_size = model_size,
        )


        if self.control_mode == "no_cond":
            cond = None
            normalized_cond = normalized_cond.clone()
            normalized_cond[:] = -1.0
            mask = normalized_cond.repeat(self.sample_image_num, 1, 1, 1)
        else:
            mask_tmp = (normalized_cond != -1.0).any(dim=0)
            cond = {
                0: {
                    "idx": torch.where(mask_tmp),
                    "val": normalized_cond,
                }
            }
            mask = normalized_cond.repeat(self.sample_image_num, 1, 1, 1)

        sample_image = self.inferencer.ema_model.sample(
            batch_size           = self.sample_image_num,
            return_all_timesteps = True,
            cond                 = cond,
            mask                 = mask,
            omega                = self.guidance_scale,
        ).detach().cpu()

        batch_images = (
            torch.permute(sample_image, (0, 1, 3, 4, 2)) * 255.0
        ).clamp(0, 255).cpu().numpy().astype(np.uint8)

        last_step_images = batch_images[:, -1, :, :, :]
        last_step_images = _crop_images_to_hw(last_step_images, original_hw)
        return last_step_images

