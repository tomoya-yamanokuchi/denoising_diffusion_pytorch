from __future__ import annotations

import numpy as np
import torch

from denoising_diffusion_pytorch.policy.types import PlanningPolicyInput
from denoising_diffusion_pytorch.utils.vaeac_utils.vaeac_utils import vaeac_validate

from .slice_image_inferencer import SliceImageInferencer
from denoising_diffusion_pytorch.models.vaeac.vaeac import EncoderDecoder


class VaeacSliceImageInferencer(SliceImageInferencer):
    def __init__(
        self,
        inferencer      : EncoderDecoder,
        sample_image_num: int,
        control_mode    : str,
    ):
        self.inferencer       = inferencer
        self.sample_image_num = int(sample_image_num)
        self.control_mode     = control_mode


    def predict(self, planning_input: PlanningPolicyInput) -> np.ndarray:
        normalized_cond = planning_input.normalized_cond
        if normalized_cond is None:
            raise ValueError("normalized_cond must not be None for VAEAC inference.")

        self.inferencer.eval()

        if self.control_mode == "no_cond":
            mask_ = torch.where(
                (normalized_cond == -1.0).all(dim=0),
                torch.tensor(1),
                torch.tensor(1),
            )
            observation = normalized_cond.repeat(self.sample_image_num, 1, 1, 1)
            mask = mask_.repeat(self.sample_image_num, 1, 1, 1)
        else:
            mask_ = torch.where(
                (normalized_cond == -1.0).all(dim=0),
                torch.tensor(1),
                torch.tensor(0),
            )
            observation = normalized_cond.repeat(self.sample_image_num, 1, 1, 1)
            mask = mask_.repeat(self.sample_image_num, 1, 1, 1)

        data = {
            "image"   : observation,
            "mask"    : mask,
            "observed": observation,
        }

        sample_image_ = vaeac_validate(
            model=self.inferencer,
            data=data,
        ).detach().cpu()

        sample_image = sample_image_.unsqueeze(1)

        batch_images = (
            ((torch.permute(sample_image, (0, 1, 3, 4, 2)) + 1.0) / 2.0) * 255.0
        ).clamp(0, 255).cpu().numpy().astype(np.uint8)

        last_step_images = batch_images[:, -1, :, :, :]
        return last_step_images
