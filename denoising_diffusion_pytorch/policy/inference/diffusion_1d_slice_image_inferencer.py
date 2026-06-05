# denoising_diffusion_pytorch/policy/inference/diffusion_1d_slice_image_inferencer.py
from __future__ import annotations

import numpy as np
import torch

from denoising_diffusion_pytorch.policy.types import PlanningPolicyInput
from denoising_diffusion_pytorch.policy.diffusion_1d_policy_utils import (
    get_2d_image_to_1d,
)
from denoising_diffusion_pytorch.utils.normalization import LimitsNormalizer
from .slice_image_inferencer import SliceImageInferencer


class Diffusion1DSliceImageInferencer(SliceImageInferencer):
    def __init__(
        self,
        inferencer,
        sample_image_num: int,
        control_mode: str,
        sampling_timesteps: int | None = None,
    ):
        # inferencer は Diffusion1DAssetsLoader から渡される Trainer1D
        self.trainer = inferencer
        self.sample_image_num = int(sample_image_num)
        self.control_mode = control_mode

        if sampling_timesteps is not None:
            self.trainer.ema.ema_model.sampling_timesteps = int(sampling_timesteps)
            print(
                "diffusion_1d model.sampling_timesteps = ",
                self.trainer.ema.ema_model.sampling_timesteps,
            )

    def predict(self, planning_input: PlanningPolicyInput) -> np.ndarray:
        normalized_cond = planning_input.normalized_cond
        if normalized_cond is None:
            raise ValueError(
                "normalized_cond must not be None for diffusion_1d inference."
            )

        self.trainer.ema.ema_model.eval()

        # normalized_cond: [C, H, W] -> [H, W, C]
        slice_image = torch.permute(normalized_cond, (1, 2, 0))

        grid_3dim = int(self.trainer.grid_3dim)

        cond_image_1d = get_2d_image_to_1d(
            image=slice_image,
            grid_3_dim=grid_3dim,
            is_shuffle=False,
        )

        cond_np = cond_image_1d.detach().cpu().numpy()

        normalizer_values = LimitsNormalizer(cond_np[3:, :])
        normalizer_indices = LimitsNormalizer(cond_np[:3, :])

        voxel_values = torch.as_tensor(
            normalizer_values.normalize(cond_np[3:, :]),
            dtype=torch.float32,
            device=normalized_cond.device,
        )
        voxel_indices = torch.as_tensor(
            normalizer_indices.normalize(cond_np[:3, :]),
            dtype=torch.float32,
            device=normalized_cond.device,
        )

        if self.control_mode == "no_cond":
            cond = None
        else:
            cond = {
                0: {
                    "idx": torch.where(voxel_values.mean(0) > -1.0),
                    "val": voxel_values,
                    "pos": voxel_indices,
                    "data": torch.cat((voxel_indices, voxel_values), dim=0),
                }
            }

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sampled_seq = self.trainer.ema.ema_model.sample(
                    batch_size=self.sample_image_num,
                    # return_all_timesteps=True,
                    return_all_timesteps=False,
                    cond=cond,
                )

            # # [B, T, C, N] -> last step [B, C, N]
            # last_seq = sampled_seq[:, -1, :, :]
            # sampled_image = self.trainer.get_1d_to_2d_images(last_seq).detach().cpu()

            # return_all_timesteps=False の場合:
            # sampled_seq: [B, C, N]
            sampled_image = self.trainer.get_1d_to_2d_images(sampled_seq).detach().cpu()

        last_step_images = (
            torch.permute(sampled_image, (0, 2, 3, 1)) * 255.0
        ).clamp(0, 255).cpu().numpy().astype(np.uint8)

        return last_step_images
