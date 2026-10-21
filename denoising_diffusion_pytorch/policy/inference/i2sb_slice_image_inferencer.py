# i2sb_slice_image_inferencer.py

from __future__ import annotations

from pathlib import Path
from PIL import Image

import numpy as np
import torch

from denoising_diffusion_pytorch.policy.types import PlanningPolicyInput
from denoising_diffusion_pytorch.policy.resize_utils import (
    _pad_cond_to_model_size,
    _crop_images_to_hw,
)
from .slice_image_inferencer import SliceImageInferencer


class I2SBSliceImageInferencer(SliceImageInferencer):
    def __init__(
        self,
        inferencer,
        sample_image_num: int,
        control_mode: str,
        sampling_timesteps: int,
    ):
        self.inferencer = inferencer
        self.sample_image_num = int(sample_image_num)
        self.control_mode = control_mode
        self.sampling_timesteps = int(sampling_timesteps)

        # The I2SB assets loader must expose both EMA and diffusion.
        self.ema = getattr(inferencer, "ema", inferencer)
        self.diffusion = getattr(inferencer, "diffusion", None)

        if not hasattr(self.ema, "ema_model"):
            raise ValueError(
                "I2SB inferencer must provide an EMA model "
                "through .ema_model or .ema.ema_model."
            )

        if self.diffusion is None:
            raise ValueError(
                "I2SB inferencer must provide .diffusion."
            )

        self.model = self.ema.ema_model
        self.model.eval()

        if not 1 <= self.sampling_timesteps < len(self.diffusion.betas):
            raise ValueError("Invalid I2SB sampling_timesteps.")

    def predict(self, planning_input: PlanningPolicyInput) -> np.ndarray:
        cond = planning_input.normalized_cond
        x1 = planning_input.normalized_x1

        if cond is None:
            raise ValueError("I2SB requires normalized_cond.")
        if x1 is None:
            raise ValueError("I2SB requires normalized_x1.")

        if cond.ndim != 3 or x1.ndim != 3:
            raise ValueError("cond and x1 must have shape [C,H,W].")

        device = next(self.model.parameters()).device
        model_size = int(x1.shape[-1])

        if x1.shape[-2] != model_size:
            raise ValueError("I2SB X1 must be square.")

        cond, original_hw = _pad_cond_to_model_size(
            cond=cond,
            model_size=model_size,
        )

        if not torch.isfinite(cond).all() or not torch.isfinite(x1).all():
            raise ValueError("I2SB input contains NaN or Inf.")

        if cond.shape != x1.shape:
            raise ValueError(
                f"cond/x1 shape mismatch: {cond.shape} vs {x1.shape}"
            )



        # --------------------------------------------------
        # Debug: save the actual I2SB input X1
        # --------------------------------------------------
        if planning_input.debug_save_dir is not None:

            save_dir = (
                Path(planning_input.debug_save_dir)
                / "conditions"
            )
            save_dir.mkdir(parents=True, exist_ok=True)

            # X1: [3, H, W], normalized to [-1, 1]
            x1_vis = (
                ((x1.detach().float().cpu() + 1.0) / 2.0)
                .clamp(0.0, 1.0)
                .permute(1, 2, 0)
                .mul(255.0)
                .round()
                .byte()
                .numpy()
            )

            step_idx = planning_input.step_idx
            filename = (
                f"step_{step_idx:03d}_x1.png"
                if step_idx is not None
                else "x1.png"
            )

            Image.fromarray(x1_vis).save(save_dir / filename)

            print(f"[I2SB] Saved input X1: {save_dir / filename}")



        # -1 denotes unobserved pixels.
        observed = (cond != -1.0).any(dim=0, keepdim=True)

        if self.control_mode == "no_cond":
            observed = torch.zeros_like(observed)

        # mask = 0 observed, 1 hidden
        mask = (~observed).float().unsqueeze(0).to(device)


        # Save I2SB inputs for debugging
        if planning_input.debug_save_dir is not None:
            self._save_debug_inputs(
                x1=x1,
                cond=cond,
                mask=mask[0],
                save_dir=planning_input.debug_save_dir,
                step_idx=planning_input.step_idx,
            )

        cond = cond.unsqueeze(0).repeat(
            self.sample_image_num, 1, 1, 1
        ).to(device)

        x1 = x1.unsqueeze(0).repeat(
            self.sample_image_num, 1, 1, 1
        ).to(device)

        mask = mask.expand(self.sample_image_num, -1, -1, -1)

        steps = np.linspace(
            0,
            len(self.diffusion.betas) - 1,
            self.sampling_timesteps + 1,
            dtype=int,
        )
        steps = np.unique(steps).tolist()

        def pred_x0_fn(xt, step_value):
            t = torch.full(
                (xt.shape[0],),
                int(step_value),
                device=device,
                dtype=torch.long,
            )

            net_out = self.model(xt, t, None, cond)

            return self.diffusion.compute_pred_x0(
                step=t,
                xt=xt,
                net_out=net_out,
                clip_denoise=True,
            )

        with torch.inference_mode():
            xs, _ = self.diffusion.ddpm_sampling(
                steps=steps,
                pred_x0_fn=pred_x0_fn,
                x1=x1,
                cond=cond,
                mask=mask,
                ot_ode=False,
                log_steps=[0],
                verbose=False,
            )

            recon = xs[:, 0].to(device)

            # Exact consistency in observed regions.
            recon = (1.0 - mask) * cond + mask * recon

            images = (
                ((recon + 1.0) / 2.0)
                .clamp(0.0, 1.0)
                .permute(0, 2, 3, 1)
                .mul(255.0)
                .byte()
                .cpu()
                .numpy()
            )

        return _crop_images_to_hw(images, original_hw)



    @staticmethod
    def _save_debug_inputs(
        *,
        x1: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        save_dir: str,
        step_idx: int | None,
    ):
        save_dir = Path(save_dir) / "conditions"
        save_dir.mkdir(parents=True, exist_ok=True)

        prefix = (
            f"step_{step_idx:03d}"
            if step_idx is not None
            else "step_unknown"
        )

        def save_rgb(tensor, path):
            image = (
                ((tensor.detach().float().cpu() + 1.0) / 2.0)
                .clamp(0.0, 1.0)
                .permute(1, 2, 0)
                .mul(255.0)
                .round()
                .byte()
                .numpy()
            )
            Image.fromarray(image).save(path)

        # X1: save only once per episode
        x1_path = save_dir / "x1.png"

        if not x1_path.exists():
            save_rgb(x1, x1_path)
            print(f"[I2SB] Saved X1: {x1_path}")

        # C: save at every step
        save_rgb(
            cond,
            save_dir / f"{prefix}_cond.png",
        )

        # M: save at every step (0 observed, 1 hidden)
        mask_image = (
            mask.detach()
            .float()
            .cpu()
            .squeeze(0)
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .byte()
            .numpy()
        )

        Image.fromarray(mask_image).save(
            save_dir / f"{prefix}_mask.png"
        )
