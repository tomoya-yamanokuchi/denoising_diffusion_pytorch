# denoising_diffusion_pytorch/policy/inference/condition_debug_image_writer.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


def save_condition_debug_images(
    *,
    normalized_cond: torch.Tensor,
    mask_observed: torch.Tensor,
    debug_save_dir: str | Path | None,
    step_idx: int | None,
) -> None:
    """
    Save debug images for diffusion inference conditions.

    normalized_cond:
        CHW tensor in [-1, 1].
        -1 is visualized as black, +1 as white.

    mask_observed:
        HW bool tensor.
        observed=True is visualized as white, unobserved=False as black.
    """
    if debug_save_dir is None or step_idx is None:
        return

    out_dir = Path(debug_save_dir) / "inference_conditions" / f"step_{step_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_rgb_uint8(
        _normalized_cond_to_rgb_uint8(normalized_cond),
        out_dir / "normalized_cond_used.png",
    )

    _save_rgb_uint8(
        _mask_to_rgb_uint8(mask_observed),
        out_dir / "mask_observed.png",
    )


def _normalized_cond_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert CHW tensor in [-1, 1] to HWC uint8 RGB image.
    """
    array = tensor.detach().cpu().float().numpy()

    if array.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={array.shape}")

    array = np.transpose(array, (1, 2, 0))
    array = ((array + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

    return array


def _mask_to_rgb_uint8(mask: torch.Tensor) -> np.ndarray:
    """
    Convert HW bool mask to HWC uint8 RGB image.
    """
    array = mask.detach().cpu().numpy().astype(np.uint8) * 255
    array = np.stack([array, array, array], axis=-1)

    return array


def _save_rgb_uint8(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array).save(path)
