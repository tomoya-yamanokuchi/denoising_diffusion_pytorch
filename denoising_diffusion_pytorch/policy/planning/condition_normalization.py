from __future__ import annotations

import numpy as np


def normalize_condition_image_to_minus1_plus1(slice_img: np.ndarray) -> np.ndarray:
    """
    Convert a 0-1 or 0-255 RGB condition image to the model's [-1, 1] range.

    Training-time observed condition images use a fixed [0, 1] -> [-1, 1]
    scaling, where unobserved black pixels become -1.0. Therefore, inference
    should not use per-image min-max normalization. In particular, an all-black
    condition image is a valid empty observation and should become all -1.0,
    not NaN.
    """
    image = np.asarray(slice_img, dtype=np.float32)

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            "Expected an RGB condition image with shape (H, W, 3), "
            f"but got shape={image.shape}."
        )

    if not np.isfinite(image).all():
        raise ValueError("Condition image contains NaN or Inf before normalization.")

    if image.max() > 1.0 + 1e-6:
        image = image / 255.0

    image = np.clip(image, 0.0, 1.0)
    normalized = image * 2.0 - 1.0

    if not np.isfinite(normalized).all():
        raise ValueError("Condition image contains NaN or Inf after normalization.")

    return normalized.astype(np.float32)





def normalize_01_array_to_minus1_plus1(x: np.ndarray) -> np.ndarray:
    """
    Convert arbitrary [0, 1] or [0, 255] array to [-1, 1].
    This is for Diffusion1D sequence tensors such as [3, N] or [6, N],
    not only RGB images with shape [H, W, 3].
    """
    arr = np.asarray(x, dtype=np.float32)

    if not np.isfinite(arr).all():
        raise ValueError("Condition array contains NaN or Inf before normalization.")

    if arr.size > 0 and arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0

    arr = np.clip(arr, 0.0, 1.0)
    normalized = arr * 2.0 - 1.0

    if not np.isfinite(normalized).all():
        raise ValueError("Condition array contains NaN or Inf after normalization.")

    return normalized.astype(np.float32)
