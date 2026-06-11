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
