import torch
import torch.nn.functional as F


def _pad_cond_to_model_size(
    cond      : torch.Tensor,
    model_size: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    cond: [C, H, W]
    """
    _, h, w = cond.shape
    original_hw = (h, w)

    if (h, w) == (model_size, model_size):
        return cond, original_hw

    if h > model_size or w > model_size:
        raise ValueError(
            f"condition image is larger than model input: "
            f"cond={h}x{w}, model={model_size}x{model_size}"
        )

    pad_h = model_size - h
    pad_w = model_size - w

    # pad=(left, right, top, bottom)
    # -1.0 は「未観測」扱い
    padded = F.pad(cond, (0, pad_w, 0, pad_h), mode="constant", value=-1.0)
    # import ipdb; ipdb.set_trace()
    return padded, original_hw


def _crop_images_to_hw(images, hw: tuple[int, int]):
    """
    images: [B, H, W, C]
    """
    h, w = hw
    return images[:, :h, :w, :]
