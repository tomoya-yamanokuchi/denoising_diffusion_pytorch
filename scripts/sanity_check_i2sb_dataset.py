from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from denoising_diffusion_pytorch.data_loader.i2sb_cond_image_data_loader \
    import I2SBCondImageDataset


@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):

    cfg = cfg.usecase

    dataset = I2SBCondImageDataset(
        cfg=cfg,
        image_size=cfg.dataset.image_size,
    )

    print("dataset size =", len(dataset))

    sample = dataset[0]

    x0 = sample["x0"]
    x1 = sample["x1"]
    cond = sample["cond"]
    mask = sample["mask"]

    print("\n--- basic information ---")

    print("x0_path =", sample["x0_path"])

    for name, tensor in [
        ("x0", x0),
        ("x1", x1),
        ("cond", cond),
        ("mask", mask),
    ]:
        print(
            name,
            "shape =", tuple(tensor.shape),
            "dtype =", tensor.dtype,
            "min =", tensor.min().item(),
            "max =", tensor.max().item(),
        )

    print("\nmask unique =", torch.unique(mask))

    # --------------------------------------------------
    # shape checks
    # --------------------------------------------------

    assert x0.shape == x1.shape
    assert x0.shape == cond.shape

    assert mask.shape[0] == 1
    assert mask.shape[1:] == x0.shape[1:]

    # --------------------------------------------------
    # range checks
    # --------------------------------------------------

    assert x0.min() >= -1.0
    assert x0.max() <= 1.0

    assert x1.min() >= -1.0
    assert x1.max() <= 1.0

    assert cond.min() >= -1.0
    assert cond.max() <= 1.0

    # --------------------------------------------------
    # binary mask check
    # --------------------------------------------------

    mask_unique = torch.unique(mask)

    assert torch.all(
        (mask_unique == 0.0)
        | (mask_unique == 1.0)
    )

    # --------------------------------------------------
    # condition semantics check
    # --------------------------------------------------

    expanded_mask = mask.expand_as(x0)

    observed_region = expanded_mask == 0
    hidden_region = expanded_mask == 1

    observed_error = (
        cond[observed_region]
        - x0[observed_region]
    ).abs().max()

    hidden_error = (
        cond[hidden_region]
        - (-1.0)
    ).abs().max()

    print("\n--- condition checks ---")

    print(
        "observed region max error =",
        observed_error.item(),
    )

    print(
        "hidden region max error =",
        hidden_error.item(),
    )

    assert observed_error < 1e-6
    assert hidden_error < 1e-6

    # --------------------------------------------------
    # mask ratio
    # --------------------------------------------------

    hidden_ratio = mask.mean().item()

    print(
        "\nhidden ratio =",
        hidden_ratio,
    )

    print("\nDataset sanity check PASSED.")


if __name__ == "__main__":
    main()
