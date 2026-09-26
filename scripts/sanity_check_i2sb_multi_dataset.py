"""
Sanity check for single/multi-product I2SB dataset pairing.

Usage:
    python scripts/sanity_check_i2sb_multi_dataset.py \
        usecase=train/complex/i2sb
"""

from collections import Counter

import hydra
import torch
from omegaconf import DictConfig

from denoising_diffusion_pytorch.data_loader.i2sb_cond_image_data_loader import (
    I2SBCondImageDataset,
)


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
    print("num products =", len(dataset.products))

    counts = Counter(
        sample["product_name"]
        for sample in dataset.samples
    )

    print("\n--- product counts ---")
    for product_name, count in counts.items():
        print(f"{product_name}: {count}")

    print("\n--- representative samples ---")

    for product in dataset.products:
        product_name = product["name"]

        idx = next(
            i
            for i, sample in enumerate(dataset.samples)
            if sample["product_name"] == product_name
        )

        sample = dataset[idx]

        print(f"\nproduct = {product_name}")
        print("index =", idx)
        print("x0_path =", sample["x0_path"])
        print("x1_path =", sample["x1_path"])
        print("x0 shape =", tuple(sample["x0"].shape))
        print("x1 shape =", tuple(sample["x1"].shape))
        print("cond shape =", tuple(sample["cond"].shape))
        print("mask shape =", tuple(sample["mask"].shape))
        print("mask unique =", torch.unique(sample["mask"]))

        assert sample["product_name"] == product_name
        assert (
            sample["x1_path"]
            == product["exterior_only_path"]
        )
        assert (
            sample["x0"].shape
            == sample["x1"].shape
            == sample["cond"].shape
        )
        assert sample["mask"].shape[0] == 1

    print(
        "\nI2SB single/multi-product dataset sanity check PASSED."
    )


if __name__ == "__main__":
    main()
