"""
Sanity check for multi-product conditional diffusion dataset.

Usage:
    python scripts/sanity_check_conditional_diffusion_multi_dataset.py \
        usecase=train/complex/conditional_diffusion
"""

from collections import Counter

import hydra
import torch
from omegaconf import DictConfig

from denoising_diffusion_pytorch.data_loader.cond_image_data_loader_multi_product import (
    Cond_image_dataloader,
)


@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):

    cfg = cfg.usecase

    dataset = Cond_image_dataloader(
        cfg=cfg,
        image_size=cfg.dataset.image_size,
    )

    print(
        "dataset size =",
        len(dataset),
    )

    print(
        "num products =",
        len(dataset.products),
    )

    counts = Counter(
        sample["product_name"]
        for sample
        in dataset.samples
    )

    print(
        "\n--- product counts ---"
    )

    for product_name, count in counts.items():
        print(
            f"{product_name}: {count}"
        )

    print(
        "\n--- representative samples ---"
    )

    for product in dataset.products:
        product_name = product[
            "name"
        ]

        idx = next(
            i
            for i, sample
            in enumerate(
                dataset.samples
            )
            if sample[
                "product_name"
            ]
            == product_name
        )

        item = dataset[
            idx
        ]

        print(
            f"\nproduct = {product_name}"
        )

        print(
            "index =",
            idx,
        )

        print(
            "path =",
            item[
                "image_path"
            ],
        )

        print(
            "image shape =",
            tuple(
                item[
                    "image"
                ].shape
            ),
        )

        print(
            "observed shape =",
            tuple(
                item[
                    "observed"
                ].shape
            ),
        )

        print(
            "mask shape =",
            tuple(
                item[
                    "mask"
                ].shape
            ),
        )

        print(
            "mask unique =",
            torch.unique(
                item[
                    "mask"
                ]
            ),
        )

        assert (
            item[
                "product_name"
            ]
            == product_name
        )

        assert (
            item[
                "image"
            ].shape
            == item[
                "observed"
            ].shape
        )

        assert (
            item[
                "mask"
            ].shape[
                0
            ]
            == 1
        )

        assert set(
            torch.unique(
                item[
                    "mask"
                ]
            ).tolist()
        ).issubset(
            {
                0.0,
                1.0,
            }
        )

        # Check that hidden pixels in condition are black (-1).
        hidden = (
            item[
                "mask"
            ]
            == 1.0
        ).expand_as(
            item[
                "observed"
            ]
        )

        if hidden.any():
            hidden_values = (
                item[
                    "observed"
                ][
                    hidden
                ]
            )

            max_error = (
                hidden_values
                + 1.0
            ).abs().max()

            print(
                "hidden condition max error from -1 =",
                float(
                    max_error
                ),
            )

            assert (
                float(
                    max_error
                )
                < 1e-6
            )

    print(
        "\nConditional diffusion multi-product "
        "dataset sanity check PASSED."
    )


if __name__ == "__main__":
    main()
