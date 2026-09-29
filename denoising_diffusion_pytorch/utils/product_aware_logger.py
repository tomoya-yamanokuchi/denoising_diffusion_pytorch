"""
Reusable product-aware logging utilities.

The logger is intentionally model-agnostic.

Responsibilities of ProductAwareLogger
--------------------------------------
1. Select fixed validation samples per product.
2. Cache those samples once so masks / conditions do not change over time.
3. Save fixed inputs once under:
       samples/<product>/static/
4. Save dynamic outputs at every checkpoint under:
       samples/<product>/<dynamic_key>/step_XXXXXX.png
5. Optionally write the same images to TensorBoard.

Responsibilities of each trainer
--------------------------------
Each trainer only needs to provide a `sample_fn(cached_item)` callback that
returns a ProductLogOutput whose tensors are already visualization-ready
in [0, 1].

This keeps model-specific sampling logic (I2SB reverse process, DDIM/DDPM,
VAE reconstruction, etc.) outside this utility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch
from torchvision import utils


TensorDict = Dict[str, torch.Tensor]


@dataclass
class ProductLogOutput:
    """
    Standardized output returned by a trainer-specific sample callback.

    All image tensors should be NCHW and in [0, 1].

    Example:
        ProductLogOutput(
            product_name="sheetsander",
            static={
                "target": target_vis,
                "cond": cond_vis,
                "mask": mask_vis,
            },
            dynamic={
                "pred": pred_vis,
            },
        )
    """

    product_name: str
    static: TensorDict = field(default_factory=dict)
    dynamic: TensorDict = field(default_factory=dict)


class ProductAwareLogger:
    def __init__(
        self,
        *,
        dataset,
        val_dataset,
        results_folder,
        samples_per_product: int = 1,
        samples_subdir: str = "samples",
        tensorboard_prefix: str = "product",
    ):
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.results_folder = Path(results_folder)
        self.samples_dir = self.results_folder / samples_subdir
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.samples_per_product = int(samples_per_product)
        self.tensorboard_prefix = str(tensorboard_prefix)

        if self.samples_per_product <= 0:
            raise ValueError(
                "samples_per_product must be positive."
            )

        self.fixed_samples = self._build_fixed_samples()

    # ------------------------------------------------------------------
    # Fixed sample selection
    # ------------------------------------------------------------------

    def _product_order(self) -> List[str]:
        """
        Prefer dataset.products order when available.
        Fall back to product names found in dataset.samples.
        """
        products = getattr(
            self.dataset,
            "products",
            None,
        )

        if products:
            names = []

            for product in products:
                if isinstance(product, Mapping):
                    names.append(
                        str(product["name"])
                    )
                else:
                    names.append(
                        str(product.name)
                    )

            return names

        samples = getattr(
            self.dataset,
            "samples",
            None,
        )

        if samples is None:
            raise AttributeError(
                "ProductAwareLogger requires dataset.samples metadata."
            )

        return sorted(
            {
                str(sample["product_name"])
                for sample in samples
            }
        )

    def _build_fixed_samples(self):
        """
        Select validation examples by product and cache dataset[idx] once.

        This is important for datasets whose __getitem__ randomly generates
        a mask / condition. Because we cache the returned item once, the same
        evaluation condition is used at every checkpoint.
        """
        samples_meta = getattr(
            self.dataset,
            "samples",
            None,
        )

        if samples_meta is None:
            raise AttributeError(
                "ProductAwareLogger requires dataset.samples metadata "
                "with a `product_name` field."
            )

        if not hasattr(
            self.val_dataset,
            "indices",
        ):
            raise TypeError(
                "val_dataset must expose `.indices` "
                "(e.g. torch.utils.data.Subset from random_split)."
            )

        product_order = self._product_order()

        selected = {
            name: []
            for name in product_order
        }

        for dataset_idx in self.val_dataset.indices:
            product_name = str(
                samples_meta[
                    dataset_idx
                ]["product_name"]
            )

            if product_name not in selected:
                selected[
                    product_name
                ] = []

            if (
                len(selected[product_name])
                < self.samples_per_product
            ):
                selected[
                    product_name
                ].append(
                    int(dataset_idx)
                )

        missing = [
            name
            for name in product_order
            if len(selected.get(name, []))
            < self.samples_per_product
        ]

        if missing:
            raise RuntimeError(
                "Not enough validation samples for products: "
                f"{missing}. Reduce samples_per_product "
                "or enlarge the validation split."
            )

        fixed_samples = []

        for product_name in product_order:
            for sample_id, dataset_idx in enumerate(
                selected[product_name]
            ):
                item = self.dataset[
                    dataset_idx
                ]

                fixed_samples.append(
                    {
                        "product_name":
                            product_name,
                        "sample_id":
                            sample_id,
                        "dataset_idx":
                            dataset_idx,
                        "item":
                            self._clone_to_cpu(
                                item
                            ),
                    }
                )

        return fixed_samples

    @classmethod
    def _clone_to_cpu(cls, value):
        """
        Recursively copy tensors to CPU so cached samples are independent
        of later dataloader / device state.
        """
        if torch.is_tensor(value):
            return (
                value.detach()
                .cpu()
                .clone()
            )

        if isinstance(value, dict):
            return {
                key: cls._clone_to_cpu(item)
                for key, item
                in value.items()
            }

        if isinstance(value, list):
            return [
                cls._clone_to_cpu(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                cls._clone_to_cpu(item)
                for item in value
            )

        return value

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_summary(
        self,
        print_fn=print,
    ):
        print_fn(
            "[ProductAwareLogger] fixed validation samples:"
        )

        for sample in self.fixed_samples:
            item = sample["item"]

            source_path = (
                item.get("image_path")
                or item.get("x0_path")
                or item.get("path")
                or ""
            )

            print_fn(
                "  "
                f"{sample['product_name']} "
                f"sample={sample['sample_id']} "
                f"dataset_idx={sample['dataset_idx']} "
                f"path={source_path}"
            )

    # ------------------------------------------------------------------
    # Main logging entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def log(
        self,
        *,
        step: int,
        sample_fn: Callable[[dict], ProductLogOutput],
        writer=None,
    ):
        """
        Run trainer-specific inference for each cached sample, then save.

        sample_fn receives one fixed sample dict:
            {
                "product_name": ...,
                "sample_id": ...,
                "dataset_idx": ...,
                "item": cached_dataset_item,
            }

        It must return ProductLogOutput.
        """
        grouped_outputs = {}

        for fixed_sample in self.fixed_samples:
            output = sample_fn(
                fixed_sample
            )

            if not isinstance(
                output,
                ProductLogOutput,
            ):
                raise TypeError(
                    "sample_fn must return ProductLogOutput."
                )

            expected_product = (
                fixed_sample[
                    "product_name"
                ]
            )

            if (
                output.product_name
                != expected_product
            ):
                raise ValueError(
                    "sample_fn returned mismatched product_name: "
                    f"{output.product_name} != {expected_product}"
                )

            grouped_outputs.setdefault(
                output.product_name,
                [],
            ).append(
                output
            )

        for product_name, outputs in grouped_outputs.items():
            self._save_product_outputs(
                product_name=product_name,
                outputs=outputs,
                step=step,
                writer=writer,
            )

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cat_field(
        outputs: Sequence[ProductLogOutput],
        section: str,
        key: str,
    ):
        tensors = []

        for output in outputs:
            mapping = getattr(
                output,
                section,
            )

            if key not in mapping:
                raise KeyError(
                    f"Missing {section} key `{key}` "
                    f"for product {output.product_name}."
                )

            tensor = mapping[
                key
            ]

            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(
                    0
                )

            if tensor.ndim != 4:
                raise ValueError(
                    f"Expected NCHW/CHW tensor for `{key}`, "
                    f"got shape {tuple(tensor.shape)}."
                )

            tensors.append(
                tensor.detach()
                .cpu()
                .clamp(
                    0.0,
                    1.0,
                )
            )

        return torch.cat(
            tensors,
            dim=0,
        )

    def _save_product_outputs(
        self,
        *,
        product_name: str,
        outputs: Sequence[ProductLogOutput],
        step: int,
        writer=None,
    ):
        product_dir = (
            self.samples_dir
            / product_name
        )

        static_dir = (
            product_dir
            / "static"
        )

        static_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        nrow = max(
            1,
            len(outputs),
        )

        # All outputs for one product must expose identical key sets.
        static_keys = list(
            outputs[0].static.keys()
        )

        dynamic_keys = list(
            outputs[0].dynamic.keys()
        )

        for output in outputs[1:]:
            if set(
                output.static.keys()
            ) != set(
                static_keys
            ):
                raise ValueError(
                    "Static log keys differ between samples "
                    f"for product {product_name}."
                )

            if set(
                output.dynamic.keys()
            ) != set(
                dynamic_keys
            ):
                raise ValueError(
                    "Dynamic log keys differ between samples "
                    f"for product {product_name}."
                )

        # --------------------------------------------------------------
        # Static images: write once per training run.
        # --------------------------------------------------------------
        for key in static_keys:
            tensor = self._cat_field(
                outputs,
                "static",
                key,
            )

            path = (
                static_dir
                / f"{key}.png"
            )

            if not path.exists():
                utils.save_image(
                    tensor,
                    str(path),
                    nrow=nrow,
                )

            if writer is not None:
                writer.add_images(
                    (
                        f"{self.tensorboard_prefix}/"
                        f"{product_name}/{key}"
                    ),
                    tensor,
                    step,
                    dataformats="NCHW",
                )

        # --------------------------------------------------------------
        # Dynamic images: one file per checkpoint.
        # --------------------------------------------------------------
        for key in dynamic_keys:
            tensor = self._cat_field(
                outputs,
                "dynamic",
                key,
            )

            dynamic_dir = (
                product_dir
                / key
            )

            dynamic_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            path = (
                dynamic_dir
                / f"step_{step:06d}.png"
            )

            utils.save_image(
                tensor,
                str(path),
                nrow=nrow,
            )

            if writer is not None:
                writer.add_images(
                    (
                        f"{self.tensorboard_prefix}/"
                        f"{product_name}/{key}"
                    ),
                    tensor,
                    step,
                    dataformats="NCHW",
                )
