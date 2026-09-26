"""
I2SB conditional image dataset with single/multi-product support.

Mask convention:
    mask == 0 : observed / known
    mask == 1 : hidden / unknown
"""

from __future__ import annotations

import os
from os.path import join
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class I2SBCondImageDataset(Dataset):

    VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

    def __init__(self, cfg, image_size: int) -> None:
        super().__init__()

        self.dataset_cfg = cfg["dataset"]
        self.image_size = int(image_size)
        self.type = self.dataset_cfg.get("type", "pattern")

        if self.type == "pattern":
            self.pattern_mask = self._get_pattern_mask()
        elif self.type == "slice":
            self.slice_grid_rows = int(self.dataset_cfg.get("slice_grid_rows", 7))
            self.slice_grid_cols = int(self.dataset_cfg.get("slice_grid_cols", 7))
            self.slice_num_observed_min = int(
                self.dataset_cfg.get("slice_num_observed_min", 1)
            )
            self.slice_num_observed_max = int(
                self.dataset_cfg.get("slice_num_observed_max", 5)
            )
            self.slice_selection = str(
                self.dataset_cfg.get("slice_selection", "random")
            )
            self._validate_slice_config()
        else:
            raise ValueError(f"Unsupported mask type: {self.type}")

        self.transform = T.Compose(
            [
                T.Resize(
                    self.image_size,
                    interpolation=InterpolationMode.NEAREST,
                ),
                T.ToTensor(),
            ]
        )

        self.products = self._parse_product_configs()
        self.samples = self._build_samples()

        # Legacy convenience attribute used by older sanity scripts.
        self.files = [sample["x0_path"] for sample in self.samples]

        # Cache one X1 image per product.
        self.x1_cache: Dict[str, torch.Tensor] = {}
        for product in self.products:
            self.x1_cache[product["name"]] = self._load_image(
                product["exterior_only_path"]
            )

        # Backward-compatible single-product attributes.
        if len(self.products) == 1:
            product = self.products[0]
            self.path = product["path"]
            self.exterior_only_path = product["exterior_only_path"]
            self.x1 = self.x1_cache[product["name"]].clone()

    # ------------------------------------------------------------------
    # Product config parsing
    # ------------------------------------------------------------------

    def _parse_product_configs(self) -> List[Dict[str, str]]:
        products_cfg = self.dataset_cfg.get("products", None)

        # Multi-product mode
        if products_cfg is not None:
            if len(products_cfg) == 0:
                raise ValueError("`dataset.products` is present but empty.")

            products = []
            seen_names = set()

            for product_cfg in products_cfg:
                name = str(product_cfg["name"])
                path = str(product_cfg["path"])
                exterior_only_path = str(product_cfg["exterior_only_path"])

                if name in seen_names:
                    raise ValueError(
                        f"Duplicate product name in dataset.products: {name}"
                    )
                seen_names.add(name)

                products.append(
                    {
                        "name": name,
                        "path": path,
                        "exterior_only_path": exterior_only_path,
                    }
                )

            return products

        # Legacy single-product mode
        if (
            "path" not in self.dataset_cfg
            or "exterior_only_path" not in self.dataset_cfg
        ):
            raise ValueError(
                "Dataset config must provide either "
                "`path` + `exterior_only_path` or `products: [...]`."
            )

        dataset_name = str(
            self.dataset_cfg.get("name", "single_product")
        )
        product_name = dataset_name

        for prefix in ("complex_2d_", "simple_2d_"):
            if product_name.startswith(prefix):
                product_name = product_name[len(prefix):]

        return [
            {
                "name": product_name,
                "path": str(self.dataset_cfg["path"]),
                "exterior_only_path": str(
                    self.dataset_cfg["exterior_only_path"]
                ),
            }
        ]

    def _find_image_files(self, path: str) -> List[str]:
        image_files = []

        for root, _, files in os.walk(join(path)):
            valid_files = sorted(
                filename
                for filename in files
                if filename.lower().endswith(
                    self.VALID_IMAGE_EXTENSIONS
                )
            )

            image_files.extend(
                join(root, filename)
                for filename in valid_files
            )

        return image_files

    def _build_samples(self) -> List[Dict[str, str]]:
        """
        Build explicit X0-X1 pair records.

        Each X0 sample is permanently associated with:
            - one product name
            - the correct exterior-only X1 for that product
        """
        samples = []

        for product in self.products:
            product_name = product["name"]
            x0_root = product["path"]
            x1_path = product["exterior_only_path"]

            if not os.path.exists(x0_root):
                raise FileNotFoundError(
                    f"X0 dataset path does not exist: {x0_root}"
                )

            if not os.path.isfile(x1_path):
                raise FileNotFoundError(
                    f"Exterior-only X1 image does not exist: {x1_path}"
                )

            x0_files = self._find_image_files(x0_root)

            if len(x0_files) == 0:
                raise RuntimeError(
                    f"No images found under product path: {x0_root}"
                )

            for x0_path in x0_files:
                samples.append(
                    {
                        "x0_path": x0_path,
                        "x1_path": x1_path,
                        "product_name": product_name,
                    }
                )

        if len(samples) == 0:
            raise RuntimeError("No I2SB samples were built.")

        return samples

    # ------------------------------------------------------------------
    # Dataset basics
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, filename: str) -> torch.Tensor:
        image = cv2.imread(filename)
        if image is None:
            raise RuntimeError(f"Failed to load image: {filename}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image)

        return (image * 2.0 - 1.0).float()

    # ------------------------------------------------------------------
    # Pattern mask
    # ------------------------------------------------------------------

    def _get_pattern_mask(self) -> np.ndarray:
        dim_scale = 6

        image = np.random.rand(600, 600)

        if self.image_size == 344:
            image = cv2.resize(
                image,
                (10000 * dim_scale, 10000 * dim_scale),
                cv2.INTER_CUBIC,
            )
        else:
            image = cv2.resize(
                image,
                (10000, 10000),
                cv2.INTER_CUBIC,
            )

        return (image > 0.25).astype(float)

    def _get_pattern_sample(self) -> np.ndarray:
        observed_ratio = 0.0
        dim_scale = 6

        while not (0.05 <= observed_ratio <= 0.9):
            if self.image_size == 344:
                max_coord = (
                    10000 * dim_scale
                    - self.image_size
                )
            else:
                max_coord = 10000 - self.image_size

            y_coord, x_coord = np.random.randint(
                max_coord,
                size=(2,),
            )

            mask = self.pattern_mask[
                y_coord:y_coord + self.image_size,
                x_coord:x_coord + self.image_size,
            ]

            observed_ratio = 1.0 - mask.mean()

        return mask

    # ------------------------------------------------------------------
    # Slice-aware mask
    # ------------------------------------------------------------------

    def _validate_slice_config(self) -> None:
        if self.slice_grid_rows <= 0 or self.slice_grid_cols <= 0:
            raise ValueError(
                "slice_grid_rows and slice_grid_cols must be positive"
            )

        num_slices = (
            self.slice_grid_rows
            * self.slice_grid_cols
        )

        if not (
            1
            <= self.slice_num_observed_min
            <= self.slice_num_observed_max
            <= num_slices
        ):
            raise ValueError(
                "Require "
                "1 <= slice_num_observed_min "
                "<= slice_num_observed_max "
                f"<= {num_slices}"
            )

        if self.slice_selection not in ("random", "contiguous"):
            raise ValueError(
                "slice_selection must be 'random' or 'contiguous'"
            )

    def _sample_observed_slice_indices(self) -> np.ndarray:
        num_slices = (
            self.slice_grid_rows
            * self.slice_grid_cols
        )

        k = np.random.randint(
            self.slice_num_observed_min,
            self.slice_num_observed_max + 1,
        )

        if self.slice_selection == "random":
            indices = np.random.choice(
                num_slices,
                size=k,
                replace=False,
            )
        else:
            start = np.random.randint(
                0,
                num_slices - k + 1,
            )
            indices = np.arange(
                start,
                start + k,
                dtype=np.int64,
            )

        return np.sort(indices)

    def _get_slice_sample(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H = self.image_size
        W = self.image_size

        # Everything hidden by default.
        mask = np.ones(
            (H, W),
            dtype=np.float32,
        )

        observed_indices = (
            self._sample_observed_slice_indices()
        )

        y_edges = np.linspace(
            0,
            H,
            self.slice_grid_rows + 1,
            dtype=int,
        )

        x_edges = np.linspace(
            0,
            W,
            self.slice_grid_cols + 1,
            dtype=int,
        )

        for idx in observed_indices:
            row = int(
                idx // self.slice_grid_cols
            )
            col = int(
                idx % self.slice_grid_cols
            )

            y0 = y_edges[row]
            y1 = y_edges[row + 1]
            x0 = x_edges[col]
            x1 = x_edges[col + 1]

            mask[y0:y1, x0:x1] = 0.0

        return mask, observed_indices

    # ------------------------------------------------------------------
    # Unified mask/condition interface
    # ------------------------------------------------------------------

    def _get_mask(self) -> torch.Tensor:
        if self.type == "pattern":
            mask_np = self._get_pattern_sample()
        elif self.type == "slice":
            mask_np, _ = self._get_slice_sample()
        else:
            raise RuntimeError(
                f"Unsupported mask type: {self.type}"
            )

        return torch.from_numpy(
            mask_np[None, :, :].astype(np.float32)
        )

    @staticmethod
    def _make_condition(
        x0: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        cond = x0.clone()

        hidden = (
            mask.expand_as(cond) == 1
        )

        cond[hidden] = -1.0

        return cond

    # ------------------------------------------------------------------
    # Item access
    # ------------------------------------------------------------------

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        sample = self.samples[idx]

        x0_path = sample["x0_path"]
        x1_path = sample["x1_path"]
        product_name = sample["product_name"]

        x0 = self._load_image(x0_path)

        # Same X1 is shared by all samples of the same product.
        x1 = self.x1_cache[
            product_name
        ].clone()

        if x0.shape != x1.shape:
            raise ValueError(
                "X0/X1 shape mismatch "
                f"for product={product_name}: "
                f"{x0.shape} vs {x1.shape}\n"
                f"X0: {x0_path}\n"
                f"X1: {x1_path}"
            )

        mask = self._get_mask()

        cond = self._make_condition(
            x0=x0,
            mask=mask,
        )

        return {
            "x0": x0,
            "x1": x1,
            "cond": cond,
            "mask": mask,
            "x0_path": x0_path,
            "x1_path": x1_path,
            "product_name": product_name,
        }
