"""
Conditional diffusion image dataset with backward-compatible single-product
and multi-product support.

Mask convention
---------------
mask == 0 : observed / known
mask == 1 : hidden / unknown

Returned fields
---------------
image        : [3,H,W] full GT image in [-1,1]
mask         : [1,H,W] binary mask, 0 observed / 1 hidden
observed     : [3,H,W] conditional image in [-1,1], hidden pixels = -1
product_name : str
image_path   : str

Supported dataset configs
-------------------------
Single product:
    dataset:
      path: /path/to/images
      ...

Multi product:
    dataset:
      products:
        - name: sheetsander
          path: /path/to/sheetsander
        - name: polisher
          path: /path/to/polisher
        - name: powercutter
          path: /path/to/powercutter

The same multi-product config used by I2SB can be reused. Any
`exterior_only_path` entries are simply ignored by conditional diffusion.
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


O_MASKS = {
    "o1": [16, 35, 26, 57],
    "o2": [28, 35, 47, 57],
    "o3": [14, 49, 26, 36],
    "o4": [14, 33, 26, 36],
    "o5": [30, 49, 26, 36],
    "o6": [20, 43, 43, 61],
}


class Cond_image_dataloader(Dataset):

    VALID_IMAGE_EXTENSIONS = (
        ".png",
        ".jpg",
        ".jpeg",
    )

    def __init__(
        self,
        cfg,
        image_size,
    ):
        super().__init__()

        self.dataset_cfg = cfg["dataset"]

        self.height = int(
            self.dataset_cfg.get(
                "h",
                32,
            )
        )

        self.type = self.dataset_cfg.get(
            "type",
            None,
        )

        self.p = float(
            self.dataset_cfg.get(
                "p",
                1.0,
            )
        )

        self.image_size = int(
            image_size
        )

        # --------------------------------------------------------------
        # Mask-specific setup
        # --------------------------------------------------------------
        self.pattern_mask = None

        if self.type == "pattern":
            self.pattern_mask = (
                self._get_pattern_mask()
            )

        elif self.type == "slice":
            self.slice_grid_rows = int(
                self.dataset_cfg.get(
                    "slice_grid_rows",
                    7,
                )
            )

            self.slice_grid_cols = int(
                self.dataset_cfg.get(
                    "slice_grid_cols",
                    7,
                )
            )

            self.slice_num_observed_min = int(
                self.dataset_cfg.get(
                    "slice_num_observed_min",
                    1,
                )
            )

            self.slice_num_observed_max = int(
                self.dataset_cfg.get(
                    "slice_num_observed_max",
                    5,
                )
            )

            self.slice_selection = str(
                self.dataset_cfg.get(
                    "slice_selection",
                    "random",
                )
            )

            self._validate_slice_config()

        # --------------------------------------------------------------
        # Image transform
        # --------------------------------------------------------------
        self.transform = T.Compose(
            [
                T.Resize(
                    self.image_size,
                    interpolation=InterpolationMode.NEAREST,
                ),
                T.ToTensor(),
            ]
        )

        # --------------------------------------------------------------
        # Single / multi-product setup
        # --------------------------------------------------------------
        self.products = (
            self._parse_product_configs()
        )

        self.samples = (
            self._build_samples()
        )

        # Backward compatibility with older code that expects self.files.
        self.files = [
            sample["image_path"]
            for sample in self.samples
        ]

        if len(self.products) == 1:
            self.path = self.products[
                0
            ]["path"]

    # ==================================================================
    # Product handling
    # ==================================================================

    def _parse_product_configs(
        self,
    ) -> List[Dict[str, str]]:

        products_cfg = self.dataset_cfg.get(
            "products",
            None,
        )

        # Multi-product mode
        if products_cfg is not None:
            if len(products_cfg) == 0:
                raise ValueError(
                    "`dataset.products` is present but empty."
                )

            products = []
            seen_names = set()

            for product_cfg in products_cfg:
                name = str(
                    product_cfg["name"]
                )

                path = str(
                    product_cfg["path"]
                )

                if name in seen_names:
                    raise ValueError(
                        "Duplicate product name: "
                        f"{name}"
                    )

                seen_names.add(
                    name
                )

                products.append(
                    {
                        "name": name,
                        "path": path,
                    }
                )

            return products

        # Legacy single-product mode
        if "path" not in self.dataset_cfg:
            raise ValueError(
                "Dataset config must provide either "
                "`dataset.path` or `dataset.products`."
            )

        dataset_name = str(
            self.dataset_cfg.get(
                "name",
                "single_product",
            )
        )

        product_name = dataset_name

        for prefix in (
            "complex_2d_",
            "simple_2d_",
        ):
            if product_name.startswith(
                prefix
            ):
                product_name = (
                    product_name[
                        len(prefix):
                    ]
                )

        return [
            {
                "name": product_name,
                "path": str(
                    self.dataset_cfg["path"]
                ),
            }
        ]

    def _find_image_files(
        self,
        path: str,
    ) -> List[str]:

        image_files = []

        for root, _, files in os.walk(
            join(path)
        ):
            valid_files = sorted(
                filename
                for filename in files
                if filename.lower().endswith(
                    self.VALID_IMAGE_EXTENSIONS
                )
            )

            image_files.extend(
                join(
                    root,
                    filename,
                )
                for filename in valid_files
            )

        return image_files

    def _build_samples(
        self,
    ) -> List[Dict[str, str]]:

        samples = []

        for product in self.products:
            product_name = product[
                "name"
            ]

            root = product[
                "path"
            ]

            if not os.path.isdir(
                root
            ):
                raise FileNotFoundError(
                    f"Dataset path does not exist: {root}"
                )

            files = self._find_image_files(
                root
            )

            if len(files) == 0:
                raise RuntimeError(
                    f"No images found under: {root}"
                )

            for image_path in files:
                samples.append(
                    {
                        "image_path": image_path,
                        "product_name":
                            product_name,
                    }
                )

        if len(samples) == 0:
            raise RuntimeError(
                "No conditional-diffusion samples were built."
            )

        return samples

    # ==================================================================
    # Pattern mask
    # ==================================================================

    def _get_pattern_mask(
        self,
    ):
        dim_scale = 6

        image = np.random.rand(
            600,
            600,
        )

        if self.image_size == 344:
            image = cv2.resize(
                image,
                (
                    10000 * dim_scale,
                    10000 * dim_scale,
                ),
                cv2.INTER_CUBIC,
            )
        else:
            image = cv2.resize(
                image,
                (
                    10000,
                    10000,
                ),
                cv2.INTER_CUBIC,
            )

        return (
            image > 0.25
        ).astype(
            np.float32
        )

    def _get_pattern_sample(
        self,
    ):
        frac = 0.0
        dim_scale = 6

        while not (
            0.05
            <= frac
            <= 0.9
        ):
            if self.image_size == 344:
                max_coord = (
                    10000
                    * dim_scale
                    - self.image_size
                )
            else:
                max_coord = (
                    10000
                    - self.image_size
                )

            y_coord, x_coord = (
                np.random.randint(
                    max_coord,
                    size=(2,),
                )
            )

            mask = self.pattern_mask[
                y_coord:
                    y_coord
                    + self.image_size,
                x_coord:
                    x_coord
                    + self.image_size,
            ]

            frac = (
                1.0
                - mask.mean()
            )

        return mask

    # ==================================================================
    # Slice-aware mask
    # ==================================================================

    def _validate_slice_config(
        self,
    ):
        if (
            self.slice_grid_rows <= 0
            or self.slice_grid_cols <= 0
        ):
            raise ValueError(
                "slice_grid_rows and slice_grid_cols "
                "must be positive."
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
                f"<= {num_slices}."
            )

        if self.slice_selection not in (
            "random",
            "contiguous",
        ):
            raise ValueError(
                "slice_selection must be "
                "'random' or 'contiguous'."
            )

    def _sample_observed_slice_indices(
        self,
    ):
        num_slices = (
            self.slice_grid_rows
            * self.slice_grid_cols
        )

        k = np.random.randint(
            self.slice_num_observed_min,
            self.slice_num_observed_max
            + 1,
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
                num_slices
                - k
                + 1,
            )

            indices = np.arange(
                start,
                start + k,
                dtype=np.int64,
            )

        return np.sort(
            indices
        )

    def _get_slice_sample(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:

        H = self.image_size
        W = self.image_size

        # 1 = hidden, 0 = observed
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
                idx
                // self.slice_grid_cols
            )

            col = int(
                idx
                % self.slice_grid_cols
            )

            y0 = y_edges[
                row
            ]

            y1 = y_edges[
                row + 1
            ]

            x0 = x_edges[
                col
            ]

            x1 = x_edges[
                col + 1
            ]

            mask[
                y0:y1,
                x0:x1,
            ] = 0.0

        return (
            mask,
            observed_indices,
        )

    # ==================================================================
    # Generic mask generation
    # ==================================================================

    def _get_mask(
        self,
        image,
    ):
        if self.type is None:
            mask = np.ones(
                image.shape,
                dtype=np.float32,
            )[:, :, :1]

            x_start, y_start = np.random.randint(
                image.shape[0]
                - self.height,
                size=(2,),
            )

            width = self.height
            height = self.height

            mask[
                y_start:
                    y_start + height,
                x_start:
                    x_start + width,
            ] = 0.0

        elif self.type == "center":
            mask = np.ones(
                image.shape,
                dtype=np.float32,
            )[:, :, :1]

            c_y = (
                image.shape[0]
                // 2
            )

            c_x = (
                image.shape[1]
                // 2
            )

            mask[
                c_y - self.height // 2:
                    c_y + self.height // 2,
                c_x - self.height // 2:
                    c_x + self.height // 2,
            ] = 0.0

        elif self.type == "random":
            pp = np.random.uniform(
                low=0.0,
                high=1.0,
            )

            mask = np.random.binomial(
                1.0,
                pp,
                image.shape,
            )[:, :, :1].astype(
                np.float32
            )

        elif self.type == "half":
            mask = np.ones(
                image.shape,
                dtype=np.float32,
            )[:, :, :1]

            left_start, top_start = (
                32
                * np.random.randint(
                    2,
                    size=(2,),
                )
            )

            if np.random.rand() < 0.5:
                mask[
                    :,
                    left_start:
                        left_start
                        + 32,
                ] = 0.0
            else:
                mask[
                    top_start:
                        top_start
                        + 32,
                    :,
                ] = 0.0

        elif self.type == "pattern":
            mask = (
                self._get_pattern_sample()[
                    :,
                    :,
                    None,
                ]
            )

        elif self.type == "slice":
            mask_2d, _ = (
                self._get_slice_sample()
            )

            mask = mask_2d[
                :,
                :,
                None,
            ]

        elif self.type in O_MASKS:
            x_start, x_end, y_start, y_end = (
                O_MASKS[
                    self.type
                ]
            )

            mask = np.ones(
                image.shape,
                dtype=np.float32,
            )[:, :, :1]

            mask[
                y_start:y_end,
                x_start:x_end,
            ] = 0.0

        else:
            raise NotImplementedError(
                f"Unknown mask type: {self.type}"
            )

        return mask.astype(
            np.float32
        )

    # ==================================================================
    # Dataset interface
    # ==================================================================

    def __len__(
        self,
    ):
        if len(
            self.samples
        ) == 0:
            raise RuntimeError(
                "Empty file list."
            )

        return len(
            self.samples
        )

    def _load_image(
        self,
        filename,
    ):
        image = cv2.imread(
            filename
        )

        if image is None:
            raise RuntimeError(
                f"Failed to load image: {filename}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            image
        )

        image = self.transform(
            image
        )

        # [3,H,W], [0,1]
        return image.float()

    def __getitem__(
        self,
        idx,
    ):
        sample = self.samples[
            idx
        ]

        filename = sample[
            "image_path"
        ]

        product_name = sample[
            "product_name"
        ]

        image_01 = self._load_image(
            filename
        )

        # [3,H,W] -> [H,W,3] for legacy mask helper.
        image_np = (
            image_01
            .permute(
                1,
                2,
                0,
            )
            .numpy()
        )

        image = (
            image_np * 2.0
            - 1.0
        ).astype(
            np.float32
        )

        mask = self._get_mask(
            image
        )

        # Conditional image:
        # observed pixels retain GT, hidden pixels become -1 (black).
        observed = image.copy()

        hidden = (
            mask[
                :,
                :,
                0,
            ]
            == 1.0
        )

        observed[
            hidden,
            :,
        ] = -1.0

        return {
            "image":
                torch.from_numpy(
                    image.transpose(
                        2,
                        0,
                        1,
                    )
                ).float(),
            "mask":
                torch.from_numpy(
                    mask.transpose(
                        2,
                        0,
                        1,
                    )
                ).float(),
            "observed":
                torch.from_numpy(
                    observed.transpose(
                        2,
                        0,
                        1,
                    )
                ).float(),
            "product_name":
                product_name,
            "image_path":
                filename,
        }
