"""I2SB conditional image dataset with pattern and slice-aware masks."""
from __future__ import annotations
import os
from os.path import join
from typing import Dict, Any, Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class I2SBCondImageDataset(Dataset):
    def __init__(self, cfg, image_size: int) -> None:
        super().__init__()
        self.path = cfg["dataset"]["path"]
        self.exterior_only_path = cfg["dataset"]["exterior_only_path"]
        self.image_size = int(image_size)
        self.type = cfg["dataset"].get("type", "pattern")

        if self.type == "pattern":
            self.pattern_mask = self._get_pattern_mask()
        elif self.type == "slice":
            self.slice_grid_rows = int(cfg["dataset"].get("slice_grid_rows", 7))
            self.slice_grid_cols = int(cfg["dataset"].get("slice_grid_cols", 7))
            self.slice_num_observed_min = int(cfg["dataset"].get("slice_num_observed_min", 1))
            self.slice_num_observed_max = int(cfg["dataset"].get("slice_num_observed_max", 5))
            self.slice_selection = str(cfg["dataset"].get("slice_selection", "random"))
            self._validate_slice_config()
        else:
            raise ValueError(f"Unsupported mask type: {self.type}")

        self.transform = T.Compose([
            T.Resize(self.image_size, interpolation=InterpolationMode.NEAREST),
            T.ToTensor(),
        ])
        self._get_files()
        self.x1 = self._load_image(self.exterior_only_path)

    def __len__(self):
        assert self.files, f"Empty file list: {self.path}"
        return len(self.files)

    def _get_files(self):
        self.files = []
        valid_extensions = (".png", ".jpg", ".jpeg")
        for root, _, files in os.walk(join(self.path)):
            image_files = [f for f in files if f.lower().endswith(valid_extensions)]
            self.files.extend(sorted(join(root, f) for f in image_files))
        assert self.files, f"No images found under: {self.path}"

    def _load_image(self, filename):
        image = cv2.imread(filename)
        assert image is not None, f"Failed to load image: {filename}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image)
        return (image * 2.0 - 1.0).float()

    # ------------------------------------------------------------
    # Existing pattern mask
    # ------------------------------------------------------------
    def _get_pattern_mask(self):
        dim_scale = 6
        image = np.random.rand(600, 600)
        if self.image_size == 344:
            image = cv2.resize(image, (10000 * dim_scale, 10000 * dim_scale), cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, (10000, 10000), cv2.INTER_CUBIC)
        return (image > 0.25).astype(float)

    def _get_pattern_sample(self):
        observed_ratio = 0.0
        dim_scale = 6
        while not (0.05 <= observed_ratio <= 0.9):
            if self.image_size == 344:
                max_coord = (10000 * dim_scale) - self.image_size
            else:
                max_coord = 10000 - self.image_size
            y_coord, x_coord = np.random.randint(max_coord, size=(2,))
            mask = self.pattern_mask[
                y_coord:y_coord + self.image_size,
                x_coord:x_coord + self.image_size,
            ]
            observed_ratio = 1.0 - mask.mean()
        return mask

    # ------------------------------------------------------------
    # Slice-aware mask
    # ------------------------------------------------------------
    def _validate_slice_config(self):
        if self.slice_grid_rows <= 0 or self.slice_grid_cols <= 0:
            raise ValueError("slice_grid_rows and slice_grid_cols must be positive")
        num_slices = self.slice_grid_rows * self.slice_grid_cols
        if not (1 <= self.slice_num_observed_min <= self.slice_num_observed_max <= num_slices):
            raise ValueError(
                f"Require 1 <= slice_num_observed_min <= slice_num_observed_max <= {num_slices}"
            )
        if self.slice_selection not in ("random", "contiguous"):
            raise ValueError("slice_selection must be 'random' or 'contiguous'")

    def _sample_observed_slice_indices(self):
        num_slices = self.slice_grid_rows * self.slice_grid_cols
        k = np.random.randint(self.slice_num_observed_min, self.slice_num_observed_max + 1)
        if self.slice_selection == "random":
            indices = np.random.choice(num_slices, size=k, replace=False)
        else:
            start = np.random.randint(0, num_slices - k + 1)
            indices = np.arange(start, start + k, dtype=np.int64)
        return np.sort(indices)

    def _get_slice_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        H = self.image_size
        W = self.image_size
        mask = np.ones((H, W), dtype=np.float32)
        observed_indices = self._sample_observed_slice_indices()

        y_edges = np.linspace(0, H, self.slice_grid_rows + 1, dtype=int)
        x_edges = np.linspace(0, W, self.slice_grid_cols + 1, dtype=int)

        for idx in observed_indices:
            row = int(idx // self.slice_grid_cols)
            col = int(idx % self.slice_grid_cols)
            y0, y1 = y_edges[row], y_edges[row + 1]
            x0, x1 = x_edges[col], x_edges[col + 1]
            mask[y0:y1, x0:x1] = 0.0

        return mask, observed_indices

    # ------------------------------------------------------------
    # Unified interface
    # ------------------------------------------------------------
    def _get_mask(self):
        if self.type == "pattern":
            mask_np = self._get_pattern_sample()
        elif self.type == "slice":
            mask_np, _ = self._get_slice_sample()
        else:
            raise RuntimeError(f"Unsupported mask type: {self.type}")
        return torch.from_numpy(mask_np[None, :, :].astype(np.float32))

    @staticmethod
    def _make_condition(x0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cond = x0.clone()
        hidden = mask.expand_as(cond) == 1
        cond[hidden] = -1.0
        return cond

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x0_path = self.files[idx]
        x0 = self._load_image(x0_path)
        x1 = self.x1.clone()
        if x0.shape != x1.shape:
            raise ValueError(f"X0/X1 shape mismatch: {x0.shape} vs {x1.shape}")
        mask = self._get_mask()
        cond = self._make_condition(x0=x0, mask=mask)
        return {
            "x0": x0,
            "x1": x1,
            "cond": cond,
            "mask": mask,
            "x0_path": x0_path,
        }
