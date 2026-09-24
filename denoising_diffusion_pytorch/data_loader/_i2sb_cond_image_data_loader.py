"""
I2SB-specific conditional image dataset.

This loader is intentionally separated from the existing
`cond_image_data_loader.py` so that the current Conditional DDPM /
VoxelDiffusionCut experimental pipeline remains untouched.

Returned items
--------------
x0   : [3, H, W], float32 in [-1, 1]
       Full slice image (exterior + internal structure).

x1   : [3, H, W], float32 in [-1, 1]
       Exterior-only slice image used as the I2SB bridge endpoint.

cond : [3, H, W], float32 in [-1, 1]
       Partial observation used as the network condition.
       It is a masked version of x0.

mask : [1, H, W], float32 in {0, 1}
       0 = observed / known
       1 = hidden / unknown

x0_path : str
       Source path of the full slice image.

Notes
-----
- Only the `pattern` mask type is supported because the current project
  configs use `type: pattern`.
- The pattern-mask generation logic follows the existing
  `cond_image_data_loader.py` behavior as closely as possible so that
  DDPM vs. I2SB comparisons do not introduce a different mask
  distribution.
"""

from __future__ import annotations

import os
from os.path import join
from typing import Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class I2SBCondImageDataset(Dataset):
    """
    Dataset for Image-to-Image Schroedinger Bridge (I2SB) training.

    Expected config entries
    -----------------------
    cfg["dataset"]["path"]
        Directory containing full slice images (X0).

    cfg["dataset"]["exterior_only_path"]
        Path to the exterior-only slice image (X1).

    cfg["dataset"]["image_size"]
        Usually supplied separately as `image_size`.

    cfg["dataset"]["type"]
        Must be "pattern".
    """

    def __init__(
        self,
        cfg,
        image_size: int,
    ) -> None:
        super().__init__()

        self.path               = cfg["dataset"]["path"]
        self.exterior_only_path = cfg["dataset"]["exterior_only_path"]
        self.image_size         = int(image_size)
        self.type               = cfg["dataset"].get("type", None)

        if self.type != "pattern":
            raise ValueError(
                "I2SBCondImageDataset currently supports only "
                f"`type: pattern`, but got: {self.type}"
            )

        # Keep the same resize interpolation as the existing loader.
        self.transform = T.Compose(
            [
                T.Resize(
                    self.image_size,
                    interpolation=InterpolationMode.NEAREST,
                ),
                T.ToTensor(),
            ]
        )

        # Build the same style of large random pattern mask used by the
        # existing conditional-diffusion data loader.
        self.pattern_mask = self._get_pattern_mask()

        # Collect full-structure images used as X0.
        self._get_files()

        # At the moment X1 is shared by all X0 samples.
        # If multiple slice indices are introduced later, this can be
        # replaced by a per-sample X0 -> X1 path mapping.
        self.x1 = self._load_image(self.exterior_only_path)

    def __len__(self) -> int:
        assert self.files, f"Empty file list: {self.path}"
        return len(self.files)

    def _get_files(self) -> None:
        """
        Recursively collect image files under the X0 dataset directory.
        """
        self.files = []

        valid_extensions = (".png", ".jpg", ".jpeg")

        for root, _, files in os.walk(join(self.path)):
            image_files = [
                filename
                for filename in files
                if filename.lower().endswith(valid_extensions)
            ]

            self.files.extend(
                sorted(
                    join(root, filename)
                    for filename in image_files
                )
            )

        assert self.files, f"No images found under: {self.path}"

    def _load_image(self, filename: str) -> torch.Tensor:
        """
        Load an RGB image and normalize it from [0, 1] to [-1, 1].

        Returns
        -------
        torch.Tensor
            Shape [3, H, W], dtype float32.
        """
        image = cv2.imread(filename)

        assert image is not None, f"Failed to load image: {filename}"

        # OpenCV BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # NumPy -> PIL -> resized tensor in [0, 1]
        image = Image.fromarray(image)
        image = self.transform(image)

        # [0, 1] -> [-1, 1]
        image = image * 2.0 - 1.0

        return image.float()

    def _get_pattern_mask(self) -> np.ndarray:
        """
        Generate the large random pattern field used for mask sampling.

        This follows the existing `cond_image_data_loader.py` logic.
        """
        dim_scale = 6  # 344 / 64 in the original implementation

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

        # Same threshold as the existing loader.
        image = (image > 0.25).astype(float)

        return image

    def _get_pattern_sample(self) -> np.ndarray:
        """
        Sample one H x W patch from the pre-generated pattern field.

        The existing loader accepts samples for which the hidden fraction
        lies between 5% and 90%.
        """
        frac = 0.0
        dim_scale = 6

        while not (0.05 <= frac <= 0.9):
            if self.image_size == 344:
                max_coord = (10000 * dim_scale) - self.image_size
            else:
                max_coord = 10000 - self.image_size

            y_coord, x_coord = np.random.randint(
                max_coord,
                size=(2,),
            )

            mask = self.pattern_mask[
                y_coord : y_coord + self.image_size,
                x_coord : x_coord + self.image_size,
            ]

            frac = 1.0 - mask.mean()

        return mask

    def _get_mask(self) -> torch.Tensor:
        """
        Generate the binary observation mask.

        Semantics are intentionally kept compatible with the existing
        conditional-diffusion loader:

            mask == 0 -> observed / known
            mask == 1 -> hidden / unknown

        Returns
        -------
        torch.Tensor
            Shape [1, H, W], float32 in {0, 1}.
        """
        pattern = self._get_pattern_sample()

        mask = torch.from_numpy(
            pattern[None, :, :].astype(np.float32)
        )

        return mask

    @staticmethod
    def _make_condition(
        x0: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create the partial observation C from X0.

        Observed pixels (mask == 0) retain X0.
        Hidden pixels (mask == 1) are replaced by -1, matching the current
        conditional-diffusion convention.
        """
        cond = x0.clone()

        hidden = mask.expand_as(cond) == 1
        cond[hidden] = -1.0

        return cond

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return one I2SB training pair and its conditioning observation.
        """
        x0_path = self.files[idx]

        # X0: full slice (exterior + internal structure)
        x0 = self._load_image(x0_path)

        # X1: exterior-only slice
        x1 = self.x1.clone()

        if x0.shape != x1.shape:
            raise ValueError(
                "X0 and X1 must have identical shapes for I2SB, "
                f"but got X0={tuple(x0.shape)}, X1={tuple(x1.shape)}. "
                f"X0 path: {x0_path}, X1 path: {self.exterior_only_path}"
            )

        # C: partial observation of X0
        mask = self._get_mask()
        cond = self._make_condition(
            x0=x0,
            mask=mask,
        )

        return {
            "x0"     : x0,
            "x1"     : x1,
            "cond"   : cond,
            "mask"   : mask,
            "x0_path": x0_path,
        }
