from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler


class PreNearByCellsFactory:
    def __init__(self, grid_config: Any, cache_config: Any | None = None):
        self.grid_config = grid_config
        self.cache_config = cache_config

    def create(self, mesh_components: Any):
        if self.cache_config is None:
            return None

        enabled = bool(getattr(self.cache_config, "enabled", False))
        if not enabled:
            return None

        cache_path = getattr(self.cache_config, "path", None)
        if cache_path is None:
            return None

        cache_path = Path(str(cache_path))
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            print(f"[PreNearByCellsFactory] load cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        print(f"[PreNearByCellsFactory] create cache: {cache_path}")

        tmp = voxel_cut_handler(
            grid_config       = self.grid_config,
            mesh_components   = mesh_components,
            zero_initialize   = False,
            pre_near_by_cells = None,
        )

        # pre_near_by_cells = tmp.voxel_hander._create_box_array()
        pre_near_by_cells = tmp.voxel_hander.get_box_array_data().boxes

        with open(cache_path, "wb") as f:
            pickle.dump(pre_near_by_cells, f)

        return pre_near_by_cells
