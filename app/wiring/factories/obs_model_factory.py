from __future__ import annotations

from typing import Any, Dict
from copy import deepcopy

from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler
from app.wiring.adapters.obs_model import VoxelObsModel


class VoxelObsModelFactory:
    def __init__(self, grid_config: Dict[str, Any], pre_near_by_cells=None):
        self.grid_config       = deepcopy(grid_config)
        self.pre_near_by_cells = pre_near_by_cells

    def create(self, mesh_components: Any) -> VoxelObsModel:
        handler = voxel_cut_handler(
            grid_config       = self.grid_config,
            mesh_components   = mesh_components,
            zero_initialize   = True,
            pre_near_by_cells = self.pre_near_by_cells,
        )
        return VoxelObsModel(handler)
