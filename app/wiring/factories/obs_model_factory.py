from __future__ import annotations
from typing import Any, Dict

from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler
from app.wiring.adapters.obs_model import VoxelObsModel
from copy import deepcopy

class VoxelObsModelFactory:
    def __init__(self, grid_config: Dict[str, Any]):
        self.grid_config = deepcopy(grid_config)

    def create(self, mesh_components: Any) -> VoxelObsModel:
        handler = voxel_cut_handler(
            grid_config     = self.grid_config,
            mesh_components = mesh_components,
            zero_initialize = True,   # policy側と同じ想定
        )
        return VoxelObsModel(handler)
