from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import dismantling_env

from app.usecases.eval.types import Envs

@dataclass
class EnvFactory:
    def __init__(self, grid_config: DictConfig):
        # --- check keys ---
        if "side_length" not in grid_config:
            raise KeyError("Missing side_length (required by dismantling_env)")
        if "bounds" not in grid_config:
            raise KeyError("Missing bounds (required by dismantling_env)")
        # -----
        self.grid_config = grid_config


    def create(self, mesh_components) -> "Envs":
        eval   = dismantling_env(grid_config=self.grid_config, mesh_components=mesh_components)
        policy = dismantling_env(grid_config=self.grid_config, mesh_components=mesh_components)
        return Envs(eval=eval, policy=policy)
