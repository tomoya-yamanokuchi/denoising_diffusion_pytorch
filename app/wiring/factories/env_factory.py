from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import DictConfig

from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import dismantling_env
from app.usecases.eval.types import Envs


@dataclass
class EnvFactory:
    def __init__(
        self,
        grid_config: DictConfig,
        pre_near_by_cells: Optional[Any] = None,
        metric_calculator: Optional[Any] = None,
    ):
        if "side_length" not in grid_config:
            raise KeyError("Missing side_length (required by dismantling_env)")
        if "bounds" not in grid_config:
            raise KeyError("Missing bounds (required by dismantling_env)")

        self.grid_config = grid_config
        self.pre_near_by_cells = pre_near_by_cells
        self.metric_calculator = metric_calculator

    def create(self, mesh_components) -> "Envs":
        eval_env = dismantling_env(
            grid_config        = self.grid_config,
            mesh_components    = mesh_components,
            pre_near_by_cells  = self.pre_near_by_cells,
            metric_calculator  = self.metric_calculator,
        )

        policy_env = dismantling_env(
            grid_config        = self.grid_config,
            mesh_components    = mesh_components,
            pre_near_by_cells  = self.pre_near_by_cells,
            metric_calculator  = self.metric_calculator,
        )

        return Envs(eval=eval_env, policy=policy_env)
