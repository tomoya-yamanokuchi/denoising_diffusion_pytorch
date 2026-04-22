from dataclasses import dataclass
from typing import Any, Dict
from omegaconf import DictConfig
from app.wiring.factories.env_factory import EnvFactory
from app.wiring.factories.obs_model_factory import VoxelObsModelFactory
from app.usecases.eval.types import CaseContext

@dataclass
class CaseContextFactory:
    env_factory      : EnvFactory
    obs_model_factory: VoxelObsModelFactory

    def create(self, cfg_case: DictConfig, mesh_components: Any) -> CaseContext:
        envs      = self.env_factory.create(mesh_components)
        obs_model = self.obs_model_factory.create(mesh_components)
        # ---
        return CaseContext(
            name                          = cfg_case.name,
            dataset_dir                   = cfg_case.dataset_dir,
            initial_global_action_indices = cfg_case.initial_global_action_indices,
            mesh_components               = mesh_components,
            envs                          = envs,
            obs_model                     = obs_model,
        )
