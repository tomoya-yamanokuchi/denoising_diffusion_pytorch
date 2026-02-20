from dataclasses import dataclass
from typing import Any, Dict
from omegaconf import DictConfig
from app.wiring.factories.env_factory import EnvFactory
from app.wiring.factories.obs_model_factory import VoxelObsModelFactory
from denoising_diffusion_pytorch.eval.types import CaseContext

@dataclass
class CaseContextFactory:
    env_factory      : EnvFactory
    obs_model_factory: VoxelObsModelFactory

    def create(self, cfg_case: DictConfig, mesh_components: Any) -> CaseContext:
        envs      = self.env_factory.create(mesh_components)
        obs_model = self.obs_model_factory.create(cfg_case, mesh_components)
        # ---
        return CaseContext(
            name             = cfg_case.name,
            dataset_dir      = cfg_case.dataset_dir,
            start_action_idx = cfg_case.start_action_idx,
            mesh_components  = mesh_components,
            envs             = envs,
            obs_model        = obs_model,
        )
