from ..types.policy_assets import PolicyAssets
from denoising_diffusion_pytorch.policy.cutting_surface_planner_v9 import cutting_surface_planner

class PolicyFactory:
    def __init__(self, assets: PolicyAssets):
        self._assets = assets

    def create(self, obs_model):
        return cutting_surface_planner(
            obs_model        = obs_model,
            # ---
            inferencer       = self._assets.trained_assets.inferencer,
            trainer          = self._assets.trained_assets.trainer,
            policy_config    = self._assets.policy_config,
        )

