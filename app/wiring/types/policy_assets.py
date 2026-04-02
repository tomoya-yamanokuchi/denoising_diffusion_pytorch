from dataclasses import dataclass
from omegaconf import DictConfig
from .trained_model_assets import TrainedModelAssets
from denoising_diffusion_pytorch.policy.types import PolicyConfig

@dataclass(frozen=True)
class PolicyAssets:
    trained_assets: TrainedModelAssets
    policy_config : PolicyConfig
