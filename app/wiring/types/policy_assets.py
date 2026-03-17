from dataclasses import dataclass
from omegaconf import DictConfig
from .trained_model_assets import TrainedModelAssets

@dataclass(frozen=True)
class PolicyAssets:
    trained_assets: TrainedModelAssets
    policy_config : DictConfig
