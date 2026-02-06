from typing import Sequence, Optional
from omegaconf import DictConfig, OmegaConf
from app.wiring.services.config_validator import ConfigValidator


def validate_key_config(cfg: DictConfig, keys: Sequence[str]) -> None:
    validator = ConfigValidator()
    validator.require_keys(cfg, keys)
