# app/wiring/services/config_validator.py
from dataclasses import dataclass
from typing import Sequence, Optional
from omegaconf import DictConfig, OmegaConf

@dataclass(frozen=True)
class ConfigValidator:
    def require_keys(self, cfg: DictConfig, keys: Sequence[str], root: Optional[str] = None) -> None:
        base = OmegaConf.select(cfg, root) if root else cfg
        if base is None:
            raise KeyError(f"Missing root config: {root}")
        for k in keys:
            if k not in base:
                prefix = f"{root}." if root else ""
                raise KeyError(f"Missing key in cfg: {prefix}{k}")
