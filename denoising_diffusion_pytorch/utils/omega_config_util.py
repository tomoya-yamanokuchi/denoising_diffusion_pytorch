from __future__ import annotations

from typing import Any, Optional, TypeVar, Callable
from omegaconf import DictConfig, ListConfig, OmegaConf

Cfg = DictConfig  # ほぼ DictConfig を想定（ListConfigでもselectは動く）
T = TypeVar("T")


def select(cfg: Any, path: str, default: Any = None) -> Any:
    """
    Safe select for OmegaConf:
      - cfg が DictConfig/ListConfig なら OmegaConf.select を使う
      - cfg が dict なら dot-path を辿る
      - それ以外は default
    """
    if isinstance(cfg, (DictConfig, ListConfig)):
        v = OmegaConf.select(cfg, path)
        return default if v is None else v

    if isinstance(cfg, dict):
        cur: Any = cfg
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    return default


def select_str(cfg: Any, path: str, default: str = "") -> str:
    v = select(cfg, path, default=None)
    if v is None:
        return default
    return str(v)


def select_int(cfg: Any, path: str, default: int = 0) -> int:
    v = select(cfg, path, default=None)
    if v is None:
        return default
    return int(v)


def select_float(cfg: Any, path: str, default: float = 0.0) -> float:
    v = select(cfg, path, default=None)
    if v is None:
        return default
    return float(v)


def select_bool(cfg: Any, path: str, default: bool = False) -> bool:
    v = select(cfg, path, default=None)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    # "true"/"false" なども許容
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)
