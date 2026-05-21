# denoising_diffusion_pytorch/policy/planning/execution_error/factory.py
from __future__ import annotations

from typing import Any

from .boundary_uniform_error_model import BoundaryUniformExecutionErrorModel
from .no_error_model import NoActionExecutionErrorModel
from .types import ActionExecutionErrorModel


def build_action_execution_error_model(
    cfg: Any,
) -> ActionExecutionErrorModel:
    """
    Build an execution error model from Hydra / dict-like config.

    Expected config:

        enabled      : false
        mode         : boundary_uniform
        max_abs_shift: 0
        seed         : 0
    """

    enabled = bool(_cfg_get(cfg, "enabled", False))

    if not enabled:
        return NoActionExecutionErrorModel()

    mode          = str(_cfg_get(cfg, "mode", "boundary_uniform"))
    max_abs_shift = int(_cfg_get(cfg, "max_abs_shift", 0))

    seed          = _cfg_get(cfg, "seed", None)
    seed          = None if seed is None else int(seed)

    if mode == "boundary_uniform":
        return BoundaryUniformExecutionErrorModel(
            max_abs_shift=max_abs_shift,
            seed=seed,
        )

    raise ValueError(f"Unknown execution error mode: {mode}")


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default

    if isinstance(cfg, dict):
        return cfg.get(key, default)

    return getattr(cfg, key, default)
