# app/wiring/factories/episode_context_factory.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from denoising_diffusion_pytorch.eval.types import CaseContext, EpisodeContext

@dataclass
class EpisodeContextFactory:
    grid_config     : Dict[str, Any]
    task_step       : int
    ctrl_mode       : str  # keep for completeness; may be used by policy/observer

    def create(self, case: CaseContext, policy: Any, episode_idx) -> EpisodeContext:
        return EpisodeContext(
            case        = case,
            policy      = policy,
            grid_config = self.grid_config,
            task_step   = self.task_step,
            ctrl_mode   = self.ctrl_mode,
            episode_idx = episode_idx,
        )
