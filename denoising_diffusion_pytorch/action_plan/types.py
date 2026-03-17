# denoising_diffusion_pytorch/eval/types.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple



@dataclass(frozen=True)
class MacroAction:
    indices: Tuple[int, ...]

    @property
    def last_atomic(self) -> int:
        return self.indices[-1]

@dataclass(frozen=True)
class ActionArtifacts:
    ensemble_image: Optional[dict[str, Any]] = None
    debug_info    : Optional[dict[str, Any]] = None

@dataclass(frozen=True)
class ActionPlan:
    action   : MacroAction
    artifacts: ActionArtifacts
