# denoising_diffusion_pytorch/eval/collector.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeCollector:
    actions             : List[Any]                = field(default_factory=list)
    rewards             : List[float]              = field(default_factory=list)
    infos               : List[Any]                = field(default_factory=list)
    removal_performance : List[float]              = field(default_factory=list)
    intermediate_actions: List[Any]                = field(default_factory=list)
    last_info           : Optional[Dict[str, Any]] = None

    def add_step(
        self,
        *,
        last_action: Any,
        reward: float,
        info: Dict[str, Any],
        intermediate_action: Any,
    ) -> None:
        self.actions.append(last_action)
        self.rewards.append(float(reward))
        self.infos.append(info.get("target_removal_rate"))
        self.removal_performance.append(info.get("removal_performance"))
        self.intermediate_actions.append(intermediate_action)
        self.last_info = info
