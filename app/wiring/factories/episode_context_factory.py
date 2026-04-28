# app/wiring/factories/episode_context_factory.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from app.usecases.eval.types import CaseContext, EpisodeContext
from app.usecases.eval.episode.episode_paths import EpisodePaths
from app.usecases.eval.episode.episode_image_writer import EpisodeImageWriter
from app.usecases.eval.episode.episode_artifact_manager import EpisodeArtifactManager
from denoising_diffusion_pytorch.policy.planning.action_planner import ActionPlanner

@dataclass
class EpisodeContextFactory:
    grid_config         : Dict[str, Any]
    task_step           : int
    ctrl_mode           : str # keep for completeness; may be used by policy/observer
    artifact_static_root: str

    def create(self,
            case          : CaseContext,
            action_planner: ActionPlanner,
            episode_idx   : int
        ) -> EpisodeContext:
        # ---------
        artifact_episodic_root = self._build_artifact_episodic_root(case.name, episode_idx)
        path = EpisodePaths(artifact_episodic_root)
        # ---------
        return EpisodeContext(
            case             = case,
            action_planner   = action_planner,
            grid_config      = self.grid_config,
            task_step        = self.task_step,
            ctrl_mode        = self.ctrl_mode,
            episode_idx      = episode_idx,
            path             = path,
            artifact_manager = EpisodeArtifactManager(artifact_episodic_root),
            image_writer     = EpisodeImageWriter(artifact_episodic_root)
        )

    def _build_artifact_episodic_root(self, case_name: str, episode_idx: int) -> Path:
        # import ipdb; ipdb.set_trace()
        return Path(self.artifact_static_root) / case_name / f"episode_{episode_idx}"
