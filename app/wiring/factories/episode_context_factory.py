# app/wiring/factories/episode_context_factory.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from denoising_diffusion_pytorch.eval.types import CaseContext, EpisodeContext
from denoising_diffusion_pytorch.eval.episode_paths import EpisodePaths
from denoising_diffusion_pytorch.eval.episode_image_writer import EpisodeImageWriter
from denoising_diffusion_pytorch.eval.episode_artifact_manager import EpisodeArtifactManager


@dataclass
class EpisodeContextFactory:
    grid_config         : Dict[str, Any]
    task_step           : int
    ctrl_mode           : str # keep for completeness; may be used by policy/observer
    artifact_static_root: str

    def create(self,
            case       : CaseContext,
            policy     : Any,
            episode_idx: int
        ) -> EpisodeContext:
        # ---------
        artifact_episodic_root = self._build_artifact_episodic_root(case.name, episode_idx)
        path = EpisodePaths(artifact_episodic_root)
        # ---------
        return EpisodeContext(
            case             = case,
            policy           = policy,
            grid_config      = self.grid_config,
            task_step        = self.task_step,
            ctrl_mode        = self.ctrl_mode,
            episode_idx      = episode_idx,
            path             = path,
            artifact_manager = EpisodeArtifactManager(artifact_episodic_root),
            image_writer     = EpisodeImageWriter(artifact_episodic_root)
        )

    def _build_artifact_episodic_root(self, case_name: str, episode_idx: int) -> Path:
        return Path(self.artifact_static_root) / case_name / f"episode_{episode_idx}"
