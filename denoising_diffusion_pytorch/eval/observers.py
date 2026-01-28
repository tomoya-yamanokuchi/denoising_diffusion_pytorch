# denoising_diffusion_pytorch/eval/observers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .types import ActionArtifacts


@dataclass(frozen=True)
class NullObserver:
    def on_reset(self, *, episode_dir, info: Dict[str, Any]) -> None: pass
    def on_step_obs(self, *, episode_dir, step: int, obs: Dict[str, Any]) -> None: pass
    def on_init_artifacts(self, *, episode_dir, artifacts: ActionArtifacts) -> None: pass
    def on_step_artifacts(self, *, episode_dir, step: int, artifacts: ActionArtifacts) -> None: pass


@dataclass(frozen=True)
class ImageObserver:
    image_writer: Any

    def on_reset(self, *, episode_dir, info: Dict[str, Any]) -> None:
        self.image_writer.save_oracle(episode_dir, info["oracle_obs"])

    def on_step_obs(self, *, episode_dir, step: int, obs: Dict[str, Any]) -> None:
        self.image_writer.save_seq(episode_dir, step, obs["sequential_obs"])

    def on_init_artifacts(self, *, episode_dir, artifacts: ActionArtifacts) -> None:
        if artifacts.ensemble_image is not None:
            self.image_writer.save_ensemble(episode_dir, -1, artifacts.ensemble_image)

    def on_step_artifacts(self, *, episode_dir, step: int, artifacts: ActionArtifacts) -> None:
        if artifacts.ensemble_image is not None:
            self.image_writer.save_ensemble(episode_dir, step, artifacts.ensemble_image)
