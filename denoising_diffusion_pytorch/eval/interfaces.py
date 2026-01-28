# denoising_diffusion_pytorch/eval/interfaces.py
from __future__ import annotations
from typing import Any, Dict, Protocol, Tuple

from .types import ActionArtifacts


class EpisodeObserver(Protocol):
    def on_reset(self, *, episode_dir, info: Dict[str, Any]) -> None: ...
    def on_step_obs(self, *, episode_dir, step: int, obs: Dict[str, Any]) -> None: ...
    def on_init_artifacts(self, *, episode_dir, artifacts: ActionArtifacts) -> None: ...
    def on_step_artifacts(self, *, episode_dir, step: int, artifacts: ActionArtifacts) -> None: ...


class OracleUpdater(Protocol):
    def update(self, *, policy: Any, info: Dict[str, Any]) -> None: ...


class StepExecutor(Protocol):
    def apply(self, *, env: Any, action: Any) -> Tuple[Dict[str, Any], float, Dict[str, Any], Any]:
        """
        Returns: (obs, reward, info, last_action)
        """


class NextActionPolicy(Protocol):
    def next_action(
        self,
        *,
        policy: Any,
        obs: Dict[str, Any],
        policy_env: Any,
        step: int,
        prev_action: Any,
        prev_action_seq: Any,
        save_dir: str,
    ) -> Tuple[Any, ActionArtifacts]:
        """
        Returns: (next_action, artifacts)
        """
