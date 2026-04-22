from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..types import StepOutcome
from denoising_diffusion_pytorch.policy.planning.action_definition.action_candidates import (
    ActionCandidates,
)


@dataclass
class EpisodeCollector:
    _observations        : list[np.ndarray]      = field(default_factory=list)
    _actions             : list[int]             = field(default_factory=list)
    _rewards             : list[float]           = field(default_factory=list)
    _infos               : list[float]           = field(default_factory=list)
    _removal_performance : list[float]           = field(default_factory=list)
    _intermediate_actions: list[list[int]]       = field(default_factory=list)
    _last_info           : dict[str, Any] | None = None

    def add_step(
        self,
        step_outcome: StepOutcome,
        action_candidates: ActionCandidates,
    ) -> None:
        self._observations.append(self._extract_observation_z(step_outcome))
        self._actions.append(int(step_outcome.last_executed_global_index))
        self._rewards.append(float(step_outcome.reward))
        self._infos.append(float(step_outcome.target_removal_rate))
        self._removal_performance.append(float(step_outcome.removal_performance))
        self._intermediate_actions.append(action_candidates.to_list())
        self._last_info = self._extract_last_info(step_outcome)

    @property
    def observations(self) -> np.ndarray:
        return np.asarray(self._observations)

    @property
    def actions(self) -> np.ndarray:
        return np.asarray(self._actions)

    @property
    def rewards(self) -> np.ndarray:
        return np.asarray(self._rewards)

    @property
    def infos(self) -> np.ndarray:
        return np.asarray(self._infos)

    @property
    def removal_performance(self) -> np.ndarray:
        return np.asarray(self._removal_performance)

    @property
    def intermediate_actions(self) -> list[list[int]]:
        return list(self._intermediate_actions)

    @property
    def last_info(self) -> dict[str, Any] | None:
        return self._last_info

    def to_rollout_data(self) -> dict[str, Any]:
        return {
            "observations"       : self.observations,
            "actions"            : self.actions,
            "rewards"            : self.rewards,
            "infos"              : self.infos,
            "removal_performance": self.removal_performance,
        }

    def to_visualization_data(self) -> dict[str, Any]:
        return {
            "observations"        : self.observations,
            "actions"             : self.actions,
            "intermediate_actions": self.intermediate_actions,
        }

    def _extract_observation_z(self, step_outcome: StepOutcome) -> np.ndarray:
        observation = step_outcome.env_result.observation

        if hasattr(observation, "axis_images") and hasattr(observation.axis_images, "z"):
            return np.asarray(observation.axis_images.z)

        if isinstance(observation, dict):
            if "sequential_obs" in observation and "z" in observation["sequential_obs"]:
                return np.asarray(observation["sequential_obs"]["z"])
            if "axis_images" in observation and "z" in observation["axis_images"]:
                return np.asarray(observation["axis_images"]["z"])

        raise ValueError("Could not extract z-axis observation from StepOutcome.env_result.observation.")

    def _extract_last_info(self, step_outcome: StepOutcome) -> dict[str, Any] | None:
        env_result = step_outcome.env_result
        if hasattr(env_result, "info"):
            return env_result.info
        return None
