from dataclasses import dataclass
from typing import Any, Optional
from ..eval.types import EpisodeContext
from .action_decision_context import ActionDecisionContext


@dataclass
class ActionDecisionContextFactory:
    def create(self,
        episode_context     : EpisodeContext,
        step_idx            : int,
        obs                 : Optional[dict],
        last_executed_action: Optional[int],
    ) -> ActionDecisionContext:
        if step_idx == 0 or obs is None:
            obs_z               = None
            observation_history = {}
        else:
            obs_z               = obs["sequential_obs"]["z"]
            observation_history = obs["observation_history"]

        return ActionDecisionContext(
            step_idx             = step_idx,
            policy               = episode_context.policy,
            env_for_policy       = episode_context.case.envs,
            obs_z                = obs_z,
            observation_history  = observation_history,
            last_executed_action = last_executed_action,
            initial_action       = episode_context.case.start_action_idx,
            grid_config          = episode_context.grid_config,
        )
