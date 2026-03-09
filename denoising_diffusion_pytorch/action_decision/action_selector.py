from __future__ import annotations

from dataclasses import dataclass

from .action_decision_context import ActionDecisionContext


@dataclass
class ActionSelector:
    def select(self, context: ActionDecisionContext) -> list[int]:
        if context.step_idx == 0:
            return self._select_initial_action(context)
        return self._select_recurrent_action(context)

    def _select_initial_action(self, context: ActionDecisionContext) -> list[int]:
        action = context.initial_action
        context.policy.update_split_obs_config(action, context.grid_config)
        return action

    def _select_recurrent_action(self, context: ActionDecisionContext) -> list[int]:
        action, _, _ = context.policy.get_optimal_act(
            context.obs_z,
            context.observation_history,
            context.env_for_policy,
            context.last_executed_action,
            context.step_idx,
        )
        return action
