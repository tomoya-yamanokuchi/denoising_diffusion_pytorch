from __future__ import annotations

from .initial_action_provider import InitialActionProvider
from .legacy_policy_planner_adapter import LegacyPolicyPlannerAdapter
from .types import ActionPlan
# from ..eval.types import StepOutcome, EpisodeContext
from .types import ActionArtifacts


class ActionPlanner:
    def __init__(self,
            initial_action_provider: InitialActionProvider,
            action_planner_adapter : LegacyPolicyPlannerAdapter,
        ):
        self.initial_action_provider = initial_action_provider
        self.action_planner_adapter  = action_planner_adapter


    def initialize(self, episode_ctx: EpisodeContext) -> ActionPlan:
        initial_action = self.initial_action_provider.provide(episode_ctx.case)

        self.action_planner_adapter.policy.update_split_obs_config(
            list(initial_action.indices),
            episode_ctx.grid_config,
        )

        return ActionPlan(
            action    = initial_action,
            artifacts = ActionArtifacts(),
        )


    def plan_next(self,
        episode_ctx      : EpisodeContext,
        executed_step_idx: int,
        executed_step    : StepOutcome,
        last_obs         : dict,
        last_info        : dict,
    ) -> ActionPlan:
        return self.action_planner_adapter.plan_next(
            episode_ctx       = episode_ctx,
            executed_step_idx = executed_step_idx,
            executed_step     = executed_step,
            last_obs          = last_obs,
            last_info         = last_info,
        )
