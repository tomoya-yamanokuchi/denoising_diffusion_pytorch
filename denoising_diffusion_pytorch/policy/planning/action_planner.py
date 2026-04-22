from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .initial_action_provider import InitialActionProvider
from .legacy_policy_planner_adapter import LegacyPolicyPlannerAdapter
from ..types import ActionPlan
from ..types import ActionArtifacts

if TYPE_CHECKING:
    from ....app.usecases.eval.types import EpisodeContext, StepOutcome


class ActionPlanner:
    def __init__(self,
            initial_action_provider: InitialActionProvider,
            action_planner_adapter : LegacyPolicyPlannerAdapter,
        ):
        self.initial_action_provider = initial_action_provider
        self.action_planner_adapter  = action_planner_adapter


    def initialize(self, episode_ctx: EpisodeContext) -> ActionPlan:
        # ---
        initial_action_candidates = self.initial_action_provider.provide(episode_ctx.case)
        self.action_planner_adapter.policy.update_visibility_constraints(initial_action_candidates)
        # ---
        return ActionPlan(
            action_candidates = initial_action_candidates,
            artifacts         = ActionArtifacts(),
        )


    def plan_next(self,
        episode_ctx      : EpisodeContext,
        executed_step_idx: int,
        step_outcome     : StepOutcome,
    ) -> ActionPlan:
        return self.action_planner_adapter.plan_next(
            episode_ctx       = episode_ctx,
            executed_step_idx = executed_step_idx,
            step_outcome      = step_outcome,
        )
