from dataclasses import dataclass

from denoising_diffusion_pytorch.action_plan.initial_action_provider import InitialActionProvider
from denoising_diffusion_pytorch.action_plan.legacy_policy_planner_adapter import LegacyPolicyPlannerAdapter
from denoising_diffusion_pytorch.action_plan.action_planner import ActionPlanner
from denoising_diffusion_pytorch.policy.cutting_surface_planner_v9 import cutting_surface_planner


@dataclass(frozen=True)
class ActionPlannerFactory:
    initial_action_provider: InitialActionProvider

    def create(self, policy: cutting_surface_planner) -> ActionPlanner:
        return ActionPlanner(
            initial_action_provider = self.initial_action_provider,
            action_planner_adapter  = LegacyPolicyPlannerAdapter(policy=policy),
        )
