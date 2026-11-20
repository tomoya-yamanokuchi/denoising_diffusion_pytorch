from dataclasses import dataclass
from typing import Optional

from denoising_diffusion_pytorch.policy.planning.initial_action_provider import InitialActionProvider
from denoising_diffusion_pytorch.policy.planning.legacy_policy_planner_adapter import LegacyPolicyPlannerAdapter
from denoising_diffusion_pytorch.policy.planning.action_planner import ActionPlanner
from denoising_diffusion_pytorch.policy.cutting_surface_planner import cutting_surface_planner
from denoising_diffusion_pytorch.policy.planning.observation_noise import (
    ObservationLabelFlipper, ObservationLabelNoiseConfig,
)


@dataclass(frozen=True)
class ActionPlannerFactory:
    initial_action_provider: InitialActionProvider
    observation_noise      : Optional[ObservationLabelNoiseConfig] = None

    def create(self, policy: cutting_surface_planner) -> ActionPlanner:
        flipper = (ObservationLabelFlipper(self.observation_noise)
                   if self.observation_noise is not None and self.observation_noise.enabled else None)
        return ActionPlanner(
            initial_action_provider = self.initial_action_provider,
            action_planner_adapter  = LegacyPolicyPlannerAdapter(policy=policy, label_flipper=flipper),
        )
