from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING


from ..types import ActionArtifacts, ActionPlan
from ..cutting_surface_planner import cutting_surface_planner

from denoising_diffusion_pytorch.utils.os_utils import create_folder
from denoising_diffusion_pytorch.utils.normalization import LimitsNormalizer
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy
from denoising_diffusion_pytorch.utils.arrays import to_torch
from denoising_diffusion_pytorch.policy.types import PlanningPolicyInput
from denoising_diffusion_pytorch.env.types import DismantlingObservation


if TYPE_CHECKING:
    from ....app.usecases.eval.types import EpisodeContext, StepOutcome


@dataclass
class LegacyPolicyPlannerAdapter:
    """
    - 既存 cutting_surface_planner.get_optimal_act(...) を包む adapter。
    - ActionPlanner と接続するためのadapter。
    """
    policy: cutting_surface_planner

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def plan_next(
        self,
        episode_ctx      : EpisodeContext,
        executed_step_idx: int,
        step_outcome     : StepOutcome,
    ) -> ActionPlan:
        execution_result = step_outcome.env_result
        self.policy.set_oracle_obs(execution_result.info.oracle_axis_images.z)


        planning_observation = self._build_planning_observation(
            episode_ctx  = episode_ctx,
            step_outcome = step_outcome,
        )

        self._save_planning_condition_image(
            planning_observation = planning_observation,
            executed_step_idx    = executed_step_idx,
            artifact_root        = str(episode_ctx.path.artifact_episodic_root),
        )

        planning_input = self._build_planning_policy_input(
            planning_observation=planning_observation,
        )

        selected_candidates, infos = self.policy.get_optimal_act(
            observation_history = execution_result.observation.observation_history,
            planning_input      = planning_input,
            iters               = executed_step_idx,
            save_path           = str(episode_ctx.path.artifact_episodic_root),
        )

        return ActionPlan(
            action_candidates = selected_candidates,
            artifacts = ActionArtifacts(
                ensemble_image=infos.get("ensemble_image"),
                debug_info={
                    "raw_infos": infos,
                },
            ),
        )


    def _build_planning_observation(
        self,
        episode_ctx  : EpisodeContext,
        step_outcome : StepOutcome,
    ):
        step_results = episode_ctx.case.envs.policy.step(
            action_idx  = step_outcome.last_executed_global_index,
            partial_obs = self.policy.visibility_constraints.to_legacy_partial_obs(),
        )
        return step_results.observation


    def _save_planning_condition_image(
        self,
        *,
        planning_observation: DismantlingObservation,
        executed_step_idx: int,
        artifact_root: str,
    ) -> None:
        cond_image_save_path = artifact_root + "/conditions/"
        create_folder(cond_image_save_path)

        pil_image_save_from_numpy(
            planning_observation.axis_images.z,
            f"{cond_image_save_path}/seq_obs_cast_{executed_step_idx}_axis_z_0.png",
        )


    def _build_planning_policy_input(
        self,
        *,
        planning_observation: DismantlingObservation,
    ):
        slice_img = planning_observation.axis_images.z # 学習とテストで固定させておく

        normalizer = LimitsNormalizer(slice_img)
        normalized_cond = normalizer.normalize(slice_img).transpose(2, 0, 1)
        normalized_cond = to_torch(normalized_cond)

        return PlanningPolicyInput(
            normalized_cond=normalized_cond,
        )
