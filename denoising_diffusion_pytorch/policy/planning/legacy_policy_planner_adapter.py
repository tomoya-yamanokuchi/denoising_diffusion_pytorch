from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from ..types import ActionArtifacts, ActionPlan
from ..planning.action_definition.action_candidates import ActionCandidates
from ..cutting_surface_planner_v9 import cutting_surface_planner

if TYPE_CHECKING:
    from ...eval.types import EpisodeContext, StepOutcome


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

        next_action, _sorted_candidates, infos = self.policy.get_optimal_act(
            observation_history               = execution_result.observation.observation_history,
            env2                              = episode_ctx.case.envs.policy,
            last_executed_global_action_index = step_outcome.last_executed_global_index,
            iters                             = executed_step_idx,
            save_path                         = str(episode_ctx.path.artifact_episodic_root),
        )

        import ipdb; ipdb.set_trace()

        # 次に採用した action を、次回の partial observation 更新用に記録
        self.policy.update_visibility_constraints(
            candidates = ActionCandidates().from_global_indices(
                global_indices = next_action,
                side_length    = episode_ctx.grid_config.side_length
            )
        )

        return ActionPlan(
            action    = macro_action,
            artifacts = ActionArtifacts(
                ensemble_image=infos.get("ensemble_image"),
                debug_info={
                    "raw_infos": infos,
                },
            ),
        )
