from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .types import ActionArtifacts, ActionPlan, MacroAction
# from ..eval.types import EpisodeContext, StepOutcome
from ..policy.cutting_surface_planner_v9 import cutting_surface_planner


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
        executed_step    : StepOutcome,
        last_obs         : dict,
        last_info        : dict,
    ) -> ActionPlan:
        self.policy.set_oracle_obs(last_info["oracle_obs"]["z"])

        next_action, _sorted_candidates, infos = self.policy.get_optimal_act(
            slice_img_          = last_obs["sequential_obs"]["z"],
            observation_history = last_obs["observation_history"],
            env2                = episode_ctx.case.envs.policy,
            tmp_action          = executed_step.last_action,
            iters               = executed_step_idx,
            save_path           = str(episode_ctx.path.artifact_episodic_root),
        )

        macro_action = MacroAction(tuple(next_action))

        # 次に採用した action を、次回の partial observation 更新用に記録
        self.policy.update_split_obs_config(
            list(macro_action.indices),
            episode_ctx.grid_config,
        )

        return ActionPlan(
            action=macro_action,
            artifacts=ActionArtifacts(
                ensemble_image=infos.get("ensemble_image"),
                debug_info={
                    "raw_infos": infos,
                },
            ),
        )
