from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from ..eval.types import EpisodeContext, StepOutcome
from ..action_plan.types import ActionArtifacts
from ..eval.episode_image_writer import EpisodeImageWriter


@dataclass
class EpisodeStepObserver:
    verbose: bool              = True
    _steps : list[StepOutcome] = field(default_factory=list)

    def on_episode_started(self, image_writer: EpisodeImageWriter, initial_info: dict[str, Any]) -> None:
        image_writer.save_oracle_obs(initial_info)

    def on_step_executed(
            self,
            episode_ctx: EpisodeContext,
            step_idx   : int,
            outcome    : StepOutcome,
            obs        : dict[str, Any],
            artifacts  : ActionArtifacts,
        ) -> None:
        self._steps.append(outcome)

        self._log_step(
            episode_ctx=episode_ctx,
            step_idx=step_idx,
            outcome=outcome,
        )

        episode_ctx.image_writer.save_seq_obs(step_idx, seq_obs=obs["sequential_obs"])
        self._save_ensemble_image(episode_ctx, step_idx, artifacts)


    def on_episode_finished(
        self,
        episode_ctx: EpisodeContext,
    ) -> None:
        # ここでは何もしない。
        # rollout_data / visualization_data の保存は別の episode_result_writer に分けてもよい。
        pass

    def _log_step(
        self,
        episode_ctx: EpisodeContext,
        step_idx: int,
        outcome: StepOutcome,
    ) -> None:
        if not self.verbose:
            return

        print("#" * 120)
        print(
            f"{episode_ctx.case.name} | "
            f"Ep.: {episode_ctx.episode_idx} | "
            f"step: {step_idx} | "
            f"cut_cost: {outcome.reward} | "
            f"target_removal_rate: {outcome.target_removal_rate} | "
            f"removal_performance: {outcome.removal_performance:.3f}"
        )
        print(f"macro_action: {outcome.macro_action}")
        print("#" * 120)


    def _save_ensemble_image(self,
        episode_ctx: EpisodeContext,
        step_idx   : int,
        artifacts  : ActionArtifacts,
    ) -> None:
        if artifacts.ensemble_image is None:
            return
        episode_ctx.image_writer.save_ensemble_image(step_idx, artifacts.ensemble_image)
