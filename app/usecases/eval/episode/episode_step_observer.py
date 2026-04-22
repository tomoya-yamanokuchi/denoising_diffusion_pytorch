from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from ..types import EpisodeContext, StepOutcome
from denoising_diffusion_pytorch.policy.types import ActionArtifacts
from .episode_image_writer import EpisodeImageWriter


@dataclass
class EpisodeStepObserver:
    verbose: bool              = True
    _steps : list[StepOutcome] = field(default_factory=list)

    def on_episode_started(self, image_writer: EpisodeImageWriter, initial_info: dict[str, Any]) -> None:
        image_writer.save_oracle_obs(initial_info)

    def on_step_executed(
            self,
            episode_ctx : EpisodeContext,
            step_idx    : int,
            step_outcome: StepOutcome,
            artifacts   : ActionArtifacts,
        ) -> None:
        self._steps.append(step_outcome)

        self._log_step(
            episode_ctx  = episode_ctx,
            step_idx     = step_idx,
            step_outcome = step_outcome,
        )
        episode_ctx.image_writer.save_seq_obs(
            step_idx = step_idx,
            seq_obs  = step_outcome.env_result.observation
        )
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
        episode_ctx : EpisodeContext,
        step_idx    : int,
        step_outcome: StepOutcome,
    ) -> None:
        if not self.verbose:
            return

        print("#" * 120)
        print(
            f"{episode_ctx.case.name} | "
            f"Ep.: {episode_ctx.episode_idx} | "
            f"step: {step_idx} | "
            f"cut_cost: {step_outcome.reward} | "
            f"target_removal_rate: {step_outcome.target_removal_rate} | "
            f"removal_performance: {step_outcome.removal_performance:.3f}"
        )
        print(f"executed_action_candidates: {step_outcome.executed_action_candidates.to_list()}")
        print("#" * 120)


    def _save_ensemble_image(self,
        episode_ctx: EpisodeContext,
        step_idx   : int,
        artifacts  : ActionArtifacts,
    ) -> None:
        if artifacts.ensemble_image is None:
            return
        episode_ctx.image_writer.save_ensemble_image(step_idx, artifacts.ensemble_image)
