# denoising_diffusion_pytorch/eval/episode_runner.py
from __future__ import annotations

from dataclasses import dataclass

from ..types import EpisodeContext, EpisodeResult
from .episode_step_observer import EpisodeStepObserver

from denoising_diffusion_pytorch.policy.planning.action_executor import ActionExecutor


@dataclass
class EpisodeRunner:
    def __init__(self,
            action_executor: ActionExecutor,
            step_observer  : EpisodeStepObserver
        ):
        self.action_executor = action_executor
        self.step_observer   = step_observer

    def run(self, context: EpisodeContext) -> EpisodeResult:
        eval_env       = context.case.envs.eval
        policy_env     = context.case.envs.policy
        action_planner = context.action_planner

        _                 = eval_env.reset()
        env_reset_results = policy_env.reset()

        context.artifact_manager.create_episodic_artifact_root_directory()

        self.step_observer.on_episode_started(
            image_writer = context.image_writer,
            initial_info = env_reset_results.info,
        )

        action_plan = action_planner.initialize(context)

        for step_idx in range(int(context.task_step)):
            step_outcome = self.action_executor.execute(
                env               = context.case.envs.eval,
                action_candidates = action_plan.action_candidates,
            )

            self.step_observer.on_step_executed(
                episode_ctx = context,
                step_idx    = step_idx,
                step_outcome= step_outcome,
                artifacts   = action_plan.artifacts,
            )

            if step_idx == context.task_step - 1:
                break

            action_plan = action_planner.plan_next(
                episode_ctx       = context,
                executed_step_idx = step_idx,
                step_outcome      = step_outcome,
            )

        self.step_observer.on_episode_finished(context)

        # import ipdb; ipdb.set_trace()


        # return EpisodeResult(
        #     actions             = collector.actions,
        #     rewards             = collector.rewards,
        #     infos               = collector.infos,
        #     removal_performance = collector.removal_performance,
        #     intermediate_actions= collector.intermediate_actions,
        #     last_info           = collector.last_info,
        # )
