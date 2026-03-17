# denoising_diffusion_pytorch/eval/episode_runner.py
from __future__ import annotations

from dataclasses import dataclass

from .types import EpisodeContext, EpisodeResult
from .episode_collector import EpisodeCollector
from app.wiring.factories.episode_context_factory import EpisodeContextFactory
from .episode_rollout_serializer import EpisodeRolloutSerializer
from .types import EpisodeRolloutSnapshot, StepOutcome
from ..action_plan.action_planner import ActionPlanner
from ..action_plan.macro_action_executor import MacroActionExecutor
from ..observer.episode_step_observer import EpisodeStepObserver

@dataclass
class EpisodeRunner:
    def __init__(self,
            action_executor: MacroActionExecutor,
            step_observer  : EpisodeStepObserver
        ):
        self.action_executor = action_executor
        self.step_observer   = step_observer

    def run(self, context: EpisodeContext) -> EpisodeResult:
        eval_env       = context.case.envs.eval
        policy_env     = context.case.envs.policy
        action_planner = context.action_planner

        _, _, _, _     = eval_env.reset()
        _, _, _, info  = policy_env.reset()

        context.artifact_manager.create_episodic_artifact_root_directory()

        self.step_observer.on_episode_started(
            image_writer = context.image_writer,
            initial_info = info,
        )


        plan = action_planner.initialize(context)

        for step_idx in range(int(context.task_step)):
            outcome, obs, info = self.action_executor.execute(
                env          = context.case.envs.eval,
                macro_action = plan.action,
            )

            self.step_observer.on_step_executed(
                episode_ctx = context,
                step_idx    = step_idx,
                outcome     = outcome,
                obs         = obs,
                artifacts   = plan.artifacts,
            )

            if step_idx == context.task_step - 1:
                break

            plan = action_planner.plan_next(
                episode_ctx       = context,
                executed_step_idx = step_idx,
                executed_step     = outcome,
                last_obs          = obs,
                last_info         = info,
            )

        self.step_observer.on_episode_finished(context)


        import ipdb; ipdb.set_trace()

        return EpisodeResult(
            actions=collector.actions,
            rewards=collector.rewards,
            infos=collector.infos,
            removal_performance=collector.removal_performance,
            intermediate_actions=collector.intermediate_actions,
            last_info=collector.last_info,
        )
