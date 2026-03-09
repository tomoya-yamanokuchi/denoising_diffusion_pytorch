# denoising_diffusion_pytorch/eval/episode_runner.py
from __future__ import annotations

from dataclasses import dataclass

from .types import EpisodeContext, EpisodeResult
from .episode_collector import EpisodeCollector
from .interfaces import EpisodeObserver, OracleUpdater, StepExecutor, NextActionPolicy
from .strategies import ActionInitStrategy
from app.wiring.factories.episode_context_factory import EpisodeContextFactory
from .episode_rollout_serializer import EpisodeRolloutSerializer
from .types import EpisodeRolloutSnapshot, StepOutcome
from ..action_decision.action_decision_context_factory import ActionDecisionContextFactory
from ..action_decision.action_selector import ActionSelector


@dataclass
class EpisodeRunner:
    def __init__(self):
        self._decision_context_factory = ActionDecisionContextFactory()
        self._action_selector          = ActionSelector()

    def run(self, context: EpisodeContext) -> EpisodeResult:
        # --- extract context ---
        policy           = context.policy
        env              = context.case.envs
        obs_model        = context.case.obs_model
        artifact_manager = context.artifact_manager
        image_writer     = context.image_writer

        # --- reset env ---
        _,_,_,_    = env.eval.reset()
        _,_,_,info = env.policy.reset()

        # --- artifact init ---
        artifact_manager.create_episodic_artifact_root_directory()
        image_writer.save_oracle_obs(info)


        # --- episode loop ---
        collector = EpisodeCollector()
        obs                  = None
        last_executed_action = None
        for step_idx in range(int(context.task_step)):
            decision_context = self._decision_context_factory.create(
                episode_context      = context,
                step_idx             = step_idx,
                obs                  = obs,
                last_executed_action = last_executed_action,
            )

            action_plan = self._action_selector.select(decision_context)

            import ipdb; ipdb.set_trace()


            # obs, step_reward, info, last_action = self.step_executor.apply(env=eval_env, action=action)

            self.observer.on_step_obs(episode_dir=ctx.episode_dir, step=step, obs=obs)
            self.oracle_updater.update(policy=policy, info=info)

            collector.add_step(
                last_action=last_action,
                reward=step_reward,
                info=info,
                intermediate_action=action,
            )

            action, step_artifacts = self.next_action_policy.next_action(
                policy=policy,
                obs=obs,
                policy_env=policy_env,
                step=step,
                prev_action=last_action,
                prev_action_seq=action,
                save_dir=str(ctx.episode_dir),
            )
            self.observer.on_step_artifacts(
                episode_dir=ctx.episode_dir,
                step=step,
                artifacts=step_artifacts,
            )

        return EpisodeResult(
            actions=collector.actions,
            rewards=collector.rewards,
            infos=collector.infos,
            removal_performance=collector.removal_performance,
            intermediate_actions=collector.intermediate_actions,
            last_info=collector.last_info,
        )
