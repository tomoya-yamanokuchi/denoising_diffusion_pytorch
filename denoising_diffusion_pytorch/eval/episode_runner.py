# denoising_diffusion_pytorch/eval/episode_runner.py
from __future__ import annotations

from dataclasses import dataclass

from .types import EpisodeContext, EpisodeResult
from .collector import EpisodeCollector
from .interfaces import EpisodeObserver, OracleUpdater, StepExecutor, NextActionPolicy
from .strategies import ActionInitStrategy
from ..macro_action_policy.action_decision_context import ActionDecisionContext
from app.wiring.factories.episode_context_factory import EpisodeContextFactory
from .episode_rollout_serializer import EpisodeRolloutSerializer
from .types import EpisodeRolloutSnapshot, StepOutcome


@dataclass
class EpisodeRunner:
    # def __init__(self, ctx: EpisodeContext):
        # self.a = ctx.case


    def run(self, context: EpisodeContext) -> EpisodeResult:
        # --- extract context ---
        policy    = context.policy
        env       = context.case.envs
        obs_model = context.case.obs_model

        # --- reset env ---
        env.eval.reset()
        env.policy.reset()

        import ipdb; ipdb.set_trace()

        # -------- temp param --------
        cond_save_path = "./"
        # ----------------------------


        for step in range(int(ctx.task_step)):


            ctx = ActionDecisionContext(
                step_idx            = step,
                case_id             = context.case.name,
                obs_z               = obs["sequential_obs"]["z"] if step > 0 else None,
                observation_history = obs["observation_history"] if step > 0 else {},
                env_for_policy      = env.policy,
                save_path           = cond_save_path,
            )



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
