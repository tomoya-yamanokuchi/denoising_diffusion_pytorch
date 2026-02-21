# denoising_diffusion_pytorch/eval/episode_runner.py
from __future__ import annotations

from dataclasses import dataclass

from .types import EpisodeContext, EpisodeResult
from .collector import EpisodeCollector
from .interfaces import EpisodeObserver, OracleUpdater, StepExecutor, NextActionPolicy
from .strategies import ActionInitStrategy

from app.wiring.factories.episode_context_factory import EpisodeContextFactory


@dataclass
class EpisodeRunner:
    # def __init__(self, ctx: EpisodeContext):
        # self.a = ctx.case




    def run(self, context: EpisodeContext) -> EpisodeResult:
        env       = context.case.envs
        obs_model = context.case.obs_model

        env.eval.reset()
        env.policy.reset()

        import ipdb; ipdb.set_trace()

        policy     = ctx.policy
        eval_env   = ctx.envs.eval
        policy_env = ctx.envs.policy


        env=env,
        policy=policy,
        start_action_idx=case.start_action_idx,
        run_dir=run_dir,
        case_name=case.name,

        obs, reward, done, info = eval_env.reset()
        policy_env.reset()

        self.observer.on_reset(episode_dir=ctx.episode_dir, info=info)

        collector = self.collector_factory()

        action, init_artifacts = self.init_action.init_action(
            policy=policy,
            policy_env=policy_env,
            grid_config=ctx.grid_config,
            start_action=ctx.start_action,
            episode_dir=str(ctx.episode_dir),
        )
        self.observer.on_init_artifacts(episode_dir=ctx.episode_dir, artifacts=init_artifacts)

        for step in range(int(ctx.task_step)):
            obs, step_reward, info, last_action = self.step_executor.apply(env=eval_env, action=action)

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
