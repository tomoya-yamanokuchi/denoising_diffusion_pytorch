# app/usecases/eval_orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from tqdm import tqdm
from typing import Any, Dict, List
from app.wiring.builder.eval_builder import EvalBuilder


@dataclass
class EvalOrchestrator:
    def __init__(self, dependency: EvalBuilder):
        self.cfg                     = dependency.cfg
        self.case_context_factory    = dependency.case_context_factory
        self.episode_context_factory = dependency.episode_context_factory
        self.episode_result_writer   = dependency.episode_result_writer
        self.episode_runner          = dependency.episode_runner
        self.policy_assets           = dependency.policy_assets
        self.mesh_factory            = dependency.mesh_factory
        self.policy_factory          = dependency.policy_factory
        self.action_planner_factory  = dependency.action_planner_factory


    def run(self) -> Dict[str, Any]:
        cases_list = self.cfg.eval.cases
        results: Dict[str, Any] = {}
        for case_spec in cases_list:
            # ----------
            dataset_dir     = case_spec.dataset_dir
            mesh_components = self.mesh_factory.create(dataset_dir)
            case_ctx        = self.case_context_factory.create(case_spec, mesh_components)
            policy          = self.policy_factory.create(obs_model=case_ctx.obs_model)
            action_planner  = self.action_planner_factory.create(policy=policy)
            # ----------
            per_case: List[Any] = []
            for k in tqdm(range(self.cfg.eval.num_episodes)):
                policy.reset()
                ep_ctx = self.episode_context_factory.create(
                    case           = case_ctx,
                    action_planner = action_planner,
                    episode_idx    = k,
                )
                # ---
                episode_result = self.episode_runner.run(ep_ctx)
                self.episode_result_writer.save(ep_ctx, episode_result)
                # ---
                per_case.append(episode_result)
            # ----
            results[case_ctx.name] = per_case

        # import ipdb; ipdb.set_trace()

        return results
