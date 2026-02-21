# app/usecases/eval_orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from omegaconf import DictConfig

from app.wiring.builder.eval_builder import EvalBuilder
from app.wiring.factories.case_context_factory import CaseContextFactory
from app.wiring.factories.episode_context_factory import EpisodeContextFactory
from denoising_diffusion_pytorch.eval.episode_runner import EpisodeRunner


@dataclass
class EvalOrchestrator:
    def __init__(self, context: EvalBuilder):
        self.cfg                     = context.cfg
        self.case_context_factory    = context.case_context_factory
        self.episode_context_factory = context.episode_context_factory
        self.episode_runner          = context.episode_runner
        self.policy                  = context.policy
        self.mesh_factory            = context.mesh_factory

    def run(self) -> Dict[str, Any]:
        cases_list = self.cfg.eval.cases
        for case_spec in cases_list:
            dataset_dir     = case_spec.dataset_dir
            mesh_components = self.mesh_factory.create(dataset_dir)
            case_ctx        = self.case_context_factory.create(case_spec, mesh_components)

            per_case: List[Any] = []
            # for k in range(self.cfg.eval.iter.start, self.cfg.eval.iter.end): # Objectごとの評価回数

            k = 0
            # ---
            ep_ctx = self.episode_context_factory.create(
                case        = case_ctx,
                policy      = self.policy,
                episode_idx = k,
            )
            # ---
            result = self.episode_runner.run(ep_ctx)
            per_case.append(result)

            results[case_ctx.name] = per_case

        return results
