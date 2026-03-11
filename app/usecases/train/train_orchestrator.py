# app/usecases/eval_orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from omegaconf import DictConfig

from app.wiring.builder.train_builder import TrainBuilder
from app.wiring.factories.case_context_factory import CaseContextFactory
from app.wiring.factories.episode_context_factory import EpisodeContextFactory
from denoising_diffusion_pytorch.eval.episode_runner import EpisodeRunner


@dataclass
class TrianOrchestrator:
    def __init__(self, dependency: TrainBuilder):
        self.cfg     = dependency.cfg
        self.trainer = dependency.trainer

    def run(self) -> Dict[str, Any]:
        self.trainer.train()
