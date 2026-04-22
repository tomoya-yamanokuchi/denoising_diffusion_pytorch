# app/usecases/eval_orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from omegaconf import DictConfig

from app.wiring.builder.train_builder import TrainBuilder


@dataclass
class TrianOrchestrator:
    def __init__(self, dependency: TrainBuilder):
        self.cfg     = dependency.cfg
        self.trainer = dependency.trainer

    def run(self) -> Dict[str, Any]:
        self.trainer.train()
