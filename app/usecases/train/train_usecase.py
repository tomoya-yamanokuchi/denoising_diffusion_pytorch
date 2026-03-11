# app/usecases/eval_usecase.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from app.wiring.builder.train_builder import TrainBuilder

@dataclass
class TrainUsecase:
    def run(self, build_context: TrainBuilder) -> Any:
        # 入口は薄く：orchestratorに委譲
        return build_context.train_orchestrator.run()
