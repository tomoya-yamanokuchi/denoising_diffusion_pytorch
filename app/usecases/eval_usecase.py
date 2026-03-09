# app/usecases/eval_usecase.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from app.wiring.builder.eval_builder import EvalBuilder

@dataclass
class EvalUsecase:
    def run(self, build_context: EvalBuilder) -> Any:
        # 入口は薄く：orchestratorに委譲
        return build_context.eval_orchestrator.run()
