from __future__ import annotations
from dataclasses import dataclass

from .types import MacroAction
# from ..eval.types import CaseContext


@dataclass(frozen=True)
class InitialActionProvider:
    def provide(self, case_ctx: CaseContext) -> MacroAction:
        return MacroAction(tuple(case_ctx.start_action_idx))
