from __future__ import annotations
from dataclasses import dataclass

# from .types import MacroAction
# from ..eval.types import CaseContext
from ..types import ActionCandidates


# @dataclass(frozen=True)
class InitialActionProvider:
    def __init__(self, voxel_grid_side_length: int):
        self.voxel_grid_side_length = voxel_grid_side_length

    def provide(self, case_ctx: CaseContext) -> ActionCandidates:
        candidates = ActionCandidates.from_global_indices(
            global_indices = case_ctx.initial_global_action_indices,
            side_length    = self.voxel_grid_side_length,
        )
        return candidates
