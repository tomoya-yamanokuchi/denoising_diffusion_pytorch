from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

# from .types import MacroAction
from denoising_diffusion_pytorch.policy.types import ActionCandidates

if TYPE_CHECKING:
    from app.usecases.eval.types import CaseContext


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
