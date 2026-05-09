from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from denoising_diffusion_pytorch.policy.types import ActionCandidates

if TYPE_CHECKING:
    from app.usecases.eval.types import CaseContext

from .closed_range import closed_range

# @dataclass(frozen=True)
class InitialActionProvider:
    def __init__(self, voxel_grid_side_length: int):
        self.voxel_grid_side_length = voxel_grid_side_length

    def provide(self, case_ctx: CaseContext) -> ActionCandidates:

        global_indices = list(closed_range(
            start = case_ctx.initial_global_action_range.start,
            stop  = case_ctx.initial_global_action_range.stop
        ))

        print(f"InitialActionProvider: Providing initial action candidates with global indices: {global_indices}")

        import ipdb; ipdb.set_trace()
        candidates = ActionCandidates.from_global_indices(
            global_indices = global_indices,
            side_length    = self.voxel_grid_side_length,
        )
        return candidates
