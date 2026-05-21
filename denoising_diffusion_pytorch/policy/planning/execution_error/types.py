# denoising_diffusion_pytorch/policy/planning/execution_error/types.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from ..action_definition.action_candidates import ActionCandidates


@dataclass(frozen=True)
class ExecutionErrorInfo:
    enabled      : bool
    mode         : str
    max_abs_shift: int

    axis         : str
    sampled_shift: int
    applied_shift: int

    planned_boundary_local_index  : int
    executed_boundary_local_index : int
    planned_boundary_global_index : int
    executed_boundary_global_index: int

    planned_global_indices : list[int]
    executed_global_indices: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionExecutionErrorModel(Protocol):
    def apply(
        self,
        planned_candidates: ActionCandidates,
    ) -> tuple[ActionCandidates, ExecutionErrorInfo]:
        ...
