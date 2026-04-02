from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..action_definition.action_candidates import ActionCandidates
from .visibility_constraint import VisibilityConstraint

Axis = Literal["x", "y", "z"]


@dataclass
class VisibilityConstraintSet:
    voxel_grid_side_length: int
    constraints: list[VisibilityConstraint] = field(default_factory=list)

    def add(self, constraint: VisibilityConstraint | None) -> None:
        if constraint is None:
            return

        # import ipdb; ipdb.set_trace()
        self.constraints.append(constraint)

    def add_from_action_candidates(
        self,
        candidates: ActionCandidates | None,
    ) -> VisibilityConstraint | None:
        constraint = VisibilityConstraint.from_action_candidates(candidates)
        self.add(constraint)
        return constraint

    def is_empty(self) -> bool:
        return len(self.constraints) == 0

    def to_legacy_partial_obs(self) -> dict[str, dict]:
        """
        Convert internal constraints to the current env.step(partial_obs=...) format.

        Legacy format example:
            {
                "[0, 2]": {"axis": "z", "range": [0, 2], "offset": 0}
            }
        """
        payload: dict[str, dict] = {}

        for c in self.constraints:
            if c.axis == "z":
                offset = 0
            elif c.axis == "x":
                offset = self.voxel_grid_side_length
            elif c.axis == "y":
                offset = 2 * self.voxel_grid_side_length
            else:
                raise ValueError(f"Unsupported axis: {c.axis}")

            key = str([c.start_local_index, c.end_local_index])
            payload[key] = {
                "axis"  : c.axis,
                "range" : [c.start_local_index, c.end_local_index],
                "offset": offset,
            }

        return payload
