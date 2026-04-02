from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..action_definition.action_candidates import ActionCandidates

Axis = Literal["x", "y", "z"]


@dataclass(frozen=True)
class VisibilityConstraint:
    """
    One future-visibility constraint induced by a selected outside->inside action sequence.

    Semantics
    ---------
    - ActionCandidates are ordered outside -> inside.
    - The last action is interpreted as the retained cut surface.
    - All preceding actions (toward the outside side) become the hidden interval.
    """
    axis             : Axis
    start_local_index: int
    end_local_index  : int

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y", "z"):
            raise ValueError(f"Unsupported axis: {self.axis}")
        if self.start_local_index > self.end_local_index:
            raise ValueError(
                "start_local_index must be <= end_local_index. "
                f"Got {self.start_local_index} > {self.end_local_index}"
            )

    @property
    def local_range(self) -> tuple[int, int]:
        return (self.start_local_index, self.end_local_index)

    @classmethod
    def from_action_candidates(
        cls,
        candidates: ActionCandidates | None,
    ) -> "VisibilityConstraint | None":
        """
        Convert an outside->inside ordered ActionCandidates into one visibility constraint.

        Rule
        ----
        - len <= 1 : no hidden interval exists
        - otherwise, values[:-1] are interpreted as the newly hidden outer interval
        - the last element is kept as the visible cut surface
        """
        if candidates is None:
            return None

        if len(candidates) <= 1:
            return None

        hidden_outer      = candidates.values[:-1]
        start_local_index = min(v.local_index for v in hidden_outer)
        end_local_index   = max(v.local_index for v in hidden_outer)

        return cls(
            axis              = candidates.axis,
            start_local_index = start_local_index,
            end_local_index   = end_local_index,
        )
