from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Axis = Literal["x", "y", "z"]


@dataclass(frozen=True)
class ActionIndex:
    global_index: int
    axis        : Axis
    local_index : int
    side_length : int

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y", "z"):
            raise ValueError(f"Unsupported axis: {self.axis}")

        if self.side_length <= 0:
            raise ValueError(f"side_length must be positive: {self.side_length}")

        if not (0 <= self.local_index < self.side_length):
            raise ValueError(
                f"local_index out of range: {self.local_index} "
                f"(side_length={self.side_length})"
            )

        expected = self.offset(self.axis, self.side_length) + self.local_index
        if self.global_index != expected:
            raise ValueError(
                "Inconsistent ActionIndex: "
                f"global_index={self.global_index}, axis={self.axis}, "
                f"local_index={self.local_index}, expected={expected}"
            )

    @classmethod
    def from_global(cls, global_index: int, side_length: int) -> "ActionIndex":
        if not (0 <= global_index < 3 * side_length):
            raise ValueError(
                f"global_index out of range: {global_index} "
                f"(side_length={side_length})"
            )

        if global_index < side_length:
            axis = "z"
            local_index = global_index
        elif global_index < 2 * side_length:
            axis = "x"
            local_index = global_index - side_length
        else:
            axis = "y"
            local_index = global_index - 2 * side_length

        return cls(
            global_index = global_index,
            axis         = axis,
            local_index  = local_index,
            side_length  = side_length,
        )

    @classmethod
    def from_axis_local(
        cls,
        axis       : str,
        local_index: int,
        side_length: int,
    ) -> "ActionIndex":
        offset = cls.offset(axis, side_length)
        return cls(
            global_index = offset + local_index,
            axis         = axis,
            local_index  = local_index,
            side_length  = side_length,
        )

    @staticmethod
    def offset(axis: str, side_length: int) -> int:
        if axis == "z":
            return 0
        if axis == "x":
            return side_length
        if axis == "y":
            return 2 * side_length
        raise ValueError(f"Unsupported axis: {axis}")
