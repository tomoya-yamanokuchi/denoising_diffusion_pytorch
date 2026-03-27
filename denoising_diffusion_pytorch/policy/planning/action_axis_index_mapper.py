from typing import List, Tuple
from .types import AxisLocalIndex


class ActionAxisIndexMapper:
    def __init__(self, side_length: int):
        self.side_length = side_length

    def axis_to_offset(self, axis: str) -> int:
        if axis == "z":
            return 0
        if axis == "x":
            return self.side_length
        if axis == "y":
            return 2 * self.side_length
        raise ValueError(f"Unsupported axis: {axis}")


    def axis_local_to_global(self, axis_local: AxisLocalIndex) -> int:
        return self.axis_to_offset(axis_local.axis) + int(axis_local.index)


    def to_axis_local(self, global_index: int) -> AxisLocalIndex:
        if global_index < self.side_length:
            return AxisLocalIndex(axis="z", index=global_index)
        if global_index < 2 * self.side_length:
            return AxisLocalIndex(axis="x", index=global_index - self.side_length)
        if global_index < 3 * self.side_length:
            return AxisLocalIndex(axis="y", index=global_index - 2 * self.side_length)
        raise ValueError(f"Out of range: {global_index}")


    def infer_axis_and_offset_from_global_range(self, slice_range: list[int]) -> tuple[str, int]:
        if len(slice_range) == 0:
            raise ValueError("slice_range must not be empty.")
        # ----
        axis_local = self.to_axis_local(min(slice_range))
        offset     = self.axis_to_offset(axis_local.axis)
        return axis_local.axis, offset
