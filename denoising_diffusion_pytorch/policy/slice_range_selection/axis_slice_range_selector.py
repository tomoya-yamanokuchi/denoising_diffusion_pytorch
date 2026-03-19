import numpy as np


class AxisSliceRangeSelector:
    def select(
        self,
        cost: np.ndarray,
        axis: str,
        observation_history: dict,
    ) -> list[int]:
        offset = self._resolve_offset(cost, axis)

        start_idx, end_idx = self._find_nonzero_bounds(cost)

        top_slice_range = np.arange(0, start_idx) + offset
        bottom_slice_range = np.arange(end_idx + 1, cost.shape[0]) + offset

        observed = set(observation_history.keys())
        top_slice_range = [idx for idx in top_slice_range if idx not in observed]
        bottom_slice_range = [idx for idx in bottom_slice_range if idx not in observed]

        if len(top_slice_range) > len(bottom_slice_range):
            return top_slice_range
        if len(top_slice_range) < len(bottom_slice_range):
            return list(reversed(bottom_slice_range))
        if len(top_slice_range) == 0:
            return [0]
        return top_slice_range

    def _find_nonzero_bounds(self, cost: np.ndarray) -> tuple[int, int]:
        nonzero_indices = np.where(cost > 0)[0]
        if len(nonzero_indices) == 0:
            return 0, cost.shape[0] - 1
        return nonzero_indices[0], nonzero_indices[-1]

    def _resolve_offset(self, cost: np.ndarray, axis: str) -> int:
        if axis == "z":
            return 0
        if axis == "x":
            return cost.shape[0]
        if axis == "y":
            return cost.shape[0] * 2
        raise ValueError(f"Unknown axis: {axis}")
