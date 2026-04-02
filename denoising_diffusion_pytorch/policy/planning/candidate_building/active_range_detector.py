from __future__ import annotations

import numpy as np

from ...types import ActiveRange, AxisCostVector


class ActiveRangeDetector:
    def __init__(self, cost_threshold: float = 0.0):
        self.cost_threshold = float(cost_threshold)


    def detect(self, axis_cost: AxisCostVector) -> ActiveRange:
        values = axis_cost.values
        mask   = (values > self.cost_threshold)

        if not np.any(mask):
            return None

        start_index = int(np.argmax(mask))
        end_index   = int(len(values) - np.argmax(mask[::-1]) - 1)

        return ActiveRange(start_index=start_index, end_index=end_index)
