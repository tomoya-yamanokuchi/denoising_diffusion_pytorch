from .types import SegmentationCostEnsemble, SegmentationCost
from .types import AxisCostEnsemble, AxisCost
import numpy as np


class SegmentationCostCollector:
    def __init__(self) -> None:
        self._ensemble: SegmentationCostEnsemble = None

    def add(self, cost: SegmentationCost) -> None:
        if self._ensemble is None:
            self._ensemble = SegmentationCostEnsemble(
                blue   = self._initialize_axis_ensemble(cost.blue),
                red    = self._initialize_axis_ensemble(cost.red),
                yellow = self._initialize_axis_ensemble(cost.yellow),
            )
            return

        self._ensemble = SegmentationCostEnsemble(
            blue   = self._append_axis_cost(self._ensemble.blue,   cost.blue),
            red    = self._append_axis_cost(self._ensemble.red,    cost.red),
            yellow = self._append_axis_cost(self._ensemble.yellow, cost.yellow),
        )

    def build(self) -> SegmentationCostEnsemble:
        if self._ensemble is None:
            raise ValueError("No segmentation cost has been collected.")
        return self._ensemble

    def _initialize_axis_ensemble(self, cost: AxisCost) -> AxisCostEnsemble:
        return AxisCostEnsemble(
            x_axis = np.expand_dims(cost.x_axis, axis=0),
            y_axis = np.expand_dims(cost.y_axis, axis=0),
            z_axis = np.expand_dims(cost.z_axis, axis=0),
        )

    def _append_axis_cost(
        self,
        ensemble: AxisCostEnsemble,
        cost    : AxisCost,
    ) -> AxisCostEnsemble:
        return AxisCostEnsemble(
            x_axis = np.vstack((ensemble.x_axis, np.expand_dims(cost.x_axis, axis=0))),
            y_axis = np.vstack((ensemble.y_axis, np.expand_dims(cost.y_axis, axis=0))),
            z_axis = np.vstack((ensemble.z_axis, np.expand_dims(cost.z_axis, axis=0))),
        )
