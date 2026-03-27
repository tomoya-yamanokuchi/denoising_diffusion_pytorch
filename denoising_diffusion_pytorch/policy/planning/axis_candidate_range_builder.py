from __future__ import annotations

from .action_axis_index_mapper import ActionAxisIndexMapper
from .active_range_detector import ActiveRangeDetector
from .axis_candidate_selection_policy import AxisCandidateSelectionPolicy
from .local_candidate_range_factory import LocalCandidateRangeFactory
from .observed_action_pruner import ObservedActionPruner
from .types import (
    AxisCostVector,
    AxisLocalIndex,
    LocalAxisCandidates,
    GlobalAxisCandidates,
    OutToInSliceIndices,
)


class AxisCandidateRangeBuilder:
    """
    Build one-axis out-to-in slice indices from an AxisCostVector.

    Responsibility
    --------------
    - detect active local region
    - construct local top/bottom cutting position candidates
    - map local candidates into global action-index space
    - prune already observed global indices
    - choose one out-to-in ordered candidate
    """

    def __init__(
        self,
        mapper                 : ActionAxisIndexMapper,
        active_range_detector  : ActiveRangeDetector,
        local_candidate_factory: LocalCandidateRangeFactory,
        pruner                 : ObservedActionPruner,
        selection_policy       : AxisCandidateSelectionPolicy,
    ):
        self.mapper                  = mapper
        self.active_range_detector   = active_range_detector
        self.local_candidate_factory = local_candidate_factory
        self.pruner                  = pruner
        self.selection_policy        = selection_policy


    def build(
        self,
        axis_cost          : AxisCostVector,
        observation_history: dict[int, dict],
    ) -> OutToInSliceIndices:
        self._validate_side_length(axis_cost)

        active_range = self.active_range_detector.detect(axis_cost)

        local_axis_candidates = self.local_candidate_factory.build(
            active_range = active_range,
            side_length  = axis_cost.side_length,
        )

        if local_axis_candidates is None:
            return self.selection_policy.choose(None)

        global_axis_candidates = self._to_global_axis_candidates(
            axis                  = axis_cost.axis,
            local_axis_candidates = local_axis_candidates,
        )

        pruned_global_axis_candidates = self._prune_global_axis_candidates(
            global_axis_candidates = global_axis_candidates,
            observation_history    = observation_history,
        )

        out_to_in_slice_indices = self.selection_policy.choose(
            pruned_global_axis_candidates
        )
        return out_to_in_slice_indices


    def _to_global_axis_candidates(
        self,
        axis                 : str,
        local_axis_candidates: LocalAxisCandidates,
    ) -> GlobalAxisCandidates:
        global_top = tuple(
            self.mapper.axis_local_to_global(
                AxisLocalIndex(axis=axis, index=local_idx)
            )
            for local_idx in local_axis_candidates.top
        )
        global_bottom = tuple(
            self.mapper.axis_local_to_global(
                AxisLocalIndex(axis=axis, index=local_idx)
            )
            for local_idx in local_axis_candidates.bottom
        )

        return GlobalAxisCandidates(
            top    = global_top,
            bottom = global_bottom,
        )

    def _prune_global_axis_candidates(
        self,
        global_axis_candidates: GlobalAxisCandidates,
        observation_history: dict[int, dict],
    ) -> GlobalAxisCandidates:
        pruned_top = tuple(
            self.pruner.prune(
                candidates=global_axis_candidates.top,
                observation_history=observation_history,
            )
        )
        pruned_bottom = tuple(
            self.pruner.prune(
                candidates=global_axis_candidates.bottom,
                observation_history=observation_history,
            )
        )

        return GlobalAxisCandidates(
            top=pruned_top,
            bottom=pruned_bottom,
        )

    def _validate_side_length(self, axis_cost: AxisCostVector) -> None:
        if axis_cost.side_length != self.mapper.side_length:
            raise ValueError(
                f"AxisCostVector side_length must match mapper.side_length. "
                f"Got axis_cost.side_length={axis_cost.side_length}, "
                f"mapper.side_length={self.mapper.side_length}."
            )
