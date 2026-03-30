from __future__ import annotations

from .action.action_candidates import ActionCandidates
from .active_range_detector import ActiveRangeDetector
from .axis_candidate_selection_policy import AxisCandidateSelectionPolicy
from .local_candidate_range_factory import LocalCandidateRangeFactory
from .observed_action_pruner import ObservedActionPruner
from .types import AxisCostVector

from pprint import pprint

class AxisCandidateRangeBuilder:
    """
    Build one-axis ordered action candidates from an AxisCostVector.
    """

    def __init__(
        self,
        active_range_detector: ActiveRangeDetector,
        local_candidate_factory: LocalCandidateRangeFactory,
        pruner: ObservedActionPruner,
        selection_policy: AxisCandidateSelectionPolicy,
        expected_side_length: int,
    ):
        self.active_range_detector = active_range_detector
        self.local_candidate_factory = local_candidate_factory
        self.pruner = pruner
        self.selection_policy = selection_policy
        self.expected_side_length = int(expected_side_length)

    def build(
        self,
        axis_cost: AxisCostVector,
        observation_history: dict[int, dict],
    ) -> ActionCandidates | None:
        self._validate_side_length(axis_cost)

        active_range = self.active_range_detector.detect(axis_cost)


        '''
            build top and bottom candidates
        '''
        top_candidates = self.local_candidate_factory.build_top(
            axis=axis_cost.axis,
            active_range=active_range,
            side_length=axis_cost.side_length,
        )
        bottom_candidates = self.local_candidate_factory.build_bottom(
            axis=axis_cost.axis,
            active_range=active_range,
            side_length=axis_cost.side_length,
        )

        '''
            prune past actons
        '''
        pruned_top_candidates = self.pruner.prune(
            candidates          = top_candidates,
            observation_history = observation_history,
        )
        pruned_bottom_candidates = self.pruner.prune(
            candidates          = bottom_candidates,
            observation_history = observation_history,
        )

        return self.selection_policy.choose(
            top_candidates    = pruned_top_candidates,
            bottom_candidates = pruned_bottom_candidates,
        )

    def _validate_side_length(self, axis_cost: AxisCostVector) -> None:
        if axis_cost.side_length != self.expected_side_length:
            raise ValueError(
                f"AxisCostVector side_length must match expected_side_length. "
                f"Got axis_cost.side_length={axis_cost.side_length}, "
                f"expected_side_length={self.expected_side_length}."
            )
