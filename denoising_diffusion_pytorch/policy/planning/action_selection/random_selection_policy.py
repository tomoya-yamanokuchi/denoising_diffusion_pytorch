from __future__ import annotations

import random

from ..action_definition.action_candidates import ActionCandidates
from ...types import SliceCandidates


class RandomSelectionPolicy:
    """
    Hierarchical random action selection.

    1. Randomly choose one valid axis from z/x/y.
    2. Randomly choose one unexecuted action from that axis.

    The returned ActionCandidates contains exactly one action.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def choose(
        self,
        candidates: SliceCandidates,
    ) -> ActionCandidates | None:
        valid_axis_candidates = [
            axis_candidates
            for axis_candidates in [candidates.z, candidates.x, candidates.y]
            if axis_candidates is not None and len(axis_candidates) > 0
        ]

        if len(valid_axis_candidates) == 0:
            return None

        chosen_axis_candidates = self._rng.choice(valid_axis_candidates)
        chosen_action = self._rng.choice(tuple(chosen_axis_candidates.values))

        return ActionCandidates(values=(chosen_action,))
