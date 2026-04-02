from __future__ import annotations

from ..action_definition.action_candidates import ActionCandidates
from ...types import SliceCandidates


class SelectionPolicy:
    """
    Choose the final slice range among x/y/z candidates.

    Current policy:
        choose the longest non-None candidate.
    """

    def choose(
        self,
        candidates: SliceCandidates,
    ) -> ActionCandidates | None:
        valid_candidates = [
            c for c in [candidates.z, candidates.x, candidates.y]
            if c is not None
        ]

        if len(valid_candidates) == 0:
            return None

        return max(valid_candidates, key=len)
