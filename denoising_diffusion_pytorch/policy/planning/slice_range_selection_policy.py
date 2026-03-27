from __future__ import annotations

from .types import SliceCandidates, OutToInSliceIndices


class SliceRangeSelectionPolicy:
    """
    Choose the final slice range among axis-wise candidates.

    Current policy
    --------------
    Select the longest candidate among z/x/y.
    This preserves the current planner behavior.
    """

    def choose(
        self,
        candidates: SliceCandidates,
    ) -> OutToInSliceIndices:
        return max(
            [candidates.z, candidates.x, candidates.y],
            key=len,
        )
