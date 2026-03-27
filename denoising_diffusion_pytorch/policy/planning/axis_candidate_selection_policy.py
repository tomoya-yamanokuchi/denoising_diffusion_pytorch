from __future__ import annotations

from .types import GlobalAxisCandidates, OutToInSliceIndices


class AxisCandidateSelectionPolicy:
    def __init__(self, empty_candidate_fallback: str = "single_zero"):
        self.empty_candidate_fallback = empty_candidate_fallback


    def choose(
        self,
        candidates: GlobalAxisCandidates | None,
    ) -> OutToInSliceIndices:
        if candidates is None:
            return self._empty_out_to_in_slice_indices()

        top_out_to_in_slice_indices = OutToInSliceIndices(
            values = tuple(candidates.top)
        )
        bottom_out_to_in_slice_indices = OutToInSliceIndices(
            values = tuple(reversed(candidates.bottom))
        )

        if (
            top_out_to_in_slice_indices.is_empty()
            and bottom_out_to_in_slice_indices.is_empty()
        ):
            return self._empty_out_to_in_slice_indices()

        if len(top_out_to_in_slice_indices) >= len(bottom_out_to_in_slice_indices):
            return top_out_to_in_slice_indices

        return bottom_out_to_in_slice_indices


    def _empty_out_to_in_slice_indices(self) -> OutToInSliceIndices:
        if self.empty_candidate_fallback == "single_zero":
            return OutToInSliceIndices(values=(0,))
        if self.empty_candidate_fallback == "empty":
            return OutToInSliceIndices(values=())
        raise ValueError(
            f"Unsupported empty_candidate_fallback: {self.empty_candidate_fallback}"
        )
