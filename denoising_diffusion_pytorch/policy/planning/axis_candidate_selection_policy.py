from __future__ import annotations

from .action.action_candidates import ActionCandidates


class AxisCandidateSelectionPolicy:
    """
    Select one ordered ActionCandidates sequence within a single axis.

    Returned candidates are always interpreted as:
        outside -> inside
    """

    def choose(
        self,
        top_candidates   : ActionCandidates | None,
        bottom_candidates: ActionCandidates | None,
    ) -> ActionCandidates | None:
        top_out_to_in    = top_candidates
        bottom_out_to_in = None if bottom_candidates is None else bottom_candidates.reversed()


        if top_out_to_in is None and bottom_out_to_in is None:
            return None

        if top_out_to_in is None:
            return bottom_out_to_in

        if bottom_out_to_in is None:
            return top_out_to_in

        if len(top_out_to_in) >= len(bottom_out_to_in):
            return top_out_to_in

        return bottom_out_to_in
