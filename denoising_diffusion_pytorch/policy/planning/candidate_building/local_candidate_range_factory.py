from __future__ import annotations

import numpy as np

from ..action_definition.action_candidates import ActionCandidates
from ...types import ActiveRange


class LocalCandidateRangeFactory:
    """
    Build axis-local candidates as ordered ActionCandidates.

    top:
        already ordered from outside -> inside
    bottom:
        ordered from inside -> outside at construction time;
        the policy will reverse it when selecting.
    """

    def build_top(
        self,
        axis        : str,
        active_range: ActiveRange | None,
        side_length : int,
    ) -> ActionCandidates | None:
        if active_range is None:
            return None

        local_indices = tuple(np.arange(0, active_range.start_index).tolist())
        return ActionCandidates.from_local_indices(
            axis          = axis,
            local_indices = local_indices,
            side_length   = side_length,
        )

    def build_bottom(
        self,
        axis        : str,
        active_range: ActiveRange | None,
        side_length : int,
    ) -> ActionCandidates | None:
        if active_range is None:
            return None

        local_indices = tuple(np.arange(active_range.end_index + 1, side_length).tolist())
        return ActionCandidates.from_local_indices(
            axis          = axis,
            local_indices = local_indices,
            side_length   = side_length,
        )
