from __future__ import annotations

import numpy as np

from .types import ActiveRange, LocalAxisCandidates


class LocalCandidateRangeFactory:
    def build(
        self,
        active_range: ActiveRange,
        side_length : int,
    ) -> LocalAxisCandidates:

        if active_range is None:
            return None

        # top    = tuple(np.arange(0, active_range.start_index-1).tolist()) # 旧実装そのまま:  off-by-one バグ
        top    = tuple(np.arange(0, active_range.start_index).tolist()) # 旧実装からの修正
        bottom = tuple(np.arange(active_range.end_index + 1, side_length).tolist())

        return LocalAxisCandidates(
            top    = top,
            bottom = bottom,
        )
