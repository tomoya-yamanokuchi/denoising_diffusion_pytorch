# denoising_diffusion_pytorch/policy/planning/action_selection/random_contiguous_selection_policy.py
from __future__ import annotations

import random
import secrets

from ..action_definition.action_candidates import ActionCandidates
from ...types import SliceCandidates


class RandomContiguousSelectionPolicy:
    """
    Randomly select one contiguous cutting range.

    Procedure:
      1. Randomly choose one valid axis from z/x/y.
      2. Randomly choose top-side or bottom-side if feasible.
      3. Randomly choose a contiguous range length up to max_range_length.
      4. Return the selected contiguous range as ActionCandidates.

    This policy does not use diffusion-based cost or presence information.
    """

    def __init__(
        self,
        seed: int | None = None,
        max_range_length: int | None = 3,
    ):
        if max_range_length is not None and max_range_length <= 0:
            raise ValueError(
                f"max_range_length must be positive or None, "
                f"got {max_range_length}."
            )

        self.seed = int(seed) if seed is not None else secrets.randbits(32)
        self.max_range_length = max_range_length
        self._rng = random.Random(self.seed)

    def choose(
        self,
        candidates: SliceCandidates,
    ) -> ActionCandidates | None:
        axis_infos = []

        for axis_candidates in [candidates.z, candidates.x, candidates.y]:
            if axis_candidates is None or len(axis_candidates) == 0:
                continue

            runs = self._build_feasible_edge_runs(axis_candidates)
            if len(runs) == 0:
                continue

            axis_infos.append((axis_candidates, runs))

        if len(axis_infos) == 0:
            return None

        chosen_axis_candidates, feasible_runs = self._rng.choice(axis_infos)
        chosen_run = self._rng.choice(feasible_runs)

        max_selectable_length = len(chosen_run)
        if self.max_range_length is not None:
            max_selectable_length = min(
                max_selectable_length,
                self.max_range_length,
            )

        selected_length        = self._rng.randint(1, max_selectable_length)
        selected_local_indices = tuple(chosen_run[:selected_length])

        return ActionCandidates.from_local_indices(
            axis=chosen_axis_candidates.axis,
            local_indices=selected_local_indices,
            side_length=chosen_axis_candidates.side_length,
        )

    def _build_feasible_edge_runs(
        self,
        candidates: ActionCandidates,
    ) -> list[list[int]]:
        available = set(candidates.local_indices)
        side_length = candidates.side_length

        top_run = self._build_run_from_top_edge(
            available=available,
            side_length=side_length,
        )

        bottom_run = self._build_run_from_bottom_edge(
            available=available,
            side_length=side_length,
        )

        runs = []
        if len(top_run) > 0:
            runs.append(top_run)
        if len(bottom_run) > 0:
            runs.append(bottom_run)

        return runs

    def _build_run_from_top_edge(
        self,
        *,
        available: set[int],
        side_length: int,
    ) -> list[int]:
        idx = 0

        while idx < side_length and idx not in available:
            idx += 1

        run = []
        while idx < side_length and idx in available:
            run.append(idx)
            idx += 1

        return run

    def _build_run_from_bottom_edge(
        self,
        *,
        available: set[int],
        side_length: int,
    ) -> list[int]:
        idx = side_length - 1

        while idx >= 0 and idx not in available:
            idx -= 1

        run = []
        while idx >= 0 and idx in available:
            run.append(idx)
            idx -= 1

        return run
