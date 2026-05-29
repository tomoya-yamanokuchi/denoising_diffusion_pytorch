from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from denoising_diffusion_pytorch.policy.planning.action_selection.action_candidates_selector import (
    ActionCandidatesSelector,
)
from denoising_diffusion_pytorch.policy.planning.action_definition.action_candidates import (
    ActionCandidates,
)
from denoising_diffusion_pytorch.utils.os_utils import pickle_utils


@dataclass
class RandomPlanningService:
    """
    Pure random planning service.

    This service does not use diffusion inference or color-based cost maps.
    It delegates action candidate generation/selection to ActionCandidatesSelector,
    which should be configured with:
      - FullActionSpaceCandidateCoordinator
      - RandomSelectionPolicy
    """

    action_candidates_selector: ActionCandidatesSelector

    def plan(
        self,
        *,
        observation_history: dict[int, dict],
        iters: int,
        save_path: str,
    ) -> tuple[ActionCandidates, dict[str, Any]]:
        selection = self.action_candidates_selector.select(
            axis_costs=None,
            observation_history=observation_history,
        )

        selected_candidates = selection.optimal_selected_slice_range

        cost_map_logs = self._build_cost_map_logs(
            selection=selection,
            selected_candidates=selected_candidates,
        )

        pickle_utils().save(
            dataset=cost_map_logs,
            save_path=save_path + f"/{iters}_cost_map_logs.pickle",
        )

        infos = {
            "ensemble_image": None,
            "planning_mode": "random",
        }

        return selected_candidates, infos

    def _build_cost_map_logs(
        self,
        *,
        selection,
        selected_candidates: ActionCandidates | None,
    ) -> dict[str, Any]:
        candidates = selection.slice_range_candidates_across_axes

        return {
            "planning_mode": "random",
            "cost_ensembles": None,
            "costs_decision": None,
            "slice_candidate": {
                "candidate_x": None if candidates.x is None else candidates.x.to_list(),
                "candidate_y": None if candidates.y is None else candidates.y.to_list(),
                "candidate_z": None if candidates.z is None else candidates.z.to_list(),
            },
            "slice_range": (
                None if selected_candidates is None else selected_candidates.to_list()
            ),
        }
