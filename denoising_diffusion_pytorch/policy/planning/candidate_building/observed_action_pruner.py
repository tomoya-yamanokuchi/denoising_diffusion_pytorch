from __future__ import annotations

from ..action_definition.action_candidates import ActionCandidates


class ObservedActionPruner:
    def prune(
        self,
        candidates         : ActionCandidates | None,
        observation_history: dict[int, dict],
    ) -> ActionCandidates | None:
        if candidates is None:
            return None
        return candidates.prune_by_observation_history(observation_history)
