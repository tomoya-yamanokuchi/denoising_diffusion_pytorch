from typing import Iterable, Sequence


class ObservedActionPruner:
    def prune(
        self,
        candidates         : Sequence[int],
        observation_history: dict[int, dict],
    ) -> list[int]:

        observed_keys = set(observation_history.keys())
        return [idx for idx in candidates if idx not in observed_keys]
