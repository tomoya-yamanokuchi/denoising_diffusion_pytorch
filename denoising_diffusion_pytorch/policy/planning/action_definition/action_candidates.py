from __future__ import annotations

from dataclasses import dataclass

from .action_index import ActionIndex


@dataclass(frozen=True)
class ActionCandidates:
    values: tuple[ActionIndex, ...]

    def __post_init__(self) -> None:
        values = tuple(self.values)
        object.__setattr__(self, "values", values)

        if len(self.values) == 0:
            raise ValueError("ActionCandidates must not be empty.")

        axis = self.values[0].axis
        side_length = self.values[0].side_length

        for value in self.values:
            if value.axis != axis:
                raise ValueError("All ActionIndex values must belong to the same axis.")
            if value.side_length != side_length:
                raise ValueError("All ActionIndex values must share the same side_length.")

    @property
    def axis(self) -> str:
        return self.values[0].axis

    @property
    def side_length(self) -> int:
        return self.values[0].side_length

    @property
    def global_indices(self) -> list[int]:
        return [v.global_index for v in self.values]

    @property
    def local_indices(self) -> list[int]:
        return [v.local_index for v in self.values]

    @property
    def first(self) -> ActionIndex:
        return self.values[0]

    @property
    def last(self) -> ActionIndex:
        return self.values[-1]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, index: int) -> ActionIndex:
        return self.values[index]

    def reversed(self) -> "ActionCandidates":
        # Reverse only the sequence order of ActionIndex objects; each ActionIndex itself is unchanged.
        # Example: (12, 13, 14, 15) -> (15, 14, 13, 12)
        return ActionCandidates(values=tuple(reversed(self.values)))

    def to_list(self) -> list[int]:
        return self.global_indices

    def prune_by_observation_history(
        self,
        observation_history: dict[int, dict],
    ) -> "ActionCandidates | None":
        observed_keys = set(observation_history.keys())
        kept = tuple(v for v in self.values if v.global_index not in observed_keys)
        if len(kept) == 0:
            return None
        return ActionCandidates(values=kept)

    @classmethod
    def from_global_indices(
        cls,
        global_indices: list[int] | tuple[int, ...],
        side_length   : int,
    ) -> "ActionCandidates | None":
        if len(global_indices) == 0:
            return None
        values = tuple(ActionIndex.from_global(idx, side_length) for idx in global_indices)
        return cls(values=values)

    @classmethod
    def from_local_indices(
        cls,
        axis         : str,
        local_indices: list[int] | tuple[int, ...],
        side_length  : int,
    ) -> "ActionCandidates | None":
        if len(local_indices) == 0:
            return None
        values = tuple(
            ActionIndex.from_axis_local(
                axis=axis,
                local_index=idx,
                side_length=side_length,
            )
            for idx in local_indices
        )
        return cls(values=values)
