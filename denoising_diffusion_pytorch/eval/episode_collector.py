from dataclasses import dataclass, field
from typing import List
from .types import StepOutcome, EpisodeRolloutSnapshot


@dataclass
class EpisodeCollector:
    _steps: List[StepOutcome] = field(default_factory=list)

    def record(self, step: StepOutcome) -> None:
        self._steps.append(step)

    def snapshot(self) -> EpisodeRolloutSnapshot:
        return EpisodeRolloutSnapshot(steps=tuple(self._steps))
