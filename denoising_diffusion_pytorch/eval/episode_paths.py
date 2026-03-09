from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

@dataclass(frozen=True)
class EpisodePaths:
    artifact_episodic_root: Path

    @property
    def rollout_data(self) -> Path:
        return self.artifact_episodic_root / "rollout_data.pickle"

    @property
    def visualization_data(self) -> Path:
        return self.artifact_episodic_root / "visualization_data.pickle"
