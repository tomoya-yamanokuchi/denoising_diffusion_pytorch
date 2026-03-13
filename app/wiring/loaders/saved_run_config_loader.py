from dataclasses import dataclass
from pathlib import Path
from omegaconf import OmegaConf

@dataclass
class SavedRunConfigLoader:
    def load(self, run_dir: str):
        run_dir = Path(run_dir)
        path = run_dir / "config_resolved.yaml"

        if not path.exists():
            raise FileNotFoundError(f"config_resolved.yaml not found: {path}")
        return OmegaConf.load(path)
