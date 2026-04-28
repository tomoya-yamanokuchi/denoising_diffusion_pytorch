# app/wiring/services/run_dir_manager.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from omegaconf import DictConfig, OmegaConf

from denoising_diffusion_pytorch.utils.RunDirPlanner import RunDirPlanner
from denoising_diffusion_pytorch.utils.RunDirInitializer import RunDirInitializer


@dataclass
class RunDirManager:
    planner    : RunDirPlanner
    initializer: RunDirInitializer

    def plan(self, cfg: DictConfig) -> Tuple[Path, str]:
        run_dir, exp_name = self.planner.plan(cfg)
        OmegaConf.update(cfg, "log.exp_name", exp_name, merge=False)
        return Path(run_dir), exp_name

    def init(self, cfg: DictConfig, run_dir: Path, exp_name: Optional[str]) -> None:
        # import ipdb; ipdb.set_trace()
        self.initializer.init(cfg, run_dir, exp_name=exp_name)
