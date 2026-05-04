from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from omegaconf import DictConfig

from denoising_diffusion_pytorch.utils.experiment_naming import collect_watch_entries
from denoising_diffusion_pytorch.utils.omega_config_util import select_str


def join_and_normalize(parts: list[Path]) -> Path:
    # 先頭が絶対パスならそのまま
    path = parts[0]

    for part in parts[1:]:
        path = path / part

    return path.expanduser()


def build_exp_name_from_watch(cfg: DictConfig) -> str:
    from denoising_diffusion_pytorch.utils.ExperimentNamer import ExperimentNamer
    watch = collect_watch_entries(cfg)
    namer = ExperimentNamer.from_cfg(
        watch,
        dedupe_by_key=True,
    )
    exp_name = namer.make(cfg)

    return exp_name if exp_name else "exp"


@dataclass(frozen=True)
class RunDirPlanner:
    """
    cfg から run_dir を決める Application Service。

    - log.exp_name があればそれを使う
    - なければ watch spec から exp_name を生成する
    - cfg.path.logs / layout / dataset / exp_name / control_mode から run_dir を作る
    """

    exp_name_key     : str = "log.exp_name"
    control_mode_key : str = "eval.policy.control.mode"
    layout_key       : str = "log.layout"
    dataset_class_key: str = "dataset.class"

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "RunDirPlanner":
        return cls()

    def plan(self, cfg: DictConfig) -> Tuple[Path, str]:
        exp_name = select_str(cfg, self.exp_name_key, default="")

        if not exp_name:
            exp_name = build_exp_name_from_watch(cfg)

        layout = select_str(cfg, self.layout_key, default="flat")

        parts = [Path(cfg.path.logs)]

        if layout == "dataset":
            dataset_class = select_str(cfg, self.dataset_class_key, default="")
            if dataset_class:
                parts.append(Path(dataset_class))

        # exp_name が "train/20260504/..." のようにサブディレクトリを含んでもOK
        parts.append(Path(exp_name))

        control_mode = select_str(cfg, self.control_mode_key, default="")
        if control_mode:
            parts.append(Path(control_mode))

        run_dir = join_and_normalize(parts)

        return run_dir, exp_name
