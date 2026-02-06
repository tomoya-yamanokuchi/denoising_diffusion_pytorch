from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from omegaconf import DictConfig, OmegaConf
from denoising_diffusion_pytorch.utils.omega_config_util import select_str


def join_and_normalize(parts: list[Path]) -> Path:
    # 先頭が絶対パスならそのまま（logbaseが絶対パスのケース）
    p = parts[0]
    for x in parts[1:]:
        p = p / x

    # 危険な ".." などがあればここで落とす/潰す設計もできるが
    # まずは normalize だけ
    return p.expanduser()


def build_exp_name_from_watch(cfg: DictConfig) -> str:
    from denoising_diffusion_pytorch.utils.ExperimentNamer import ExperimentNamer
    watch    = OmegaConf.select(cfg, "watch.watch_base")
    namer    = ExperimentNamer.from_cfg(watch)
    exp_name = namer.make(cfg)
    return exp_name if exp_name else "exp"


@dataclass(frozen=True)
class RunDirPlanner:
    """
    cfg から「run_dir を決める」だけの責務。

    - exp_name が未計算なら watch から作る（ExperimentNamer を利用）
    - logbase + exp_name (+ 任意で dataset/name/suffix) などのレイアウトを適用
    - mkdir はしない（RunDirInitializer に委譲）
    """
    # logbase_key       : str = "log.logbase"
    exp_name_key      : str = "log.exp_name"
    suffix_key        : str = "log.suffix"
    layout_key        : str = "log.layout"     # optional
    dataset_class_key: str  = "dataset.class"  # optional

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "RunDirPlanner":
        # 必要なら cfg からキー名を上書きできるようにしても良いが、
        # まずは固定で十分
        return cls()

    def plan(self, cfg: DictConfig) -> Tuple[Path, str]:
        """
        Returns:
          (run_dir_path, exp_name_str)
        """

        # logbase  = select_str(cfg, self.logbase_key, default="logs")
        exp_name = select_str(cfg, self.exp_name_key, default="")

        # exp_name が未設定/空なら watch から作る
        if not exp_name:
            exp_name = build_exp_name_from_watch(cfg)

        # layout（任意）
        layout = select_str(cfg, self.layout_key, default="flat")
        """
        - flat   : logbase/exp_name
        - dataset: logbase/<dataset.name>/exp_name
        - method : logbase/<cfg.name>/exp_name（必要なら追加）
        """
        # parts = [Path(logbase)]
        parts = [Path(cfg.path.logs)]

        if layout == "dataset":
            ds_name = select_str(cfg, self.dataset_class_key, default="")
            if ds_name:
                parts.append(Path(ds_name))

        # exp_name が "vaeac/..." のようにサブディレクトリを含んでもOK
        parts.append(Path(exp_name))

        # suffix（任意）
        suffix = select_str(cfg, self.suffix_key, default="")
        if suffix:
            parts.append(Path(suffix))

        run_dir = join_and_normalize(parts)

        return run_dir, exp_name





