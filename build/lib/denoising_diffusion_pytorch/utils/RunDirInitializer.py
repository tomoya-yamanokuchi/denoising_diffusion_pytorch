from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from denoising_diffusion_pytorch.utils.omega_config_util import select_str


@dataclass(frozen=True)
class RunDirInitResult:
    run_dir             : Path
    config_path         : Path
    resolved_config_path: Path
    args_json_path      : Path
    diff_path           : Optional[Path]


class RunDirInitializer:
    """
    run_dir を実際に作って、再現性のためのスナップショットを保存する責務。
    - mkdir
    - config.yaml / config_resolved.yaml / args.json
    - (optional) git diff 保存
    """

    def __init__(
        self,
        config_filename         : str = "config.yaml",
        resolved_config_filename: str = "config_resolved.yaml",
        args_json_filename      : str = "args.json",
        diff_filename           : str = "diff.txt",
    ) -> None:
        self.config_filename          = config_filename
        self.resolved_config_filename = resolved_config_filename
        self.args_json_filename       = args_json_filename
        self.diff_filename            = diff_filename

        # cfg 上のキー（RunDirPlanner と同じ log.* に寄せる）
        self.allow_existing_key = "log.allow_existing"
        self.save_git_diff_key  = "log.save_git_diff"


    def init(self, cfg: DictConfig, run_dir: Path, exp_name: Optional[str] = None) -> RunDirInitResult:
        allow_existing = self._select_bool(cfg, self.allow_existing_key, default=True)

        # 1) mkdir
        if run_dir.exists() and not allow_existing:
            raise FileExistsError(f"run_dir already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)

        # exp_name を cfg に書き戻す（任意）
        if exp_name:
            OmegaConf.update(cfg, "log.exp_name", exp_name, merge=False)

        # 2) config.yaml（未解決のまま保存）
        config_path = run_dir / self.config_filename
        OmegaConf.save(cfg, config_path)

        # 3) resolved config（参照を解決して保存）
        resolved_config_path = run_dir / self.resolved_config_filename
        resolved_container = OmegaConf.to_container(cfg, resolve=True)
        # dict -> yaml 文字列
        resolved_yaml = OmegaConf.to_yaml(resolved_container)
        resolved_config_path.write_text(resolved_yaml, encoding="utf-8")

        # 4) args.json（resolved を JSON でも）
        args_json_path = run_dir / self.args_json_filename
        args_json_path.write_text(
            json.dumps(resolved_container, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 5) git diff（任意）
        diff_path: Optional[Path] = None
        if self._select_bool(cfg, self.save_git_diff_key, default=True):
            try:
                from denoising_diffusion_pytorch.utils.git_utils import save_git_diff
                diff_path = run_dir / self.diff_filename
                save_git_diff(str(diff_path))
            except Exception:
                diff_path = None

        return RunDirInitResult(
            run_dir=run_dir,
            config_path=config_path,
            resolved_config_path=resolved_config_path,
            args_json_path=args_json_path,
            diff_path=diff_path,
        )


    def _select_bool(self, cfg: DictConfig, path: str, default: bool) -> bool:
        v = OmegaConf.select(cfg, path)
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(v)
