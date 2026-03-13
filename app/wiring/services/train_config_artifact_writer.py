from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class TrainConfigArtifactWriter:
    """
    学習実行時の Hydra/OmegaConf 設定を、run dir 配下へ YAML として保存する責務を持つ。
    """

    def write(self, cfg: Any, artifact_static_root: str | Path) -> None:
        root = Path(artifact_static_root)
        root.mkdir(parents=True, exist_ok=True)

        config_path          = root / "config.yaml"
        resolved_config_path = root / "config_resolved.yaml"

        # 元の構造をそのまま保存
        OmegaConf.save(cfg, config_path, resolve=False)

        # interpolate / reference を解決した最終形を保存
        OmegaConf.save(cfg, resolved_config_path, resolve=True)
