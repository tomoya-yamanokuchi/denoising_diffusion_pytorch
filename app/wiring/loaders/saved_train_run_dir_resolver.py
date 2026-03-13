from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SavedTrainRunDirResolver:
    """
    評価設定で与えられた loadpath から、実際の学習 run directory を解決する。

    旧保存規約:
        logbase / dataset / exp_name / suffix?

    解決戦略:
    1. loadpath 自体が run dir ならそのまま採用
    2. logbase / dataset / loadpath を試す
    3. suffix があればさらに末尾へ連結した候補も試す
    4. 最初に config_resolved.yaml が見つかった候補を返す
    """

    config_filename: str = "config_resolved.yaml"

    def resolve(
        self,
        loadpath: str | Path,
        logbase: str | Path | None = None,
        dataset: str | None = None,
        suffix: str | None = None,
    ) -> Path:
        candidates = list(
            self._iter_candidates(
                loadpath=Path(loadpath),
                logbase=Path(logbase) if logbase is not None else None,
                dataset=dataset,
                suffix=suffix,
            )
        )

        for candidate in candidates:
            if self._is_valid_run_dir(candidate):
                return candidate

        joined = "\n".join(f"  - {c}" for c in candidates)
        raise FileNotFoundError(
            "Could not resolve saved train run dir.\n"
            f"Tried:\n{joined}\n"
            f"Expected `{self.config_filename}` under one of the above directories."
        )

    def _iter_candidates(
        self,
        loadpath: Path,
        logbase: Path | None,
        dataset: str | None,
        suffix: str | None,
    ) -> Iterable[Path]:
        # 1. まずはそのまま
        yield loadpath

        # 2. suffix を除いた絶対/相対候補
        if suffix:
            yield loadpath / suffix

        # 3. 旧保存規約に基づく候補
        if logbase is not None and dataset is not None:
            base = logbase / dataset / loadpath
            yield base
            if suffix:
                yield base / suffix

    def _is_valid_run_dir(self, path: Path) -> bool:
        return (path / self.config_filename).exists()
