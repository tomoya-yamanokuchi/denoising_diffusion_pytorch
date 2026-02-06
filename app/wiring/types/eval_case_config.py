from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalCaseConfig:
    """
    1ケース分の「評価入力（設定）」だけを保持する軽量オブジェクト。
    - YAMLに書かれる内容とほぼ1対1に対応させる
    - ロード済みmeshなどの“重いもの”は持たない（Builderが別で作る）
    """
    name            : str
    dataset_dir     : Path
    start_action_idx: tuple[int, ...]  # yaml list -> tuple にして不変化

    def validate(self) -> None:
        if not self.name:
            raise ValueError("EvalCaseConfig.name must be non-empty")
        if not self.dataset_dir:
            raise ValueError("EvalCaseConfig.dataset_dir must be set")
        if len(self.start_action_idx) == 0:
            raise ValueError(f"EvalCaseConfig.start_action_idx is empty (case={self.name})")
