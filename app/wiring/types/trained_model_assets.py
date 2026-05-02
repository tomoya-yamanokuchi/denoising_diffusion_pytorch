from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainedModelAssets:
    infer_model: str
    inferencer : Any
    dataset    : Any
    epoch      : int
    cfg_train  : Any  # ロードしたモデルの config を eval の builder で保持しておくためのフィールド --- IGNORE ---
