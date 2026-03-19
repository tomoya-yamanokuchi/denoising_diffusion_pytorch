from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainedModelAssets:
    infer_model: str
    inferencer : Any
    trainer    : Any
    dataset    : Any
    epoch      : int
