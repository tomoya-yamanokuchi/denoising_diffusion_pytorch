from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainedModelAssets:
    inferencer: Any
    trainer   : Any
    dataset   : Any
    epoch     : int
