from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class PolicyModelAssets:
    model  : Any
    method : Any
    trainer: Any
    dataset: Any
    epoch  : int

    @property
    def diffusion(self) -> Any:
        return self.method

    @property
    def ema(self) -> Any:
        return self.trainer.ema

    @property
    def renderer(self) -> Any:
        return None
