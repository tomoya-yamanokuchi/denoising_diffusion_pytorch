from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyModelAssets:
    """
    case 非依存で共有される、学習済みモデル群。
    policy 本体ではなく、policy を構成するための重い推論資産を表す。
    """
    diffusion: Any
    dataset  : Any
    trainer  : Any
