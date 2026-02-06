from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Envs:
    eval  : Any
    policy: Any
