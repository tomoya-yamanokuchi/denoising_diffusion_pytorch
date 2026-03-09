from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class ActionDecisionContext:
    step_idx           : int
    case_id            : str
    obs_z              : Any
    observation_history: dict
    env_for_policy     : Any
    save_path          : str


class ActionDecisionStrategy:
    def decide(self, ctx: ActionDecisionContext) -> list[int]:
        raise NotImplementedError
