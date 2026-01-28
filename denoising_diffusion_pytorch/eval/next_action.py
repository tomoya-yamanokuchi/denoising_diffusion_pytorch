# denoising_diffusion_pytorch/eval/next_action.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .types import ActionArtifacts


@dataclass(frozen=True)
class DefaultNextActionPolicy:
    def next_action(
        self,
        *,
        policy: Any,
        obs: Dict[str, Any],
        policy_env: Any,
        step: int,
        prev_action: Any,
        prev_action_seq: Any,
        save_dir: str,
    ) -> Tuple[Any, ActionArtifacts]:

        if step == 0:
            next_action, _, infos = policy.get_optimal_act(
                obs["sequential_obs"]["z"],
                obs["observation_history"],
                policy_env,
                prev_action,
                step,
                save_dir,
            )
        else:
            next_action, _, infos = policy.get_optimal_act(
                obs["sequential_obs"]["z"],
                obs["observation_history"],
                policy_env,
                prev_action_seq[-1],
                step,
                save_dir,
            )

        ensemble = infos.get("ensemble_image") if isinstance(infos, dict) else None
        return next_action, ActionArtifacts(ensemble_image=ensemble)
