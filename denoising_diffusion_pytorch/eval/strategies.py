# denoising_diffusion_pytorch/eval/strategies.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple

from .types import ActionArtifacts


class ActionInitStrategy(Protocol):
    def init_action(
        self,
        *,
        policy: Any,
        policy_env: Any,
        grid_config: Dict[str, Any],
        start_action: Any,
        episode_dir: str,
    ) -> Tuple[Any, ActionArtifacts]:
        ...


@dataclass(frozen=True)
class DefaultActionInit:
    def init_action(
        self,
        *,
        policy: Any,
        policy_env: Any,
        grid_config: Dict[str, Any],
        start_action: Any,
        episode_dir: str,
    ) -> Tuple[Any, ActionArtifacts]:
        try:
            policy.update_split_obs_config(start_action, grid_config)
        except Exception:
            pass
        return start_action, ActionArtifacts()


@dataclass(frozen=True)
class PriorBasedActionInit:
    tmp_action: str = "prior_based_ep_00"

    def init_action(
        self,
        *,
        policy: Any,
        policy_env: Any,
        grid_config: Dict[str, Any],
        start_action: Any,
        episode_dir: str,
    ) -> Tuple[Any, ActionArtifacts]:
        action, _, infos = policy.get_optimal_act(
            slice_img_=None,
            observation_history={},
            env2=policy_env,   # policy API が env2 固定ならここだけ env2= でOK
            tmp_action=self.tmp_action,
            iters=0,
            save_path=episode_dir,
        )
        ensemble = infos.get("ensemble_image") if isinstance(infos, dict) else None
        return action, ActionArtifacts(ensemble_image=ensemble)


def make_action_init_strategy(ctrl_mode: str) -> ActionInitStrategy:
    return PriorBasedActionInit() if ctrl_mode == "prior_based_ep_00" else DefaultActionInit()
