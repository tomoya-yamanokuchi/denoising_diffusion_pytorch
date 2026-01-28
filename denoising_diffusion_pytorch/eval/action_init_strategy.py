# denoising_diffusion_pytorch/eval/strategies.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple


class ActionInitStrategy(Protocol):
    """
    エピソード開始時の action 決定を担当。
    ctrl_mode による分岐を EpisodeRunner から排除する。
    """
    def init_action(
        self,
        *,
        policy: Any,
        env2: Any,
        grid_config: Dict[str, Any],
        start_action: Any,
        episode_dir: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Returns:
            action: 初期 action（リスト/np配列など元実装互換）
            meta:   画像保存などに使う付随情報（例: ensemble_image）
        """
        ...


@dataclass(frozen=True)
class DefaultActionInit:
    """
    start_action をそのまま使う通常ルート。
    """
    def init_action(
        self,
        *,
        policy: Any,
        env2: Any,
        grid_config: Dict[str, Any],
        start_action: Any,
        episode_dir: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        # 元コード: policy.update_split_obs_config(action, grid_config)
        try:
            policy.update_split_obs_config(start_action, grid_config)
        except Exception:
            # policy がこの API を持たないケースでも eval が落ちないよう保険
            pass
        return start_action, {}


@dataclass(frozen=True)
class PriorBasedActionInit:
    """
    ctrl_mode == 'prior_based_ep_00' の特殊ルート。
    policy.get_optimal_act を使って初期 action を決める。
    """
    tmp_action: str = "prior_based_ep_00"

    def init_action(
        self,
        *,
        policy: Any,
        env2: Any,
        grid_config: Dict[str, Any],
        start_action: Any,
        episode_dir: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        action, _, infos = policy.get_optimal_act(
            slice_img_=None,
            observation_history={},
            env2=env2,
            tmp_action=self.tmp_action,
            iters=0,
            save_path=episode_dir,
        )
        meta = {}
        if isinstance(infos, dict) and "ensemble_image" in infos:
            meta["ensemble_image"] = infos["ensemble_image"]
        return action, meta


def make_action_init_strategy(ctrl_mode: str) -> ActionInitStrategy:
    if ctrl_mode == "prior_based_ep_00":
        return PriorBasedActionInit()
    return DefaultActionInit()
