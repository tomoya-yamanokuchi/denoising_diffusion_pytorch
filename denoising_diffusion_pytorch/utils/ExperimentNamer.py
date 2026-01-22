from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Sequence, Tuple, Union

from omegaconf import DictConfig, ListConfig, OmegaConf


# watch の1要素は、YAMLで書くなら以下を許すと便利
# - [path, label]
# - {key: path, label: "..."}
WatchItem = Union[
    Tuple[str, str],
    Sequence[Any],               # e.g. ["dataset.image_size", "D"]
    Mapping[str, Any],           # e.g. {"key": "...", "label": "..."}
    DictConfig,                  # Hydra/OmegaConf mapping node
    ListConfig,                  # Hydra/OmegaConf sequence node
]


@dataclass(frozen=True)
class ExperimentNamer:
    """watch spec から exp_name を作るだけの小さな責務クラス。"""
    watch: List[Tuple[str, str]]

    @staticmethod
    def from_cfg(watch_spec: Iterable[WatchItem]) -> "ExperimentNamer":
        return ExperimentNamer(watch=_normalize_watch_spec(watch_spec))

    def make(self, cfg: Any) -> str:
        """
        cfg: DictConfig を推奨（OmegaConf.select が使える）
             dict / namespace でも一応動くが、dot-path は DictConfig 前提。
        """
        parts: List[str] = []

        for path, label in self.watch:
            val = _select_value(cfg, path)
            if val is None:
                continue

            # 旧実装互換：dict値は "k-v_k-v" のように連結
            if isinstance(val, dict):
                val = "_".join(f"{k}-{v}" for k, v in val.items())

            parts.append(f"{label}{val}")

        exp_name = "_".join(parts)

        # 旧 sanitize の挙動を維持
        exp_name = exp_name.replace("/_", "/")
        exp_name = exp_name.replace("(", "").replace(")", "")
        exp_name = exp_name.replace(", ", "-")

        return exp_name


def _select_value(cfg: Any, path: str) -> Any:
    """
    DictConfig の場合：OmegaConf.select で dot-path を辿る（安全に None を返す）
    dict の場合：path にドットが無い場合のみ対応（必要なら拡張可）
    namespace の場合：getattr（ドット無しのみ）
    """
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.select(cfg, path)

    # cfg が普通の dict の場合は、dot-path を簡易対応（必要最低限）
    if isinstance(cfg, dict):
        if "." not in path:
            return cfg.get(path, None)
        # dot-path を dict でも辿りたい場合（任意）
        cur: Any = cfg
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    # args-like object（SimpleNamespace等）
    if "." not in path:
        return getattr(cfg, path, None)

    # dot-path の getattr はできないので None
    return None


def _normalize_watch_spec(spec: Iterable[WatchItem]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in spec or []:
        # Hydra の DictConfig/ListConfig を普通コンテナ化
        if isinstance(item, (DictConfig, ListConfig)):
            item = OmegaConf.to_container(item, resolve=True)

        # 2要素ペア（["path", "D"] / ("path","D")）
        if isinstance(item, (list, tuple)) and len(item) == 2 and not isinstance(item, dict):
            path, label = item
            out.append((str(path), "" if label is None else str(label)))
            continue

        # dict形式（{"key": "...", "label": "..." }）
        if isinstance(item, dict):
            path = item.get("key", None)
            if path is None:
                raise KeyError("watch item must have 'key'")
            label = item.get("label", "")
            out.append((str(path), "" if label is None else str(label)))
            continue

        raise TypeError(f"Unsupported watch spec item: {type(item)}: {item}")

    return out
