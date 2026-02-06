from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Sequence, Tuple, Union

from omegaconf import DictConfig, ListConfig, OmegaConf

_TEMPLATE_RE = re.compile(r"\{([^{}]+)\}")


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


import re

def _normalize_template_ws(s: str) -> str:
    # 1) 複数行なら連結（改行が残っているケース）
    s = "".join(line.strip() for line in s.splitlines() if line.strip())
    # 2) YAML '>' で畳まれてしまった「不要な空白」を除去
    #    特に "...} _a{...}" のような箇所を "...}_a{...}" にする
    s = re.sub(r"}\s+(_)", r"}\1", s)
    # 3) 念のため、連続空白を1つに → その後すべて消す（テンプレ用途なら安全）
    s = re.sub(r"\s+", "", s)
    return s


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

            print(f"path = {path}, label = {label}, val = {val}")

            if ("log.tag" == path) and (cfg.name == "eval"):
                val = self.render_template(
                    template=cfg.log.tag_template, cfg=cfg)

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



    @staticmethod
    def render_template(template: str, cfg: Any) -> str:
        """
        template が 'f:' で始まる場合、{path} を OmegaConf.select(cfg, path) で置換する。
        任意の python 式は実行しない（安全）。
        """
        if template is None:
            return ""

        s = str(template).strip()
        if s.startswith("f:"):
            s = s[2:]

        s = _normalize_template_ws(s)

        def repl(m: re.Match) -> str:
            path = m.group(1).strip()
            val = _select_value(cfg, path)  # 既存の _select_value を流用
            if val is None:
                return ""  # 無ければ空にする（好みで例外にしても良い）
            # dict/list は旧挙動に合わせて文字列化
            if isinstance(val, dict):
                return "_".join(f"{k}-{v}" for k, v in val.items())
            return str(val)

        out = _TEMPLATE_RE.sub(repl, s)

        # 旧 sanitize に合わせる（必要なら make() と共通関数化）
        out = out.replace("/_", "/")
        out = out.replace("(", "").replace(")", "")
        out = out.replace(", ", "-")

        return out




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
