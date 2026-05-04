from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

from denoising_diffusion_pytorch.utils.experiment_naming.name_part_formatter import (
    NamePartFormatter,
)
from .watch_entry import WatchEntry



_TEMPLATE_RE = re.compile(r"\{([^{}]+)\}")


class WatchValueResolver:
    """
    WatchEntry を実際の値へ解決する Domain Service。

    - config  : cfg から key で値を読む
    - date    : datetime.now() から値を作る
    - literal : entry.value をそのまま返す
    """

    def __init__(self, formatter: Optional[NamePartFormatter] = None) -> None:
        self.formatter = formatter or NamePartFormatter()

    def resolve(self, entry: WatchEntry, cfg: Any) -> Any:
        if entry.kind == "date":
            # return datetime.now().strftime(entry.fmt)
            return datetime.now(ZoneInfo(entry.timezone)).strftime(entry.fmt)

        if entry.kind == "literal":
            return entry.value

        if entry.kind == "config":
            return self._resolve_config_value(entry, cfg)

        raise ValueError(f"Unsupported watch entry kind: {entry.kind}")

    def _resolve_config_value(self, entry: WatchEntry, cfg: Any) -> Any:
        if entry.key is None:
            return None

        # eval時の log.tag は tag_template から生成する既存仕様を維持
        if entry.key == "log.tag" and select_value(cfg, "name") == "eval":
            return self.render_template(
                template=select_value(cfg, "log.tag_template"),
                cfg=cfg,
            )

        return select_value(cfg, entry.key)

    def render_template(self, template: str, cfg: Any) -> str:
        """
        template が 'f:' で始まる場合、{path} を cfg の値で置換する。
        Python の任意式は実行しない。
        """
        if template is None:
            return ""

        text = str(template).strip()

        if text.startswith("f:"):
            text = text[2:]

        text = _normalize_template_ws(text)

        def repl(match: re.Match) -> str:
            path = match.group(1).strip()
            value = select_value(cfg, path)

            if value is None:
                return ""

            return self.formatter.format(value)

        out = _TEMPLATE_RE.sub(repl, text)

        # template 単体でも旧 sanitize 相当を軽く適用
        out = out.replace("/_", "/")
        out = out.replace("(", "").replace(")", "")
        out = out.replace(", ", "-")

        return out


def select_value(cfg: Any, path: str) -> Any:
    """
    DictConfig / dict / namespace から dot path で値を読む。
    """
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.select(cfg, path)

    if isinstance(cfg, dict):
        cur: Any = cfg
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur

    cur = cfg
    for key in path.split("."):
        if not hasattr(cur, key):
            return None
        cur = getattr(cur, key)

    return cur


def _normalize_template_ws(text: str) -> str:
    # 複数行 YAML template を安全に詰める
    text = "".join(line.strip() for line in text.splitlines() if line.strip())

    # "} _a{" のような YAML 折り返し由来の空白を詰める
    text = re.sub(r"}\s+(_)", r"}\1", text)

    # その他の空白も除去
    text = re.sub(r"\s+", "", text)

    return text
