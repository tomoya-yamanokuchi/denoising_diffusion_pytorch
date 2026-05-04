from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


class NamePartFormatter:
    """
    cfg から解決された値を、実験名に使う文字列へ変換する。
    """

    def format(self, value: Any) -> str:
        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value, resolve=True)

        if isinstance(value, dict):
            return "_".join(f"{k}-{v}" for k, v in value.items())

        return str(value)
