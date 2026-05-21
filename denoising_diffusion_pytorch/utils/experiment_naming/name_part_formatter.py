from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


class NamePartFormatter:
    """
    cfg から解決された値を、実験名に使う文字列へ変換する。
    """

    def format(
        self,
        value: Any,
        value_format: str = "default",
    ) -> str:
        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value, resolve=True)

        if isinstance(value, dict):
            return "_".join(f"{k}-{v}" for k, v in value.items())

        if value_format in ("default", "", None):
            return str(value)

        if value_format == "safe_float":
            return self._format_safe_float(value)

        if value_format == "safe_str":
            return self._format_safe_str(value)

        raise ValueError(f"Unsupported value_format: {value_format}")

    def _format_safe_float(self, value: Any) -> str:
        numeric_value = float(value)

        text = f"{numeric_value:.6f}".rstrip("0").rstrip(".")

        if "." not in text:
            text = f"{text}.0"

        return text.replace("-", "m").replace(".", "p")

    def _format_safe_str(self, value: Any) -> str:
        return str(value).replace("-", "m").replace(".", "p")
