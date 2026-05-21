from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from omegaconf import DictConfig, ListConfig


@dataclass(frozen=True)
class WatchEntry:
    kind    : str           = "config"
    key     : Optional[str] = None
    label   : str           = ""
    as_dir  : bool          = False
    fmt     : str           = "%Y%m%d"
    timezone: str           = "Asia/Tokyo"
    value   : Any           = None
    value_format: str       = "default"

    @property
    def dedupe_key(self) -> Optional[str]:
        """
        重複排除に使うキー。

        config 由来の watch だけ key 単位で後勝ちにする。
        date / literal は key を持たないため重複排除しない。
        """
        if self.kind == "config":
            return self.key
        return None


WatchItem = Union[
    WatchEntry,
    Tuple[str, str],
    Sequence[Any],
    Mapping[str, Any],
    DictConfig,
    ListConfig,
]
