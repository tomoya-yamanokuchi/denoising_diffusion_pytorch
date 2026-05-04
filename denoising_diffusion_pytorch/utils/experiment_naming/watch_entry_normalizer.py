from __future__ import annotations

from typing import Iterable, List, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

from .watch_entry import (
    WatchEntry,
    WatchItem,
)


def normalize_watch_entry(
    spec: Optional[Iterable[WatchItem]],
) -> List[WatchEntry]:
    """
    YAML / OmegaConf / 旧tuple形式の watch を WatchEntry に正規化する。

    許容形式:
      - ["dataset.image_size", "D"]
      - { key: dataset.image_size, label: "D" }
      - { key: log.prefix, label: "", as_dir: true }
      - { kind: date, format: "%Y%m%d", as_dir: true }
      - { value: "debug", label: "" }
    """
    out: List[WatchEntry] = []

    for item in spec or []:
        if isinstance(item, WatchEntry):
            out.append(item)
            continue

        if isinstance(item, (DictConfig, ListConfig)):
            item = OmegaConf.to_container(item, resolve=True)

        # 旧形式: ["path", "D"] / ("path", "D")
        if isinstance(item, (list, tuple)) and len(item) == 2 and not isinstance(item, dict):
            path, label = item
            out.append(WatchEntry(
                kind="config",
                key=str(path),
                label="" if label is None else str(label),
                as_dir=False,
            ))
            continue

        # dict形式
        if isinstance(item, dict):
            kind = str(item.get("kind", item.get("type", "config"))).lower()
            label = item.get("label", "")
            as_dir = bool(item.get("as_dir", False))

            if kind in {"date", "today"}:
                out.append(WatchEntry(
                    kind="date",
                    key=None,
                    label="" if label is None else str(label),
                    as_dir=as_dir,
                    fmt=str(item.get("format", item.get("fmt", "%Y%m%d"))),
                    timezone=str(item.get("timezone", item.get("tz", "Asia/Tokyo"))),
                ))
                continue

            if kind == "literal" or "value" in item:
                out.append(WatchEntry(
                    kind="literal",
                    key=None,
                    label="" if label is None else str(label),
                    as_dir=as_dir,
                    value=item.get("value"),
                ))
                continue

            path = item.get("key", item.get("path", None))
            if path is None:
                raise KeyError(
                    "watch item must have 'key' unless it uses kind: date or value"
                )

            if kind in {"basename", "path_name", "dirname"}:
                out.append(WatchEntry(
                    kind=kind,
                    key=str(path),
                    label="" if label is None else str(label),
                    as_dir=as_dir,
                ))
                continue

            if kind != "config":
                raise ValueError(f"Unsupported watch item kind: {kind}")

            out.append(WatchEntry(
                kind="config",
                key=str(path),
                label="" if label is None else str(label),
                as_dir=as_dir,
            ))
            continue

        raise TypeError(f"Unsupported watch spec item: {type(item)}: {item}")

    return out


def dedupe_watch_by_key(watch: Iterable[WatchEntry]) -> List[WatchEntry]:
    """
    同じ config key は後勝ちにする。

    例:
      watch.watch_base:
        - log.prefix
        - dataset.image_size
        - log.tag

      inferencer.watch:
        - inferencer.name
        - inferencer.diffusion.timesteps
        - dataset.image_size

    のように重複した場合、後ろにある watch の指定を優先する。
    date / literal は dedupe_key を持たないので重複排除しない。
    """
    seen: set[str] = set()
    out_reversed: List[WatchEntry] = []

    for entry in reversed(list(watch)):
        dedupe_key = entry.dedupe_key

        if dedupe_key is not None:
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

        out_reversed.append(entry)

    return list(reversed(out_reversed))
