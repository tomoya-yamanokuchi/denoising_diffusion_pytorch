from __future__ import annotations

from typing import Iterable, List, Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf

from .watch_entry import WatchItem


DEFAULT_WATCH_ENTRY_PATHS: Sequence[str] = (
    "inferencer.watch.train.watch_base", # for train
    "watch.eval.watch_base", # for eval
)


def collect_watch_entries(
    cfg  : DictConfig,
    paths: Iterable[str] = DEFAULT_WATCH_ENTRY_PATHS,
) -> List[WatchItem]:
    """
    cfg 上に分散した watch entries を集める。

    優先順:
      1. inferencer.watch.train.watch_base
      2. inferencer.watch
      3. watch.watch_extra

    後段で dedupe_by_key=True にすることで、同じ key は後勝ちになる。
    """
    collected: List[WatchItem] = []

    for path in paths:
        watch_entry = OmegaConf.select(cfg, path)

        if not watch_entry:
            continue

        if isinstance(watch_entry, ListConfig):
            collected.extend(list(watch_entry))
            continue

        if isinstance(watch_entry, list):
            collected.extend(watch_entry)
            continue

        # 単一 dict / DictConfig も一応許容
        collected.append(watch_entry)
    return collected
