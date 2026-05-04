from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List

from denoising_diffusion_pytorch.utils.experiment_naming import (
    ExperimentNameBuilder,
    WatchEntry,
    WatchItem,
    WatchValueResolver,
    dedupe_watch_by_key,
    normalize_watch_entry,
)


@dataclass(frozen=True)
class ExperimentNamer:
    """
    後方互換用 Facade。

    既存コードの:
        ExperimentNamer.from_cfg(watch).make(cfg)

    という呼び出しを維持しつつ、実装は experiment_naming 配下へ委譲する。
    """

    watch: List[WatchEntry]
    builder: ExperimentNameBuilder = field(default_factory=ExperimentNameBuilder)

    @staticmethod
    def from_cfg(
        watch_spec: Iterable[WatchItem],
        *,
        dedupe_by_key: bool = False,
    ) -> "ExperimentNamer":
        watch = normalize_watch_entry(watch_spec)

        if dedupe_by_key:
            watch = dedupe_watch_by_key(watch)

        return ExperimentNamer(watch=watch)

    def make(self, cfg: Any) -> str:
        return self.builder.build(self.watch, cfg)

    @staticmethod
    def render_template(template: str, cfg: Any) -> str:
        """
        旧 ExperimentNamer.render_template(...) 互換用。
        """
        return WatchValueResolver().render_template(template=template, cfg=cfg)
