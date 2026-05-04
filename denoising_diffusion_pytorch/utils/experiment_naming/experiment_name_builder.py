from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List

from .name_part_formatter import (
    NamePartFormatter,
)
from .name_sanitizer import (
    ExperimentNameSanitizer,
)
from .watch_entry import WatchEntry
from denoising_diffusion_pytorch.utils.experiment_naming.watch_value_resolver import (
    WatchValueResolver,
)


@dataclass(frozen=True)
class ExperimentNameBuilder:
    """
    WatchEntry の列から exp_name を組み立てる Domain Service。
    """

    resolver : WatchValueResolver      = field(default_factory=WatchValueResolver)
    formatter: NamePartFormatter       = field(default_factory=NamePartFormatter)
    sanitizer: ExperimentNameSanitizer = field(default_factory=ExperimentNameSanitizer)

    def build(self, watch: Iterable[WatchEntry], cfg: Any) -> str:
        parts: List[str] = []

        for entry in watch:
            value = self.resolver.resolve(entry, cfg)

            if value is None:
                continue

            value_str = self.formatter.format(value)
            part = f"{entry.label}{value_str}"

            if not part:
                continue

            if entry.as_dir:
                part = f"{part.rstrip('/')}/"

            parts.append(part)

        exp_name = "_".join(parts)

        return self.sanitizer.sanitize(exp_name)
