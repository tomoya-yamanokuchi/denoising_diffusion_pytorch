# app/wiring/builder/context.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from omegaconf import DictConfig

@dataclass(frozen=True)
class BuildContext:
    cfg    : DictConfig
    run_dir: Union[str, Path]
