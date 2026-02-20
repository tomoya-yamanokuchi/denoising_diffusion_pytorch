# app/wiring/experiment.py
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class Components:
    dataset   : Any
    model     : Any
    method    : Any
    device    : str
    image_size: int
    run_dir   : str

@dataclass
class TrainExperiment:
    c      : Components
    trainer: Any

@dataclass
class EvalExperiment:
    c        : Components
    evaluator: Any
