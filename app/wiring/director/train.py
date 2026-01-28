from __future__ import annotations
from typing import Protocol
from omegaconf import DictConfig
from ..builder.context import BuildContext
from ..builder.train_builder import TrainBuilder
from ..experiment import Components, TrainExperiment

class Recipe(Protocol):
    name: str
    def build(self, cfg_task: DictConfig, run_dir: str): ...


class TrainDirector:
    name = "train"

    def build(self, cfg_method: DictConfig, run_dir: str) -> TrainExperiment:
        b = TrainBuilder(cfg_method)
        # ---
        b.build_config_validator()
        b.validate_config()
        b.set_important_params()
        # ---
        b.build_run_dir_manager()
        b.plan_and_initialize_run_dir()
        # ---
        b.build_method()

        c = Components(
            dataset    = b.dataset,
            model      = b.model,
            method     = b.method,
            device     = str(cfg_method.device),
            image_size = b.image_size,
            run_dir    = run_dir,
        )
        return TrainExperiment(c=c, trainer=b.trainer)

