from __future__ import annotations
from typing import Protocol
from omegaconf import DictConfig
from ..builder.context import BuildContext
from ..builder.eval_builder  import EvalBuilder
from ..experiment import Components, EvalExperiment

class Recipe(Protocol):
    name: str
    def build(self, cfg_task: DictConfig, run_dir: str): ...


class EvalDirector:
    name = "eval"

    def build(self, cfg_method: DictConfig, run_dir: str) -> EvalExperiment:

        b = EvalBuilder(BuildContext(cfg_method, run_dir))
        b.validate()
        b.build_run_dir_planner()
        b.build_method()

        c = Components(
            dataset    = b.dataset,
            model      = b.model,
            method     = b.method,
            device     = str(cfg_method.device),
            image_size = b.cfg.dataset.image_size,
            run_dir    =  run_dir,
        )
        return EvalExperiment(c=c, evaluator=b.evaluator)

