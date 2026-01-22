from __future__ import annotations
from typing import Protocol
from omegaconf import DictConfig
from ..builder import Builder, BuildContext
from ..experiment import Components, TrainExperiment

class Recipe(Protocol):
    name: str
    def build(self, cfg_task: DictConfig, run_dir: str): ...


class TrainDirector:
    name = "train"

    def build(self, cfg_method: DictConfig, run_dir: str) -> TrainExperiment:

        b = Builder(BuildContext(cfg_method, run_dir))
        b.validate()

        # dataset    = b.build_dataset()
        # image_size = b.infer_image_size(dataset)


        b.build_exp_name()

        b.build_run_dir_planner()

        b.build_method()

        c = Components(
            dataset    = b.dataset,
            model      = b.model,
            method     = b.method,
            device     = str(cfg_method.device),
            image_size = b.image_size,
            run_dir    =  run_dir,
        )
        # trainer = b.build_trainer(algorithm=algorithm, dataset=dataset)
        # return TrainExperiment(c=c, trainer=trainer)
        return TrainExperiment(c=c, trainer=b.trainer)

