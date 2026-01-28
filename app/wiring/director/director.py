# app/wiring/recipes.py
from __future__ import annotations
from typing import Protocol
from omegaconf import DictConfig
from ..train_builder import Builder, BuildContext
from ..experiment import Components, TrainExperiment, EvalExperiment

class Recipe(Protocol):
    name: str
    def build(self, cfg_task: DictConfig, run_dir: str): ...

class TrainRecipe:
    name = "train"

    def build(self, cfg_task: DictConfig, run_dir: str) -> TrainExperiment:
        b = Builder(BuildContext(cfg_task, run_dir))
        b.validate()

        dataset = b.build_dataset()
        image_size = b.infer_image_size(dataset)

        model = b.build_model(image_size=image_size)
        diffusion = b.build_diffusion(model=model, image_size=image_size)

        c = Components(
            dataset=dataset,
            model=model,
            diffusion=diffusion,
            device=str(cfg_task.device),
            image_size=image_size,
            run_dir=run_dir,
        )
        trainer = b.build_trainer(diffusion=diffusion, dataset=dataset)
        return TrainExperiment(c=c, trainer=trainer)



class EvalRecipe:
    name = "eval"

    def build(self, cfg_task: DictConfig, run_dir: str) -> EvalExperiment:
        b = Builder(BuildContext(cfg_task, run_dir))
        b.validate()

        dataset = b.build_dataset()
        image_size = b.infer_image_size(dataset)

        model = b.build_model(image_size=image_size)
        diffusion = b.build_diffusion(model=model, image_size=image_size)

        c = Components(
            dataset=dataset,
            model=model,
            diffusion=diffusion,
            device=str(cfg_task.device),
            image_size=image_size,
            run_dir=run_dir,
        )
        evaluator = b.build_evaluator(diffusion=diffusion, dataset=dataset)
        return EvalExperiment(c=c, evaluator=evaluator)
