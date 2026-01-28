# app/wiring/components.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hydra
from omegaconf import DictConfig

@dataclass
class BuildContext:
    cfg_task: DictConfig
    run_dir: str

class EvalBuilder:
    def __init__(self, ctx: BuildContext):
        self.ctx = ctx
        self.cfg = ctx.cfg_task


    def build_run_dir_planner(self):
        '''
            cfg から「run_dir を決める」だけのオブジェクト
        '''
        from denoising_diffusion_pytorch.utils.RunDirPlanner import RunDirPlanner
        self.planner = RunDirPlanner.from_cfg(self.cfg)
        self.run_dir, self.exp_name = self.planner.plan(self.cfg)

        # import ipdb; ipdb.set_trace()

    def validate(self) -> None:
        # for k in ["dataset", "model", "diffusion", "device"]:
        for k in ["dataset", "model", "trainer", "device"]:
            if k not in self.cfg:
                raise KeyError(f"Missing key in task cfg: {k}")

    # def build_dataset(self) -> Any:
    #     from denoising_diffusion_pytorch.data_loader.cond_image_data_loader import Cond_image_dataloader
    #     self.dataset = Cond_image_dataloader(
    #         cfg        = self.cfg,
    #         image_size = self.image_size,
    #     )

    # def infer_image_size(self, dataset: Any) -> int:
    #     if hasattr(dataset, "image_size"):
    #         return int(getattr(dataset, "image_size"))
    #     if "image_size" in self.cfg.dataset:
    #         return int(self.cfg.dataset.image_size)
    #     if "image_size" in self.cfg:
    #         return int(self.cfg.image_size)
    #     raise ValueError("Cannot infer image_size")

    # def build_model(self, image_size: int) -> Any:
        # # 例：mask_dim を inject
        # model = hydra.utils.instantiate(self.cfg.model, mask_dim=image_size)
        # return self._maybe_to_device(model)


    def build_method(self) -> Any:
        if "conditional_image_diffusion" in self.cfg.name:
            from .method.ConditionalImageDiffusionBuilder import ConditionalImageDiffusionBuilder
            b = ConditionalImageDiffusionBuilder(self)

            self.dataset = b.build_dataset()
            self.model   = b.build_model()
            self.method  = b.build_method()
            self.trainer = b.build_trainer()

        if "vaeac" in self.cfg.name:
            from .method.VAEACBuilder import VAEACBuilder
            b = VAEACBuilder(self)
            self.dataset = b.build_dataset()
            self.model   = b.build_model()
            self.method  = b.build_method()
            self.trainer = b.build_trainer()


    # def build_trainer(self, algorithm: Any, dataset: Any) -> Any:
    #     return hydra.utils.instantiate(
    #         self.cfg.trainer,
    #         algorithm=algorithm,
    #         dataset=dataset,
    #         results_folder=self.ctx.run_dir,
    #     )

    def build_evaluator(self, algorithm: Any, dataset: Any) -> Any:
        return hydra.utils.instantiate(
            self.cfg.evaluator,
            algorithm=algorithm,
            dataset=dataset,
            results_folder=self.ctx.run_dir,
        )

