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

class Builder:
    def __init__(self, ctx: BuildContext):
        self.ctx = ctx
        self.cfg = ctx.cfg_task
        # self._set_params()

    def _set_params(self):

        self.image_size = self.cfg.dataset.image_size

        from denoising_diffusion_pytorch.utils import make_save_path
        from denoising_diffusion_pytorch.utils.LogPathManager import watch



        self.save_path = make_save_path(
            logbase = self.cfg.log.logbase,
            dataset = 'Image_diffusion_2D',
            exp_name = watch(self.cfg.watch),
        )

        import ipdb; ipdb.set_trace()


    def build_exp_name(self):
        '''
            configの情報から実験ディレクトリの名前とパスを生成するためのオブジェクト
        '''
        from denoising_diffusion_pytorch.utils.ExperimentNamer import ExperimentNamer
        self.namer    = ExperimentNamer.from_cfg(self.cfg.watch)
        self.exp_name = self.namer.make(self.cfg)  # cfg は DictConfig のままでOK

    def build_run_dir_planner(self):
        '''
            cfg から「run_dir を決める」だけのオブジェクト
        '''
        from denoising_diffusion_pytorch.utils.RunDirPlanner import RunDirPlanner
        self.planner = RunDirPlanner.from_cfg(self.cfg)
        run_dir, exp_name = self.planner.plan(self.cfg)

        import ipdb; ipdb.set_trace()

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
            # self.method  = b.build_method()
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

