from dataclasses import dataclass
from pathlib import Path

from .saved_run_config_loader import SavedRunConfigLoader
from .checkpoint_path_resolver import CheckpointPathResolver
from ..types.trained_model_assets import TrainedModelAssets

@dataclass
class TrainedModelAssetsLoader:
    config_loader           : SavedRunConfigLoader
    checkpoint_path_resolver: CheckpointPathResolver

    def load(
        self,
        run_dir: str,
        epoch  : str = "latest",
        device : str = "cuda:0",
    ) -> TrainedModelAssets:

        cfg = self.config_loader.load(run_dir)

        if cfg.inferencer.name != "conditional_image_diffusion":
            raise NotImplementedError(
                f"Unsupported method: {cfg.inferencer.name}"
            )

        dataset    = self._build_dataset(cfg)
        inferencer = self._build_inferencer(cfg, device)
        trainer    = self._build_trainer(cfg, inferencer, dataset)

        # checkpoint 復元
        trainer.load(epoch)

        return TrainedModelAssets(
            inferencer = inferencer,
            trainer    = trainer,
            dataset    = dataset,
            epoch      = epoch,
        )

    def _build_dataset(self, cfg):
        from denoising_diffusion_pytorch.data_loader.cond_image_data_loader import Cond_image_dataloader
        return Cond_image_dataloader(
            cfg=cfg,
            image_size=cfg.dataset.image_size,
        )


    def __build_network(self, cfg, device: str):
        from denoising_diffusion_pytorch.models.unet_2d_simple_devel2 import Unet
        network = Unet(
            dim            = cfg.inferencer.network.dim,
            dim_mults      = cfg.inferencer.network.dim_mults,
            flash_attn     = cfg.inferencer.network.flash_attn,
            self_condition = cfg.inferencer.network.self_condition,
            mask_dim       = cfg.dataset.image_size,
        )
        return network.to(device)

    def _build_inferencer(self, cfg, device: str):
        network = self.__build_network(cfg, device)
        # ----
        from denoising_diffusion_pytorch.models.conditional_image_diffusion_cfg_devel2 import GaussianDiffusion
        method = GaussianDiffusion(
            model      = network,
            image_size = cfg.dataset.image_size,
            **cfg.inferencer.diffusion,
        )
        return method.to(device)

    def _build_trainer(self, cfg, model, dataset):
        from denoising_diffusion_pytorch.trainer.diffusion_conditional_image_trainer import Trainer
        return Trainer(
            diffusion_model = model,
            dataset         = dataset,
            **cfg.inferencer.trainer,
        )
