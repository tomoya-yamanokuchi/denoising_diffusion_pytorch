from dataclasses import dataclass

from .saved_run_config_loader import SavedRunConfigLoader
from .checkpoint_path_resolver import CheckpointPathResolver
from ..types.trained_model_assets import TrainedModelAssets


@dataclass
class ConditionalDiffusionAssetsLoader:
    config_loader           : SavedRunConfigLoader
    checkpoint_path_resolver: CheckpointPathResolver

    def load(
        self,
        run_dir    : str,
        epoch      : str = "latest",
        device     : str = "cuda:0",
        infer_model: str = None,
    ) -> TrainedModelAssets:

        self.run_dir = run_dir
        cfg = self.config_loader.load(run_dir) # == cfg_usecase (train config)

        if cfg.inferencer.name != "conditional_diffusion":
            raise NotImplementedError(
                f"Unsupported method: {cfg.inferencer.name}"
            )

        dataset    = self._build_dataset(cfg)
        inferencer = self._build_inferencer(cfg, device)
        trainer    = self._build_trainer(cfg, inferencer, dataset)


        # checkpoint 復元
        trainer.load(epoch)
        inferencer = trainer.ema # diffusionの場合にはEMAがラップされているので


        self._validate_loader_contract(
            infer_model = infer_model,
            inferencer  = inferencer,
            trainer     = trainer,
        )

        return TrainedModelAssets(
            infer_model= infer_model,
            inferencer = inferencer,
            dataset    = dataset,
            epoch      = epoch,
            cfg_train  = cfg,
        )

    def _build_dataset(self, cfg):
        from denoising_diffusion_pytorch.data_loader.cond_image_data_loader import Cond_image_dataloader

        return Cond_image_dataloader(
            cfg=cfg,
            image_size=cfg.dataset.image_size,
        )


    def __build_unet(self, cfg, device: str):
        from denoising_diffusion_pytorch.models.unet_2d_simple_devel2 import Unet

        network = Unet(
            dim            = cfg.inferencer.network.dim,
            dim_mults      = cfg.inferencer.network.dim_mults,
            flash_attn     = cfg.inferencer.network.flash_attn,
            self_condition = cfg.inferencer.network.self_condition,
            mask_dim       = cfg.dataset.image_size,
        )
        return network.to(device)

    def __build_dit(self, cfg, device: str):
        from denoising_diffusion_pytorch.models.experimental.dit import DiT

        network = DiT(
            dim        = cfg.inferencer.network.dim,
            depth      = cfg.inferencer.network.depth,
            heads      = cfg.inferencer.network.heads,
            dim_head   = cfg.inferencer.network.dim_head,
            patch_size = cfg.inferencer.network.patch_size,
        )
        return network.to(device)

    def __build_network(self, cfg, device: str):
        network_name = str(cfg.inferencer.network.name).lower()

        if "unet" in network_name:
            return self.__build_unet(cfg, device)
        if "dit" in network_name:
            return self.__build_dit(cfg, device)

        raise ValueError(
            f"Unknown conditional diffusion architecture: {network_name}"
        )

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
            results_folder  = self.run_dir,
            **cfg.inferencer.trainer,
        )


    def _validate_loader_contract(self, infer_model, inferencer, trainer) -> None:
        inferencer_type = type(inferencer).__name__
        print(
            "[ConditionalDiffusionAssetsLoader] "
            f"infer_model={infer_model}, "
            f"inferencer_type={inferencer_type}, "
            f"has_ema_model={hasattr(inferencer, 'ema_model')}"
        )

        if not hasattr(inferencer, "ema_model"):
            raise TypeError(
                "Conditional diffusion inferencer must expose ema_model. "
                f"Got: {inferencer_type}"
            )
