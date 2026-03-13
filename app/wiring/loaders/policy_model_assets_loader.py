from dataclasses import dataclass
from pathlib import Path

from .saved_run_config_loader import SavedRunConfigLoader
from ..types.policy_model_assets import PolicyModelAssets
from .checkpoint_path_resolver import CheckpointPathResolver

@dataclass
class PolicyModelAssetsLoader:
    config_loader           : SavedRunConfigLoader
    checkpoint_path_resolver: CheckpointPathResolver

    def load(
        self,
        run_dir: str ,
        epoch: str = "latest",
        device: str = "cuda:0",
        load_dataset: bool = False,
    ) -> PolicyModelAssets:

        cfg = self.config_loader.load(run_dir)

        if cfg.method.name != "conditional_image_diffusion":
            raise NotImplementedError(
                f"Unsupported method: {cfg.method.name}"
            )

        dataset = self._build_dataset(cfg) if load_dataset else None
        model   = self._build_model(cfg, device)
        method  = self._build_method(cfg, model, device)
        trainer = self._build_trainer(cfg, method, dataset)

        # checkpoint 復元
        trainer.load(epoch)

        return PolicyModelAssets(
            dataset=dataset,
            model=model,
            method=method,
            trainer=trainer,
            epoch=epoch,
        )

    def _build_dataset(self, cfg):
        from denoising_diffusion_pytorch.data_loader.cond_image_data_loader import Cond_image_dataloader
        return Cond_image_dataloader(
            cfg=cfg,
            image_size=cfg.dataset.image_size,
        )

    def _build_model(self, cfg, device: str):
        from denoising_diffusion_pytorch.models.unet_2d_simple_devel2 import Unet
        model = Unet(
            dim=cfg.method.model.dim,
            dim_mults=cfg.method.model.dim_mults,
            mask_dim=cfg.dataset.image_size,
            flash_attn=cfg.method.model.flash_attn,
            self_condition=cfg.method.model.self_condition,
        )
        return model.to(device)

    def _build_method(self, cfg, model, device: str):
        from denoising_diffusion_pytorch.models.conditional_image_diffusion_cfg_devel2 import GaussianDiffusion
        method = GaussianDiffusion(
            model=model,
            image_size=cfg.dataset.image_size,
            **cfg.method.diffusion,
        )
        return method.to(device)

    def _build_trainer(self, cfg, method, dataset):
        from denoising_diffusion_pytorch.trainer.diffusion_conditional_image_trainer import Trainer
        return Trainer(
            diffusion_model=method,
            dataset=dataset,
            **cfg.method.trainer,
        )
