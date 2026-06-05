# app/wiring/loaders/diffusion_1d_assets_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .saved_run_config_loader import SavedRunConfigLoader
from ..types.trained_model_assets import TrainedModelAssets


@dataclass
class Diffusion1DAssetsLoader:
    config_loader: SavedRunConfigLoader

    def load(
        self,
        run_dir    : str,
        epoch      : str = "latest",
        device     : str = "cuda:0",
        infer_model: str | None = None,
    ) -> TrainedModelAssets:
        self.run_dir = Path(run_dir)
        cfg = self.config_loader.load(run_dir)

        if infer_model not in {"diffusion_1d", "diffusion_1D"}:
            raise ValueError(
                "Diffusion1DAssetsLoader only supports "
                f"infer_model='diffusion_1d' or 'diffusion_1D', got: {infer_model}"
            )

        if str(cfg.inferencer.name) != "diffusion_1d":
            raise ValueError(
                f"Saved train config is not diffusion_1d: "
                f"cfg.inferencer.name={cfg.inferencer.name}"
            )

        dataset    = self._build_dataset(cfg)
        inferencer = self._build_inferencer(cfg, dataset, device)
        trainer    = self._build_trainer(cfg, inferencer, dataset)

        loaded_epoch = self._restore_checkpoint(
            trainer=trainer,
            run_dir=self.run_dir,
            epoch=epoch,
        )

        trainer.ema.ema_model.eval()
        self._validate_loader_contract(trainer)

        return TrainedModelAssets(
            infer_model=infer_model,
            inferencer=trainer,
            dataset=dataset,
            epoch=loaded_epoch,
            cfg_train=cfg,
        )

    def _build_dataset(self, cfg):
        from denoising_diffusion_pytorch.data_loader.image_data_loader import Dataset1D

        return Dataset1D(
            folder=cfg.dataset.path,
            image_size=cfg.dataset.image_size,
            grid_3dim=cfg.dataset.grid_3dim,
            is_shuffle=cfg.dataset.is_shuffle,
            augment_horizontal_flip=cfg.dataset.horizontal_flip,
            convert_image_to=cfg.dataset.convert_image_to,
        )

    def _build_inferencer(self, cfg, dataset, device: str):
        from denoising_diffusion_pytorch.models.unet_1d import Unet1D
        from denoising_diffusion_pytorch.models.pointnet_1d import PointNet1D
        from denoising_diffusion_pytorch.models.diffusion_1d import GaussianDiffusion1D

        channels, seq_len = dataset[0].shape
        network_cfg = cfg.inferencer.network

        if network_cfg.name == "unet1d":
            network = Unet1D(
                dim             = network_cfg.dim,
                dim_mults       = network_cfg.dim_mults,
                channels        = channels,
                out_dim         = channels,
                self_condition  = network_cfg.self_condition,
            ).to(device)

        elif "pointnet1d" in network_cfg.name:
            network = PointNet1D(
                dim             = network_cfg.dim,
                channels        = channels,
                self_condition  = network_cfg.self_condition,
                depth           = network_cfg.get("depth", 6),
                global_dim      = network_cfg.get("global_dim", network_cfg.dim),
            ).to(device)

        else:
            raise ValueError(f"Unknown network: {network_cfg.name}")

        method = GaussianDiffusion1D(
            model=network,
            seq_length=seq_len,
            **cfg.inferencer.diffusion,
        ).to(device)

        return method

    def _build_trainer(self, cfg, inferencer, dataset):
        from denoising_diffusion_pytorch.trainer.diffusion_1d_trainer import Trainer1D

        return Trainer1D(
            diffusion_model=inferencer,
            dataset=dataset,
            results_folder=str(self.run_dir),
            **cfg.inferencer.trainer,
        )

    def _restore_checkpoint(self, trainer, run_dir: Path, epoch) -> int:
        if epoch == "latest":
            epoch = self._resolve_latest_epoch(run_dir)

        epoch = int(epoch)
        trainer.load(epoch)

        print(f"[Diffusion1DAssetsLoader] Loading model epoch: {epoch}")
        print(f"[Diffusion1DAssetsLoader] checkpoint: {run_dir / f'model-{epoch}.pt'}")

        return epoch

    def _resolve_latest_epoch(self, run_dir: Path) -> int:
        candidates = sorted(run_dir.glob("model-*.pt"))

        if not candidates:
            raise FileNotFoundError(
                f"No Diffusion1D checkpoints found under: {run_dir}"
            )

        def parse_epoch(path: Path) -> int:
            match = re.search(r"model-(\d+)\.pt$", path.name)
            if match is None:
                return -1
            return int(match.group(1))

        latest = max(candidates, key=parse_epoch)
        latest_epoch = parse_epoch(latest)

        if latest_epoch < 0:
            raise FileNotFoundError(
                f"No valid Diffusion1D checkpoint filenames found under: {run_dir}"
            )

        return latest_epoch

    def _validate_loader_contract(self, trainer) -> None:
        trainer_type = type(trainer).__name__

        print(
            "[Diffusion1DAssetsLoader] "
            f"inferencer_type={trainer_type}, "
            f"has_ema={hasattr(trainer, 'ema')}, "
            f"has_get_1d_to_2d_images={hasattr(trainer, 'get_1d_to_2d_images')}"
        )

        if not hasattr(trainer, "ema"):
            raise TypeError(
                "Diffusion1D trainer must expose ema. "
                f"Got: {trainer_type}"
            )

        if not hasattr(trainer, "get_1d_to_2d_images"):
            raise TypeError(
                "Diffusion1D trainer must expose get_1d_to_2d_images(). "
                f"Got: {trainer_type}"
            )
