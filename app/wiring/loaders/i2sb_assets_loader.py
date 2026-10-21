from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import torch
from ema_pytorch import EMA

from app.wiring.method.I2SBImageDiffusionBuilder import I2SBImageDiffusionBuilder
from app.wiring.types.trained_model_assets import TrainedModelAssets
from .saved_run_config_loader import SavedRunConfigLoader
from .checkpoint_path_resolver import CheckpointPathResolver


@dataclass
class I2SBInferenceBundle:
    ema_model: torch.nn.Module
    diffusion: object
    image_size: int
    cfg_train: object


@dataclass
class I2SBAssetsLoader:
    config_loader: SavedRunConfigLoader
    checkpoint_path_resolver: CheckpointPathResolver

    def load(self, run_dir: str, epoch: str = "latest", device: str = "cuda:0",
             infer_model: str = "i2sb") -> TrainedModelAssets:
        cfg = self.config_loader.load(run_dir)
        if str(cfg.inferencer.name).lower() != "i2sb":
            raise ValueError(f"Expected i2sb training config; got {cfg.inferencer.name}")

        # Rebuild only network and bridge; no dataset, optimizer or trainer.
        cfg.device = device
        builder = I2SBImageDiffusionBuilder(cfg=cfg, artifact_static_root=str(run_dir))
        network = builder.build_model()
        diffusion = builder.build_method()

        ckpt = self._resolve_checkpoint(Path(run_dir), epoch)
        data = torch.load(str(ckpt), map_location=device, weights_only=False)
        if "ema" not in data:
            raise KeyError(f"Checkpoint has no EMA state: {ckpt}")

        trainer_cfg = cfg.inferencer.trainer
        ema = EMA(
            network,
            beta=float(trainer_cfg.get("ema_decay", 0.995)),
            update_every=int(trainer_cfg.get("ema_update_every", 10)),
        ).to(device)
        ema.load_state_dict(data["ema"], strict=True)
        ema.ema_model.eval()
        for parameter in ema.ema_model.parameters():
            parameter.requires_grad_(False)

        print(f"[I2SBAssetsLoader] loaded {ckpt} (step={data.get('step', '?')})")
        bundle = I2SBInferenceBundle(
            ema_model=ema.ema_model,
            diffusion=diffusion,
            image_size=int(cfg.dataset.image_size),
            cfg_train=cfg,
        )
        return TrainedModelAssets(
            infer_model=infer_model,
            inferencer=bundle,
            dataset=None,
            epoch=epoch,
            cfg_train=cfg,
        )

    @staticmethod
    def _resolve_checkpoint(run_dir: Path, epoch: str) -> Path:
        if str(epoch) != "latest":
            path = run_dir / f"model-{epoch}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        matches = []
        for path in run_dir.glob("model-*.pt"):
            match = re.fullmatch(r"model-(\d+)\.pt", path.name)
            if match:
                matches.append((int(match.group(1)), path))
        if not matches:
            raise FileNotFoundError(f"No model-<step>.pt in {run_dir}")
        return max(matches, key=lambda item: item[0])[1]
