# app/wiring/loaders/vaeac_assets_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import torch

from .saved_run_config_loader import SavedRunConfigLoader
from ..types.trained_model_assets import TrainedModelAssets


@dataclass
class VaeacAssetsLoader:
    config_loader: SavedRunConfigLoader

    def load(
        self,
        run_dir: str,
        epoch: str = "latest",
        device: str = "cuda:0",
        infer_model: str = None,
    ) -> TrainedModelAssets:
        self.run_dir = Path(run_dir)
        cfg = self.config_loader.load(run_dir)

        if infer_model != "vaeac":
            raise ValueError(
                f"VaeacAssetsLoader only supports infer_model='vaeac', got: {infer_model}"
            )

        if str(cfg.inferencer.name) != "vaeac":
            raise ValueError(
                f"Saved train config is not VAEAC: cfg.inferencer.name={cfg.inferencer.name}"
            )

        dataset = self._build_dataset(cfg)
        inferencer = self._build_inferencer(cfg, device)

        loaded_epoch = self._restore_checkpoint(
            run_dir=self.run_dir,
            epoch=epoch,
            model=inferencer,
            device=device,
        )

        inferencer.eval()
        self._validate_loader_contract(inferencer)

        return TrainedModelAssets(
            infer_model=infer_model,
            inferencer=inferencer,
            dataset=dataset,
            epoch=loaded_epoch,
            cfg_train=cfg,
        )

    def _build_dataset(self, cfg):
        from denoising_diffusion_pytorch.data_loader.cond_image_data_loader import (
            Cond_image_dataloader,
        )

        return Cond_image_dataloader(
            cfg=cfg,
            image_size=cfg.dataset.image_size,
        )

    def _build_inferencer(self, cfg, device: str):
        from denoising_diffusion_pytorch.models.vaeac.vaeac import EncoderDecoder

        model = EncoderDecoder(cfg=cfg.inferencer)
        return model.to(device)

    def _restore_checkpoint(
        self,
        run_dir: Path,
        epoch,
        model,
        device: str,
    ) -> int:
        if epoch == "latest":
            epoch = self._resolve_latest_epoch(run_dir)

        epoch = int(epoch)
        ckpt_path = run_dir / f"model_checkpoint_{epoch}.pt"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"VAEAC checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location=device)

        if "model" not in state:
            raise KeyError(
                f"VAEAC checkpoint must contain key 'model'. "
                f"Available keys: {list(state.keys())}"
            )

        model.load_state_dict(state["model"])

        loaded_epoch = int(state.get("ckpt", epoch))

        print(f"[VaeacAssetsLoader] Loading model epoch: {loaded_epoch}")
        print(f"[VaeacAssetsLoader] checkpoint: {ckpt_path}")

        return loaded_epoch

    def _resolve_latest_epoch(self, run_dir: Path) -> int:
        candidates = sorted(run_dir.glob("model_checkpoint_*.pt"))

        if not candidates:
            raise FileNotFoundError(
                f"No VAEAC checkpoints found under: {run_dir}"
            )

        def parse_epoch(path: Path) -> int:
            match = re.search(r"model_checkpoint_(\d+)\.pt$", path.name)
            if match is None:
                return -1
            return int(match.group(1))

        latest = max(candidates, key=parse_epoch)
        latest_epoch = parse_epoch(latest)

        if latest_epoch < 0:
            raise FileNotFoundError(
                f"No valid VAEAC checkpoint filenames found under: {run_dir}"
            )

        return latest_epoch

    def _validate_loader_contract(self, inferencer) -> None:
        inferencer_type = type(inferencer).__name__
        print(
            "[VaeacAssetsLoader] "
            f"inferencer_type={inferencer_type}, "
            f"has_ema_model={hasattr(inferencer, 'ema_model')}"
        )

        if hasattr(inferencer, "ema_model"):
            raise TypeError(
                "VAEAC inferencer should be a raw model, not an EMA wrapper. "
                f"Got: {inferencer_type}"
            )

        if not hasattr(inferencer, "eval"):
            raise TypeError(
                "VAEAC inferencer must support eval(). "
                f"Got: {inferencer_type}"
            )
