from dataclasses import dataclass
from pathlib import Path

from .saved_run_config_loader import SavedRunConfigLoader
from ..types.trained_model_assets import TrainedModelAssets


@dataclass
class VaeacAssetsLoader:
    config_loader: SavedRunConfigLoader

    def load(
        self,
        run_dir    : str,
        epoch      : str = "latest",
        device     : str = "cuda:0",
        infer_model: str = None,
    ) -> TrainedModelAssets:
        cfg = self.config_loader.load(run_dir)

        if infer_model != "vaeac":
            raise ValueError(
                f"VaeacAssetsLoader only supports infer_model='vaeac', got: {infer_model}"
            )

        dataset = self._build_dataset(cfg)
        model   = self._build_model(cfg, device)
        trainer = self._build_trainer(cfg, model, dataset)

        loaded_epoch = self._restore_checkpoint(
            run_dir = run_dir,
            epoch   = epoch,
            model   = model,
            trainer = trainer,
        )

        self._validate_loader_contract(model)

        return TrainedModelAssets(
            infer_model = infer_model,
            inferencer  = model,
            trainer     = trainer,
            dataset     = dataset,
            epoch       = loaded_epoch,
        )

    def _build_dataset(self, cfg):
        # まずは saved config から dataset callable を辿る方が自然だが、
        # 現状の project 構成に合わせて適宜調整。
        if hasattr(cfg, "dataset") and callable(cfg.dataset):
            return cfg.dataset()
        if hasattr(cfg, "dataset_config") and callable(cfg.dataset_config):
            return cfg.dataset_config()

        raise NotImplementedError(
            "VAEAC dataset construction is not defined in saved config loader path."
        )

    def _build_model(self, cfg, device: str):
        if hasattr(cfg, "model") and callable(cfg.model):
            model = cfg.model()
        elif hasattr(cfg, "model_config") and callable(cfg.model_config):
            model = cfg.model_config()
        else:
            raise NotImplementedError(
                "VAEAC model construction is not defined in saved config loader path."
            )

        return model.to(device) if hasattr(model, "to") else model

    def _build_trainer(self, cfg, model, dataset):
        if hasattr(cfg, "trainer") and callable(cfg.trainer):
            return cfg.trainer(model, dataset)
        if hasattr(cfg, "trainer_config") and callable(cfg.trainer_config):
            return cfg.trainer_config(model, dataset)

        raise NotImplementedError(
            "VAEAC trainer construction is not defined in saved config loader path."
        )

    def _restore_checkpoint(self, run_dir: str, epoch, model, trainer):
        from denoising_diffusion_pytorch.utils.vaeac_utils.vaeac_utils import (
            load_ckpt,
            get_optimizers,
        )

        if epoch == "latest":
            epoch = self._resolve_latest_epoch(run_dir)

        ckpt_path = Path(run_dir) / f"model_checkpoint_{epoch}.pt"
        trainer.config.train_config["pretrained"] = str(ckpt_path)

        optim = get_optimizers(model, trainer.config.model_config)
        restored_model, _, _ = load_ckpt(model, optim, trainer.config.train_config)

        # load_ckpt が model を返すので、中身を trainer/model 両方に反映
        model.load_state_dict(restored_model.state_dict())
        trainer.step = epoch

        print(f"[VaeacAssetsLoader] Loading model epoch: {epoch}")
        print(f"[VaeacAssetsLoader] checkpoint: {ckpt_path}")

        return epoch

    def _resolve_latest_epoch(self, run_dir: str) -> int:
        from denoising_diffusion_pytorch.utils.serialization import get_latest_epoch
        return get_latest_epoch((run_dir,))

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
