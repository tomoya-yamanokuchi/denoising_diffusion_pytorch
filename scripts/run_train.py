# scripts/run.py
from __future__ import annotations

import torch
# --- Compute optimizations (Ampere+) ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

import hydra
from omegaconf import DictConfig

from app.wiring.builder.train_builder import TrainBuilder
from app.usecases.train.train_usecase import TrainUsecase

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1) build (wiring)
    builder = TrainBuilder(cfg)
    build_context = builder.build_all()

    # 2) run usecase (application)
    usecase = TrainUsecase()
    usecase.run(build_context)


if __name__ == "__main__":
    main()
