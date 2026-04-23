# scripts/run.py
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from app.wiring.builder.eval_builder import EvalBuilder
from app.usecases.eval.eval_usecase import EvalUsecase

#  Global Compute Settings for better performance on Ampere+ GPUs (A100, H100, etc.)
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1) build (wiring)
    builder = EvalBuilder(cfg)
    build_context = builder.build_all()

    # 2) run usecase (application)
    usecase = EvalUsecase()
    usecase.run(build_context)


if __name__ == "__main__":
    main()
