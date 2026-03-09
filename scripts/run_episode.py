# scripts/run.py
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from app.wiring.builder.eval_builder import EvalBuilder
from app.usecases.eval_usecase import EvalUsecase


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
