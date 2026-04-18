from __future__ import annotations

import hydra
from omegaconf import DictConfig

from app.wiring.builder.eval_builder import EvalBuilder
from app.wiring.builder.train_builder import TrainBuilder  # 既にある前提


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # usecase 名で分岐（あなたが以前導入した “usecase” の思想）
    usecase = str(cfg.usecase.name)

    if usecase == "train":
        b = TrainBuilder(cfg.method)
        b.build_all()
        # 例: b.trainer.train()
        b.trainer.train()

    elif usecase == "eval":
        b = EvalBuilder(cfg)
        b.build_all()
        b.evaluator.run()

    else:
        raise ValueError(f"Unknown usecase: {usecase}")


if __name__ == "__main__":
    main()
