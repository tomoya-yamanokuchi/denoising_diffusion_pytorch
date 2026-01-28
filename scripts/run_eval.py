# scripts/run.py
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from app.wiring.director.train import EvalRecipe
from app.wiring.runner import run_train, run_eval

USECASE = {
    # "train": TrainDirector(),
    "eval": EvalRecipe(),
}

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    run_dir = os.getcwd()
    OmegaConf.save(cfg, os.path.join(run_dir, "original_configs_backup.yaml"))

    usecase_name = str(cfg.usecase.name)
    usecase      = USECASE[usecase_name]

    cfg_method = cfg.method

    exp = usecase.build(cfg_method, run_dir)

    if usecase_name == "train":
        run_train(exp)
    elif usecase_name == "eval":
        run_eval(exp)
    else:
        raise ValueError(f"Unknown usecase: {usecase_name}")

if __name__ == "__main__":
    main()
