# app/wiring/runner.py
from .experiment import TrainExperiment, EvalExperiment

def run_train(exp: TrainExperiment) -> None:
    exp.trainer.train()

def run_eval(exp: EvalExperiment) -> None:
    exp.evaluator.evaluate()
