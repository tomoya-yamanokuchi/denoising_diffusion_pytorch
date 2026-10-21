"""Generate an ensemble with zero observed cutting surfaces.

Run separately for each trained model, using its existing Hydra eval config:

    python scripts/eval_zero_observation_ensemble.py usecase=eval/complex/i2sb \
        usecase.eval.task.num_episodes=1

    python scripts/eval_zero_observation_ensemble.py usecase=eval/complex/conditional_diffusion \
        usecase.eval.task.num_episodes=1

The Hydra override for conditional_diffusion must match a configuration present
in your repository. Output goes to <eval run dir>/zero_observation/<case>/.

The script deliberately intercepts the first policy creation, calls the actual
SliceImageInferencer.predict(), saves its ensemble and pixel-wise mean, then stops BEFORE any cut.
"""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image
import torch

from app.wiring.builder.eval_builder import EvalBuilder
from app.usecases.eval.eval_usecase import EvalUsecase
from app.wiring.factories.policy_factory import PolicyFactory
from denoising_diffusion_pytorch.policy.types import PlanningPolicyInput


class _FinishedProbe(Exception):
    pass


def _save_ensemble(images: np.ndarray, out: Path, model_name: str, x1: torch.Tensor | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError(f"Expected uint8 [N,H,W,3], got {images.shape} {images.dtype}")
    n, h, w, _ = images.shape
    for idx, image in enumerate(images):
        Image.fromarray(image).save(out / f"ensemble_{idx:03d}.png")
    # Match EnsembleImageBuilder.build_from_generated_samples():
    # take the pixel-wise RGB mean across all generated samples.
    # Keep float precision until the final PNG conversion.
    mean_image = images.astype(np.float32).mean(axis=0)
    mean_uint8 = np.rint(mean_image).clip(0, 255).astype(np.uint8)
    Image.fromarray(mean_uint8).save(out / "ensemble_mean.png")
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(out / "x_cond_black.png")
    if x1 is not None:
        vis = ((x1.detach().cpu().float() + 1) / 2).clamp(0, 1)
        vis = (vis.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(vis).save(out / "x1.png")
    with (out / "metadata.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "num_samples", "height", "width", "zero_observation", "has_x1"])
        writer.writerow([model_name, n, h, w, True, x1 is not None])
    print(f"[ZeroObs] Saved {n} samples to {out}")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    original_create = PolicyFactory.create
    saved = {"done": False}
    # Run directory is created by the existing EvalBuilder.
    builder = EvalBuilder(cfg)
    context = builder.build_all()
    root = Path(builder.artifact_static_root) / "zero_observation"

    def create_and_probe(self, *args, **kwargs):
        policy = original_create(self, *args, **kwargs)
        inferencer = policy.slice_image_inferencer
        model_name = str(self._assets.policy_config.inference.model)
        x1 = getattr(policy, "normalized_x1", None)
        if model_name == "i2sb":
            if x1 is None:
                raise RuntimeError("I2SB X1 missing. Check PolicyFactory.create(case_spec=...).")
            model_size = int(x1.shape[-1])
        elif model_name == "conditional_diffusion":
            model_size = int(inferencer.inferencer.ema_model.image_size)
        else:
            raise ValueError(f"This probe supports i2sb and conditional_diffusion, not {model_name}")

        # normalized black RGB: 0 in [0,255] corresponds to -1 in [-1,1].
        black = torch.full((3, model_size, model_size), -1.0, dtype=torch.float32)
        inp = PlanningPolicyInput(normalized_cond=black, normalized_x1=x1)
        samples = inferencer.predict(inp)
        case_spec = kwargs.get("case_spec", None)
        case_name = str(getattr(case_spec, "name", "case_unknown"))
        _save_ensemble(samples, root / case_name, model_name, x1)
        saved["done"] = True
        raise _FinishedProbe()

    PolicyFactory.create = create_and_probe
    try:
        EvalUsecase().run(context)
    except _FinishedProbe:
        pass
    finally:
        PolicyFactory.create = original_create
    if not saved["done"]:
        raise RuntimeError("No policy was created; no zero-observation ensemble saved.")


if __name__ == "__main__":
    main()
