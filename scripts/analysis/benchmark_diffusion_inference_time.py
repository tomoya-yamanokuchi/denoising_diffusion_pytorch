# scripts/analysis/benchmark_diffusion_inference_time.py
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from app.wiring.loaders.conditional_diffusion_assets_loader import (
    ConditionalDiffusionAssetsLoader,
)
from app.wiring.loaders.saved_run_config_loader import SavedRunConfigLoader
from app.wiring.loaders.checkpoint_path_resolver import CheckpointPathResolver
from denoising_diffusion_pytorch.policy.inference.conditional_diffusion_slice_image_inferencer import (
    ConditionalDiffusionSliceImageInferencer,
)
from denoising_diffusion_pytorch.policy.types import PlanningPolicyInput


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def sync_if_cuda(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def build_dummy_condition(
    *,
    image_size: int,
    device: str,
    observed_patch_size: int = 4,
) -> torch.Tensor:
    """
    Build a minimal dummy condition tensor with shape [C, H, W].

    - -1.0 means unobserved.
    - A small observed patch is filled with 0.0 so that conditional inference path
      is actually exercised.
    """
    cond = torch.full(
        (3, image_size, image_size),
        fill_value=-1.0,
        dtype=torch.float32,
        device=device,
    )

    patch = min(observed_patch_size, image_size)
    cond[:, :patch, :patch] = 0.0
    return cond


def apply_sampling_timesteps(model, sampling_timesteps: int) -> None:
    """
    Force evaluation-time DDIM sampling steps.

    This mirrors the intended behavior in ConditionalDiffusionSliceImageInferencer.
    Keeping this here also makes the benchmark robust even if the wrapper changes.
    """
    sampling_timesteps = int(sampling_timesteps)

    if sampling_timesteps <= 0:
        raise ValueError(f"sampling_timesteps must be positive: {sampling_timesteps}")

    if sampling_timesteps > model.num_timesteps:
        raise ValueError(
            f"sampling_timesteps={sampling_timesteps} exceeds "
            f"num_timesteps={model.num_timesteps}"
        )

    model.sampling_timesteps = sampling_timesteps
    model.is_ddim_sampling = model.sampling_timesteps < model.num_timesteps


@torch.inference_mode()
def benchmark_one_condition(
    *,
    ema_wrapper,
    sample_image_num: int,
    sampling_timesteps: int,
    guidance_scale: float,
    device: str,
    warmup: int,
    repeats: int,
) -> dict:
    model = ema_wrapper.ema_model
    apply_sampling_timesteps(model, sampling_timesteps)

    image_size = int(model.image_size)
    normalized_cond = build_dummy_condition(
        image_size=image_size,
        device=device,
    )
    planning_input = PlanningPolicyInput(normalized_cond=normalized_cond)

    slice_inferencer = ConditionalDiffusionSliceImageInferencer(
        inferencer=ema_wrapper,
        sample_image_num=sample_image_num,
        control_mode="conditional",
        guidance_scale=guidance_scale,
        sampling_timesteps=sampling_timesteps,
    )

    # Warmup
    for _ in range(warmup):
        _ = slice_inferencer.predict(planning_input)
    sync_if_cuda(device)

    elapsed = []

    for _ in range(repeats):
        sync_if_cuda(device)
        start = time.perf_counter()

        _ = slice_inferencer.predict(planning_input)

        sync_if_cuda(device)
        end = time.perf_counter()
        elapsed.append(end - start)

    elapsed = np.asarray(elapsed, dtype=float)

    return {
        "sample_image_num": int(sample_image_num),
        "sampling_timesteps": int(sampling_timesteps),
        "guidance_scale": float(guidance_scale),
        "num_repeats": int(repeats),
        "mean_time_sec": float(elapsed.mean()),
        "std_time_sec": float(elapsed.std(ddof=1)) if len(elapsed) > 1 else 0.0,
        "sem_time_sec": (
            float(elapsed.std(ddof=1) / np.sqrt(len(elapsed)))
            if len(elapsed) > 1
            else 0.0
        ),
        "all_times_sec": elapsed.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_run_dir", type=str, required=True)
    parser.add_argument("--epoch", type=str, default="100000")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_csv", type=Path, required=True)

    parser.add_argument("--guidance_scale", type=float, default=0.2)

    parser.add_argument(
        "--sample_image_nums",
        type=str,
        default="4,8,16,32,64",
        help="Comma-separated M values.",
    )
    parser.add_argument(
        "--sampling_timesteps_list",
        type=str,
        default="2,5,10,20,50",
        help="Comma-separated S values.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["M_sweep", "S_sweep", "both"],
        default="both",
    )
    parser.add_argument("--default_M", type=int, default=32)
    parser.add_argument("--default_S", type=int, default=20)

    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)

    args = parser.parse_args()

    loader = ConditionalDiffusionAssetsLoader(
        config_loader=SavedRunConfigLoader(),
        checkpoint_path_resolver=CheckpointPathResolver(),
    )

    assets = loader.load(
        run_dir=args.train_run_dir,
        epoch=args.epoch,
        device=args.device,
        infer_model="conditional_diffusion",
    )

    rows = []

    if args.mode in ["M_sweep", "both"]:
        for M in parse_int_list(args.sample_image_nums):
            row = benchmark_one_condition(
                ema_wrapper=assets.inferencer,
                sample_image_num=M,
                sampling_timesteps=args.default_S,
                guidance_scale=args.guidance_scale,
                device=args.device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            row["sweep"] = "sample_image_num"
            rows.append(row)

    if args.mode in ["S_sweep", "both"]:
        for S in parse_int_list(args.sampling_timesteps_list):
            row = benchmark_one_condition(
                ema_wrapper=assets.inferencer,
                sample_image_num=args.default_M,
                sampling_timesteps=S,
                guidance_scale=args.guidance_scale,
                device=args.device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            row["sweep"] = "sampling_timesteps"
            rows.append(row)

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"[OK] Saved benchmark results: {args.out_csv}")
    print(df[
        [
            "sweep",
            "sample_image_num",
            "sampling_timesteps",
            "mean_time_sec",
            "std_time_sec",
            "sem_time_sec",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
