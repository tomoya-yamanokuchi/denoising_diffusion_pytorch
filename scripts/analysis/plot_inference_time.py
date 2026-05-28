# scripts/analysis/plot_inference_time.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


AXIS_LABELS = {
    "sample_image_num": "Number of generated samples (M)",
    "sampling_timesteps": "DDIM sampling steps (S)",
}


def plot_time(
    df: pd.DataFrame,
    *,
    x_axis: str,
    out_path: Path,
) -> None:
    subset = df[df["sweep"] == x_axis].copy()
    subset = subset.sort_values(x_axis)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    ax.errorbar(
        subset[x_axis],
        subset["mean_time_sec"],
        yerr=subset["sem_time_sec"],
        marker="o",
        capsize=4,
    )

    ax.set_xlabel(AXIS_LABELS.get(x_axis, x_axis), fontsize=14)
    ax.set_ylabel("Diffusion inference time [s]", fontsize=14)

    ticks = sorted(subset[x_axis].unique())
    ax.set_xticks(ticks)

    if x_axis == "sample_image_num":
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(v)) for v in ticks])

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for x_axis in ["sample_image_num", "sampling_timesteps"]:
        out_path = args.out_dir / f"inference_time_vs_{x_axis}.{args.format}"
        plot_time(df, x_axis=x_axis, out_path=out_path)
        print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
