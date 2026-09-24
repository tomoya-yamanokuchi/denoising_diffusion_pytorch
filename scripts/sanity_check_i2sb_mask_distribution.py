"""
Sanity check for comparing pattern-mask distributions between:
- Existing Conditional Diffusion loader
- I2SB conditional image loader

Mask convention:
    mask == 0 : observed / known
    mask == 1 : hidden / unknown
"""

from __future__ import annotations

import csv
import gc
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from denoising_diffusion_pytorch.data_loader.cond_image_data_loader import (
    Cond_image_dataloader,
)
from denoising_diffusion_pytorch.data_loader.i2sb_cond_image_data_loader import (
    I2SBCondImageDataset,
)

NUM_MASKS = 1000
NUM_EXAMPLES = 16
NUM_HIST_BINS = 30
BASE_SEED = 12345


def sample_existing(cfg, image_size):
    np.random.seed(BASE_SEED)
    loader = Cond_image_dataloader(cfg=cfg, image_size=image_size)

    hidden = []
    observed = []
    examples = []

    for i in range(NUM_MASKS):
        mask = loader._get_pattern_sample().astype(np.float32)
        h = float(mask.mean())
        o = 1.0 - h
        hidden.append(h)
        observed.append(o)
        if i < NUM_EXAMPLES:
            examples.append(mask.copy())

    del loader
    gc.collect()

    return np.asarray(hidden), np.asarray(observed), examples


def sample_i2sb(cfg, image_size):
    np.random.seed(BASE_SEED)
    loader = I2SBCondImageDataset(cfg=cfg, image_size=image_size)

    hidden = []
    observed = []
    examples = []

    for i in range(NUM_MASKS):
        mask = loader._get_pattern_sample().astype(np.float32)
        h = float(mask.mean())
        o = 1.0 - h
        hidden.append(h)
        observed.append(o)
        if i < NUM_EXAMPLES:
            examples.append(mask.copy())

    del loader
    gc.collect()

    return np.asarray(hidden), np.asarray(observed), examples


def summarize(name, hidden, observed):
    return {
        "name": name,
        "num_masks": len(hidden),
        "hidden_mean": float(hidden.mean()),
        "hidden_std": float(hidden.std()),
        "hidden_min": float(hidden.min()),
        "hidden_max": float(hidden.max()),
        "observed_mean": float(observed.mean()),
        "observed_std": float(observed.std()),
        "observed_min": float(observed.min()),
        "observed_max": float(observed.max()),
        "observed_q05": float(np.quantile(observed, 0.05)),
        "observed_q25": float(np.quantile(observed, 0.25)),
        "observed_q50": float(np.quantile(observed, 0.50)),
        "observed_q75": float(np.quantile(observed, 0.75)),
        "observed_q95": float(np.quantile(observed, 0.95)),
        "p_obs_lt_0_10": float(np.mean(observed < 0.10)),
        "p_obs_lt_0_20": float(np.mean(observed < 0.20)),
        "p_obs_lt_0_30": float(np.mean(observed < 0.30)),
        "p_obs_gt_0_50": float(np.mean(observed > 0.50)),
    }


def print_summary(s):
    print(f"\n--- {s['name']} ---")
    print(f"num masks       = {s['num_masks']}")
    print(
        f"hidden ratio   = {s['hidden_mean']:.4f} +/- {s['hidden_std']:.4f} "
        f"(min={s['hidden_min']:.4f}, max={s['hidden_max']:.4f})"
    )
    print(
        f"observed ratio = {s['observed_mean']:.4f} +/- {s['observed_std']:.4f} "
        f"(min={s['observed_min']:.4f}, max={s['observed_max']:.4f})"
    )
    print(
        "observed quantiles [5,25,50,75,95]% = "
        f"[{s['observed_q05']:.3f}, {s['observed_q25']:.3f}, "
        f"{s['observed_q50']:.3f}, {s['observed_q75']:.3f}, "
        f"{s['observed_q95']:.3f}]"
    )
    print(f"P(observed < 10%) = {s['p_obs_lt_0_10']:.3f}")
    print(f"P(observed < 20%) = {s['p_obs_lt_0_20']:.3f}")
    print(f"P(observed < 30%) = {s['p_obs_lt_0_30']:.3f}")
    print(f"P(observed > 50%) = {s['p_obs_gt_0_50']:.3f}")


def save_histogram(output_dir, cond_values, i2sb_values, title, xlabel, filename):
    bins = np.linspace(0.0, 1.0, NUM_HIST_BINS + 1)

    plt.figure(figsize=(9, 6))
    plt.hist(
        cond_values,
        bins=bins,
        alpha=0.55,
        density=True,
        label="Conditional Diffusion",
    )
    plt.hist(
        i2sb_values,
        bins=bins,
        alpha=0.55,
        density=True,
        label="I2SB",
    )
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=160)
    plt.close()


def save_example_montage(output_dir, cond_examples, i2sb_examples, filename, observed_view):
    n = min(len(cond_examples), len(i2sb_examples))
    cols = 4
    rows_per_loader = int(np.ceil(n / cols))
    total_rows = rows_per_loader * 2

    fig, axes = plt.subplots(
        total_rows,
        cols,
        figsize=(3.2 * cols, 3.2 * total_rows),
    )
    axes = np.asarray(axes).reshape(total_rows, cols)

    for i in range(rows_per_loader * cols):
        r = i // cols
        c = i % cols
        ax_cond = axes[r, c]
        ax_i2sb = axes[r + rows_per_loader, c]

        if i >= n:
            ax_cond.axis("off")
            ax_i2sb.axis("off")
            continue

        cond_mask = cond_examples[i]
        i2sb_mask = i2sb_examples[i]

        cond_img = 1.0 - cond_mask if observed_view else cond_mask
        i2sb_img = 1.0 - i2sb_mask if observed_view else i2sb_mask

        cond_obs = 1.0 - float(cond_mask.mean())
        i2sb_obs = 1.0 - float(i2sb_mask.mean())

        ax_cond.imshow(cond_img, cmap="gray", vmin=0.0, vmax=1.0)
        ax_cond.set_title(f"Cond #{i} obs={cond_obs:.3f}")
        ax_cond.axis("off")

        ax_i2sb.imshow(i2sb_img, cmap="gray", vmin=0.0, vmax=1.0)
        ax_i2sb.set_title(f"I2SB #{i} obs={i2sb_obs:.3f}")
        ax_i2sb.axis("off")

    label_text = (
        "WHITE = observed (mask=0)"
        if observed_view
        else "WHITE = hidden (mask=1)"
    )
    fig.suptitle(f"Pattern mask examples: {label_text}", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    cfg = cfg.usecase

    if cfg.dataset.type != "pattern":
        raise ValueError(
            f"This sanity check expects dataset.type=pattern, got {cfg.dataset.type}"
        )

    image_size = int(cfg.dataset.image_size)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/i2sb_mask_distribution_sanity_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("Pattern-mask distribution sanity check")
    print("========================================")
    print("image_size =", image_size)
    print("num masks per loader =", NUM_MASKS)
    print("mask semantics: 0=observed, 1=hidden")
    print("output_dir =", output_dir.resolve())

    print("\nSampling Conditional Diffusion masks ...")
    cond_hidden, cond_observed, cond_examples = sample_existing(
        cfg=cfg,
        image_size=image_size,
    )

    print("Sampling I2SB masks ...")
    i2sb_hidden, i2sb_observed, i2sb_examples = sample_i2sb(
        cfg=cfg,
        image_size=image_size,
    )

    cond_summary = summarize(
        "Conditional Diffusion",
        cond_hidden,
        cond_observed,
    )
    i2sb_summary = summarize(
        "I2SB",
        i2sb_hidden,
        i2sb_observed,
    )

    print_summary(cond_summary)
    print_summary(i2sb_summary)

    mean_diff = i2sb_summary["observed_mean"] - cond_summary["observed_mean"]
    median_diff = i2sb_summary["observed_q50"] - cond_summary["observed_q50"]

    print("\n--- direct comparison ---")
    print(f"observed mean difference   = {mean_diff:.8e}")
    print(f"observed median difference = {median_diff:.8e}")

    # Save summaries
    keys = list(cond_summary.keys())
    with open(output_dir / "mask_statistics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow(cond_summary)
        writer.writerow(i2sb_summary)

    # Save per-sample ratios
    with open(output_dir / "mask_ratios.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample",
                "conditional_hidden_ratio",
                "conditional_observed_ratio",
                "i2sb_hidden_ratio",
                "i2sb_observed_ratio",
            ]
        )
        for i in range(NUM_MASKS):
            writer.writerow(
                [
                    i,
                    cond_hidden[i],
                    cond_observed[i],
                    i2sb_hidden[i],
                    i2sb_observed[i],
                ]
            )

    # Save histograms
    save_histogram(
        output_dir,
        cond_observed,
        i2sb_observed,
        "Observed-region ratio distribution",
        "Observed ratio = 1 - mask.mean()",
        "observed_ratio_histogram.png",
    )

    save_histogram(
        output_dir,
        cond_hidden,
        i2sb_hidden,
        "Hidden-region ratio distribution",
        "Hidden ratio = mask.mean()",
        "hidden_ratio_histogram.png",
    )

    # Save visual examples
    save_example_montage(
        output_dir,
        cond_examples,
        i2sb_examples,
        "mask_examples_hidden_white.png",
        observed_view=False,
    )

    save_example_montage(
        output_dir,
        cond_examples,
        i2sb_examples,
        "mask_examples_observed_white.png",
        observed_view=True,
    )

    report = f"""Pattern-mask sanity check

Mask semantics:
  mask=0 -> observed / known
  mask=1 -> hidden / unknown

Conditional Diffusion:
  observed mean   = {cond_summary['observed_mean']:.6f}
  observed median = {cond_summary['observed_q50']:.6f}
  observed min    = {cond_summary['observed_min']:.6f}
  observed max    = {cond_summary['observed_max']:.6f}

I2SB:
  observed mean   = {i2sb_summary['observed_mean']:.6f}
  observed median = {i2sb_summary['observed_q50']:.6f}
  observed min    = {i2sb_summary['observed_min']:.6f}
  observed max    = {i2sb_summary['observed_max']:.6f}

Difference (I2SB - Conditional):
  observed mean difference   = {mean_diff:.8e}
  observed median difference = {median_diff:.8e}

Important:
  The current pattern sampler accepts crops satisfying

      0.05 <= (1 - mask.mean()) <= 0.9

  Since mask=1 means hidden, (1 - mask.mean()) is the OBSERVED ratio.
"""

    (output_dir / "report.txt").write_text(report, encoding="utf-8")

    print("\nSaved results to:")
    print(output_dir.resolve())
    print("\nMask distribution sanity check PASSED.")


if __name__ == "__main__":
    main()
