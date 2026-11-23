# scripts/analysis/plot_label_flip.py
"""
観測した切断面の部品ラベルの誤り (査読指摘 E.6 / R1.4) の結果の図。

  左: 誤りの割合と占有率 / 右: 誤りの割合と誤切断の頻度
  実線: Proposed (Object ごと) / 点線: Proposed-Nocond (観測を使わないので誤りの影響を受けない基準)

入力:
  --summary : aggregate_label_flip.sh が出力する label_flip_summary.csv
  --nocond  : Proposed-Nocond の per_distribution_summary.csv (aggregate_label_flip.sh の nocond/ の下)

使い方:
  python scripts/analysis/plot_label_flip.py \
    --summary $OUT/label_flip_summary.csv --nocond $OUT/nocond/per_distribution_summary.csv \
    --out $OUT/label_flip.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--nocond", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=float, default=6.3, help="図の横幅 [インチ] (論文の本文の幅)")
    ap.add_argument("--font-size", type=float, default=9.0)
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    df["error_freq_pct"] = 100.0 * df["num_episodes_with_cut_error"] / df["num_episodes_total"]
    df["rate_pct"] = 100.0 * df["flip_rate"]
    nocond = pd.read_csv(args.nocond) if args.nocond is not None and args.nocond.exists() else None
    if nocond is not None:
        nocond["error_freq_pct"] = 100.0 * nocond["num_episodes_with_cut_error"] / nocond["num_episodes_total"]

    fs = args.font_size
    plt.rcParams.update({"font.size": fs, "axes.labelsize": fs, "xtick.labelsize": fs - 1,
                         "ytick.labelsize": fs - 1, "legend.fontsize": fs - 1})
    fig, axes = plt.subplots(1, 2, figsize=(args.width, args.width * 0.38))
    objects = sorted(df["object"].unique())
    rates = sorted(df["rate_pct"].unique())
    for i, o in enumerate(objects):
        c = COLORS[i % len(COLORS)]
        g = df[df["object"] == o].sort_values("rate_pct")
        axes[0].plot(g["rate_pct"], g["part_occupancy_rate_mean"], "o-", color=c, markersize=4, label=o)
        axes[1].plot(g["rate_pct"], g["error_freq_pct"], "o-", color=c, markersize=4, label=o)
        if nocond is not None and (nocond["object"] == o).any():
            n = nocond[nocond["object"] == o].iloc[0]
            axes[0].axhline(n["part_occupancy_rate_mean"], color=c, linestyle="--", linewidth=0.9)
            axes[1].axhline(n["error_freq_pct"], color=c, linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("Part occupancy rate [%]")
    axes[1].set_ylabel("Cutting error freq. [%]")
    axes[1].set_ylim(-5, 105)
    for ax in axes:
        ax.set_xlabel("Label flip rate [%]")
        ax.set_xticks(rates)
        ax.set_xticklabels([f"{r:g}" for r in rates])
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    handles = [plt.Line2D([], [], color=h.get_color(), marker="o", markersize=4) for h in handles]
    labels = [f"Proposed ({l})" for l in labels]
    if nocond is not None:
        handles.append(plt.Line2D([], [], color="0.4", linestyle="--"))
        labels.append("Proposed-Nocond")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"[OK] {args.out.with_suffix('.pdf')} / .png")


if __name__ == "__main__":
    main()
