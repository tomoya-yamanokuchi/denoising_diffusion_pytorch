# scripts/analysis/plot_resolution_scaling.py
"""
benchmark_resolution_scaling.py の結果 (CSV) から、ボクセル解像度ごとの
推論時間と GPU メモリの図を作る (査読指摘 E.7 / R3.2, Appendix E 用)。

  (a) 推論時間 [s]   vs ボクセル数 K^3   (両対数。目盛りは 16^3, 25^3, ... と表示)
  (b) GPU メモリ [GiB] vs ボクセル数 K^3 (両対数)
  - U-Net の基本幅ごとに線を引き、各点に値を書き込む
  - --gpu_limit_csv を渡すと、その GPU の総メモリを (b) に水平線で示し、
    その GPU で OOM になった設定を注記する (例: RTX 4090 の結果の CSV)

使い方:
  python scripts/analysis/plot_resolution_scaling.py \
    --csv analysis/revise/scalability/resolution_scaling_h100.csv \
    --gpu_limit_csv analysis/revise/scalability/resolution_scaling_rtx4090.csv \
    --out analysis/revise/scalability/resolution_scaling.pdf

  4090 の CSV が手元に無い場合は、上限の値を直接指定する:
    --gpu_limit_gib 23.51 --gpu_limit_label "RTX 4090 memory (24 GB)"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MiB_PER_GiB = 1024.0
WIDTH_STYLE = {64: dict(marker="o", linestyle="-"), 128: dict(marker="s", linestyle="--")}


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df[df["model"] == "untrained"].copy()
    df["voxels"] = df["K"].astype(int) ** 3
    df["memory_GiB"] = df["peak_allocated_MiB"] / MiB_PER_GiB
    return df


def annotate(ax, x, y, text, dy_points=6, dx_points=0, ha="center"):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx_points, dy_points),
                ha=ha, fontsize=9)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True, help="図にする計測結果 (例: H100)")
    p.add_argument("--gpu_limit_csv", type=Path, default=None,
                   help="メモリ上限を示したい GPU の計測結果 (例: RTX 4090)")
    p.add_argument("--gpu_limit_gib", type=float, default=None,
                   help="CSV を使わずに、メモリ上限の線を GiB で直接指定する "
                        "(RTX 4090 なら 23.51 = PyTorch が報告する総メモリ 24,078.875 MiB)")
    p.add_argument("--gpu_limit_label", type=str, default=None,
                   help="上限線のラベル (省略時は CSV の GPU 名)")
    p.add_argument("--out", type=Path, required=True, help="出力ファイル (.pdf / .png)")
    p.add_argument("--scale", choices=["linear", "log"], default="linear",
                   help="linear: 横軸 K・縦軸とも普通の目盛り (既定) / log: 横軸 K^3・縦軸とも対数")
    args = p.parse_args()

    df = load(args.csv)
    ok = df[df["status"] == "ok"]
    gpu_name = str(df["gpu"].iloc[0])
    Ks = sorted(df["K"].astype(int).unique())

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, col, ylabel, fmt in [
        (axes[0], "mean_time_sec", "Inference time [s]", "{:.2f}"),
        (axes[1], "memory_GiB", "Peak GPU memory [GiB]", "{:.1f}"),
    ]:
        for width, g in ok.groupby("unet_dim"):
            g = g.sort_values("voxels")
            style = WIDTH_STYLE.get(int(width), dict(marker="^", linestyle=":"))
            xs = g["voxels"] if args.scale == "log" else g["K"]
            ax.plot(xs, g[col], label=f"{int(width)} channels", **style)
            for x, (_, r) in zip(xs, g.iterrows()):
                if args.scale == "log":
                    annotate(ax, x, r[col], fmt.format(r[col]),
                             dy_points=6 if int(width) == 128 else -12)
                elif int(width) == 128:      # 普通の目盛り: 上の線は点の左上、下の線は点の右下に置く
                    annotate(ax, x, r[col], fmt.format(r[col]), dy_points=5, dx_points=-4, ha="right")
                else:
                    annotate(ax, x, r[col], fmt.format(r[col]), dy_points=-9, dx_points=4, ha="left")
        lo, hi = ok[col].min(), ok[col].max()
        if args.scale == "log":
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(lo / 1.8, hi * 2.2)          # 値の注記が軸やタイトルに重ならない余白
            ax.set_xlim(Ks[0] ** 3 / 1.6, Ks[-1] ** 3 * 1.6)
            ax.set_xticks([k ** 3 for k in Ks])
            ax.set_xticklabels([f"${k}^3$" for k in Ks])
            ax.minorticks_off()
            ax.set_xlabel("Number of voxels $K^3$")
        else:
            ax.set_ylim(-hi * 0.10, hi * 1.15)
            ax.set_xlim(Ks[0] - 7, Ks[-1] + 8)     # 数値の注記が枠にかからない左右の余白
            ax.set_xticks(Ks)
            ax.set_xlabel("Voxel resolution $K$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", alpha=0.3)

    if args.gpu_limit_csv is None and args.gpu_limit_gib is not None:
        label = args.gpu_limit_label or "GPU memory limit"
        axes[1].axhline(args.gpu_limit_gib, color="tab:red", linewidth=1.0, linestyle=":")
        axes[1].annotate(label, ((Ks[0] ** 3) if args.scale == "log" else Ks[0], args.gpu_limit_gib), textcoords="offset points",
                         xytext=(0, 3), fontsize=9, color="tab:red")
    if args.gpu_limit_csv is not None:
        lim = pd.read_csv(args.gpu_limit_csv)
        limit_gib = float(lim["gpu_total_MiB"].iloc[0]) / MiB_PER_GiB
        label = args.gpu_limit_label or f"{lim['gpu'].iloc[0]} memory"
        axes[1].axhline(limit_gib, color="tab:red", linewidth=1.0, linestyle=":")
        axes[1].annotate(label, ((Ks[0] ** 3) if args.scale == "log" else Ks[0], limit_gib), textcoords="offset points",
                         xytext=(0, 3), fontsize=9, color="tab:red")
        oom = lim[(lim["model"] == "untrained") & (lim["status"] == "OOM")]
        for _, r in oom.iterrows():
            print(f"[info] {lim['gpu'].iloc[0]} で OOM: K={int(r['K'])}, base width={int(r['unet_dim'])}")

    axes[0].legend(loc="upper left")
    axes[0].set_title("(a)", fontsize=11)
    axes[1].set_title("(b)", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"[OK] {args.out}  (measured on {gpu_name})")


if __name__ == "__main__":
    main()
