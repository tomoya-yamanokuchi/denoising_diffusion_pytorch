# scripts/analysis/plot_tradeoff.py
"""
誤切断の頻度と占有率のトレードオフの図を描く (査読指摘 E.3)。

  横軸: 誤切断の頻度 (誤切断があったエピソードの割合 [%])
  縦軸: 占有率 (配置ごとの平均を、配置間で平均した値 [%])。縦の線は配置間の標準偏差
  点  : 手法ごと。η を変えた集計結果を複数渡すと、同じ手法の点を η の順に線で結び、η の値を書き添える
        既定で η=1.0 は除外する (--exclude-eta)。重なりを避けるため手法ごとに横に少しずらす (--jitter)
  図  : 分布 (Object) ごとに1枚の小図

入力は aggregate_by_configuration.py の per_distribution_summary.csv。η ごとに別々に集計したものを
「η=パス」の形で渡す (η を変えない手法、たとえば Random や Proposed-GT は1点になる)。

使い方:
  python scripts/analysis/plot_tradeoff.py \
    --summary 0.5=<…>/per_distribution_summary.csv [0.25=<…> ...] \
    --objects "Object A,Object B,Object C" \
    --display-names "Object A=Distribution A,Object B=Distribution B,Object C=Distribution C" \
    --out tradeoff_simple.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STYLE = {  # 手法ごとの色と印
    "Random":          dict(color="tab:gray",   marker="x"),
    "VAEAC":           dict(color="tab:purple", marker="v"),
    "PCD-DM":          dict(color="tab:brown",  marker="s"),
    "Proposed-Nocond": dict(color="tab:orange", marker="^"),
    "Proposed":        dict(color="tab:blue",   marker="o"),
    "Proposed-GT":     dict(color="black",      marker="*"),
}


def load(specs: list[str]) -> pd.DataFrame:
    frames = []
    for s in specs:
        eta, path = s.split("=", 1) if "=" in s else ("", s)
        df = pd.read_csv(path)
        df["eta"] = float(eta) if eta else float("nan")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["error_freq_pct"] = 100.0 * df["num_episodes_with_cut_error"] / df["num_episodes_total"]
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", nargs="+", required=True, help="η=per_distribution_summary.csv (複数可)")
    p.add_argument("--objects", type=str, required=True, help="小図にする Object の順 (カンマ区切り)")
    p.add_argument("--display-names", type=str, default=None, help="'Object A=Distribution A,...'")
    p.add_argument("--methods", type=str, default="Random,VAEAC,PCD-DM,Proposed-Nocond,Proposed,Proposed-GT")
    p.add_argument("--no-errorbar", action="store_true", help="占有率の標準偏差の線を描かない")
    p.add_argument("--exclude-eta", type=str, default="1.0",
                   help="図から除く η (カンマ区切り。既定 1.0)。現在のスコア p+sqrt(p(1-p)) は p について単調でなく、"
                        "η=1.0 では全サンプルが目標部品を含む面 (p=1) まで安全と判定されるため")
    p.add_argument("--jitter", type=float, default=1.5,
                   help="点の重なりを避けるため、手法ごとに横軸をずらす幅 [%%ポイント] (0 で無効)")
    p.add_argument("--width", type=float, default=6.3,
                   help="図の横幅 [インチ]。論文に載せる幅にすると縮小されず、文字が --font-size のまま載る")
    p.add_argument("--font-size", type=float, default=9.0, help="文字サイズ [pt]")
    p.add_argument("--eta-style", choices=["size", "label"], default="size",
                   help="size: η を点の大きさと濃さで表す (既定) / label: η の値を点に書き込む")
    p.add_argument("--label-eta", type=str, default="Proposed,Proposed-Nocond",
                   help="点に η の値を書き込む手法 (カンマ区切り。全手法だと文字が重なりやすい)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    df = load(args.summary)
    excluded = [float(e) for e in args.exclude_eta.split(",") if e.strip()]
    if excluded:
        n0 = len(df)
        df = df[~df["eta"].apply(lambda e: any(abs(e - x) < 1e-9 for x in excluded))]
        print(f"[info] η={excluded} を除外 ({n0 - len(df)} 行)")
    objects = [o.strip() for o in args.objects.split(",")]
    names = dict(i.split("=", 1) for i in (args.display_names or "").split(",") if "=" in i)
    methods = [m.strip() for m in args.methods.split(",")]
    label_eta = {m.strip() for m in args.label_eta.split(",") if m.strip()}

    from matplotlib.colors import to_rgb
    etas = sorted(e for e in df["eta"].dropna().unique())
    eta_rank = {e: i for i, e in enumerate(etas)}
    n_eta = max(len(etas) - 1, 1)

    def eta_size(rank: int) -> float:          # η が大きいほど大きな点
        return (10 + 45 * rank / n_eta) * (fs / 9.0) ** 2

    def shade(color, rank: int):               # η が大きいほど濃い (小さい η は白に近づける)
        c = to_rgb(color)
        w = 0.75 * (1 - rank / n_eta)
        return tuple(ci * (1 - w) + w for ci in c)

    fs = args.font_size
    plt.rcParams.update({"font.size": fs, "axes.titlesize": fs, "axes.labelsize": fs,
                         "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1, "legend.fontsize": fs - 1})
    panel = args.width / len(objects)
    fig, axes = plt.subplots(1, len(objects), figsize=(args.width, panel * 1.0 + 0.75), squeeze=False,
                             sharey=True)
    for ax, obj in zip(axes[0], objects):
        g = df[df["object"] == obj]
        # ずらすのは η の曲線を持つ手法だけ。1 点だけの参考値 (Proposed-GT, Random) は本来の位置に置く
        curves = [m for m in methods if g[g["method"] == m]["eta"].nunique() > 1]
        for mi, m in enumerate(methods):
            h = g[g["method"] == m].sort_values("eta")
            if h.empty:
                continue
            st = STYLE.get(m, dict(color="tab:green", marker="D"))
            # 横軸の値が飛び飛び (エピソード数が少ない) なので、手法ごとに少しずらして重なりを避ける
            shift = args.jitter * (curves.index(m) - (len(curves) - 1) / 2) if m in curves else 0.0
            h = h.assign(error_freq_pct=h["error_freq_pct"] + shift)
            multi = h["eta"].notna().sum() > 1
            if args.eta_style == "size" and multi:
                ax.plot(h["error_freq_pct"], h["part_occupancy_rate_mean"], linestyle="-", linewidth=1.0,
                        color=st["color"], alpha=0.6, zorder=1)
                for _, r in h.iterrows():
                    rank = eta_rank[r["eta"]]
                    ax.scatter(r["error_freq_pct"], r["part_occupancy_rate_mean"],
                               s=eta_size(rank), marker=st["marker"], color=shade(st["color"], rank),
                               edgecolors=st["color"], linewidths=0.8,
                               zorder=3 + (n_eta - rank))   # 小さい点を上に描き、同じ位置の点は同心円に見せる
                ax.plot([], [], linestyle="-", label=m, **st)   # 凡例用 (手法の色と印)
            else:
                ax.plot(h["error_freq_pct"], h["part_occupancy_rate_mean"], linestyle="-", linewidth=1.2,
                        markersize=(5.5 if m != "Proposed-GT" else 9) * fs / 9.0, label=m, **st)
            if not args.no_errorbar:
                ax.errorbar(h["error_freq_pct"], h["part_occupancy_rate_mean"],
                            yerr=h["part_occupancy_rate_std_between"], fmt="none",
                            ecolor=st["color"], alpha=0.35, capsize=2)
            if args.eta_style == "label" and multi and m in label_eta:
                for _, r in h.iterrows():
                    ax.annotate(f"η={r['eta']:g}", (r["error_freq_pct"], r["part_occupancy_rate_mean"]),
                                textcoords="offset points", xytext=(4, 4), fontsize=7, color=st["color"])
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 110)
        ax.set_xlabel("Cutting error freq. [%]")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(names.get(obj, obj), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.annotate("better", xy=(4, 100), xytext=(30, 80), fontsize=fs - 1.5, color="0.4",
                    arrowprops=dict(arrowstyle="->", color="0.5"))
    axes[0][0].set_ylabel("Part occupancy rate [%]")
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:   # どの小図にも無い手法があっても凡例に入れる
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.6, handletextpad=0.4, columnspacing=1.0)
    bottom = 0.10
    if args.eta_style == "size" and len(etas) > 1:
        eh = [plt.scatter([], [], s=eta_size(eta_rank[e]), marker="o", color=shade("0.35", eta_rank[e]),
                          edgecolors="0.35", linewidths=0.8) for e in etas]
        fig.legend(eh, [f"η = {e:g}" for e in etas], loc="lower center", ncol=len(etas), frameon=False,
                   bbox_to_anchor=(0.5, 0.065), handletextpad=0.3, columnspacing=1.5)
        bottom = 0.16
    fig.tight_layout(rect=(0, bottom, 1, 1))   # 凡例は点を隠さないよう図の下にまとめる
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"[OK] {args.out}")


if __name__ == "__main__":
    main()
