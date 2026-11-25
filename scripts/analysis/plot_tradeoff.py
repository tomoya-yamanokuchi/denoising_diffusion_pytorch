"""Plot cutting-error frequency versus part-occupancy rate across eta values."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

STYLE = {
    "Random":          {"color": "tab:gray",   "marker": "x"},
    "VAEAC":           {"color": "tab:purple", "marker": "v"},
    "PCD-DM":          {"color": "tab:brown",  "marker": "s"},
    "Proposed-Nocond": {"color": "tab:orange", "marker": "^"},
    "Proposed":        {"color": "tab:blue",   "marker": "o"},
    "Proposed-GT":     {"color": "black",      "marker": "*"},
}

# ηを比較する手法に使う形。 ηの小さい順に割り当てる。
ETA_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
# ηに依存しない参照手法は、図中でも凡例でも元の印を維持する。
ETA_INDEPENDENT_METHODS = {"Random", "Proposed-GT"}


def load(specs: list[str]) -> pd.DataFrame:
    frames = []
    for spec in specs:
        eta, path = spec.split("=", 1) if "=" in spec else ("", spec)
        frame = pd.read_csv(Path(path).expanduser())
        frame["eta"] = float(eta) if eta else float("nan")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["error_freq_pct"] = (
        100.0 * df["num_episodes_with_cut_error"] / df["num_episodes_total"]
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs="+", required=True,
                        help="η=per_distribution_summary.csv をηごとに指定")
    parser.add_argument("--objects", required=True,
                        help="小図にする Object の順（カンマ区切り）")
    parser.add_argument("--display-names", default=None,
                        help="表示名の対応。例: 'Object D=Object D,...'")
    parser.add_argument(
        "--methods", default="Random,VAEAC,PCD-DM,Proposed-Nocond,Proposed,Proposed-GT"
    )
    parser.add_argument("--no-errorbar", action="store_true")
    parser.add_argument("--exclude-eta", default="1.0",
                        help="図から除外するη（カンマ区切り、既定: 1.0）")
    parser.add_argument("--jitter", type=float, default=1.5,
                        help="曲線の重なりを避ける横方向のずらし幅[%ポイント]。0で無効")
    parser.add_argument("--width", type=float, default=6.3,
                        help="図の横幅[インチ]")
    parser.add_argument("--font-size", type=float, default=9.0,
                        help="文字サイズ[pt]")
    parser.add_argument("--eta-style", choices=["shape", "size"], default="shape",
                        help="shape: ηをマーカー形状で区別（既定） / size: 大きさと濃淡")
    parser.add_argument(
        "--annotate-methods", default="",
        help="選択したObjectの手法だけηラベルを追加。例: 'Object D=Proposed;Object E=PCD-DM'",
    )
    parser.add_argument("--annotate-etas", default="0,0.25,0.5,0.75",
                        help="ラベルを付けるη（カンマ区切り、既定: 0,0.5,1.0）")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = load(args.summary)
    excluded = [float(value.strip()) for value in args.exclude_eta.split(",")
                if value.strip()]
    if excluded:
        before = len(df)
        df = df[~df["eta"].apply(
            lambda value: pd.notna(value)
            and any(abs(value - excluded_value) < 1e-9 for excluded_value in excluded)
        )]
        print(f"[info] η={excluded} を除外 ({before - len(df)} 行)")

    objects = [value.strip() for value in args.objects.split(",")]
    names = dict(item.split("=", 1) for item in (args.display_names or "").split(",")
                 if "=" in item)
    methods = [value.strip() for value in args.methods.split(",")]
    annotations = {}
    for item in args.annotate_methods.split(";"):
        if "=" in item:
            obj_name, method_name = item.split("=", 1)
            annotations.setdefault(obj_name.strip(), set()).add(method_name.strip())
    annotate_etas = [float(value.strip()) for value in args.annotate_etas.split(",")
                     if value.strip()]
    etas = sorted(value for value in df["eta"].dropna().unique())
    if len(etas) > len(ETA_MARKERS):
        raise ValueError(f"ηの数が多すぎます（最大 {len(ETA_MARKERS)}）: {etas}")
    eta_to_marker = {eta: ETA_MARKERS[index] for index, eta in enumerate(etas)}
    eta_rank = {eta: index for index, eta in enumerate(etas)}
    max_rank = max(len(etas) - 1, 1)

    fs = args.font_size
    plt.rcParams.update({
        "font.size": fs,
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs - 1,
        "ytick.labelsize": fs - 1,
        "legend.fontsize": fs - 1,
    })
    panel_width = args.width / len(objects)
    fig, axes = plt.subplots(
        1, len(objects), figsize=(args.width, panel_width + 0.75),
        squeeze=False, sharey=True,
    )

    # ηを持つ曲線同士だけを少し横にずらし、参照点は本来の位置に置く。
    curve_methods = [
        method for method in methods
        if method not in ETA_INDEPENDENT_METHODS
        and df[df["method"] == method]["eta"].nunique() > 1
    ]

    for ax, obj in zip(axes[0], objects):
        obj_df = df[df["object"] == obj]
        for method in methods:
            points = obj_df[obj_df["method"] == method].sort_values("eta")
            if points.empty:
                continue
            style = STYLE.get(method, {"color": "tab:green", "marker": "D"})
            color = style["color"]
            shift = (
                args.jitter * (curve_methods.index(method) - (len(curve_methods) - 1) / 2)
                if method in curve_methods else 0.0
            )
            points = points.assign(error_freq_pct=points["error_freq_pct"] + shift)
            is_eta_curve = (
                method not in ETA_INDEPENDENT_METHODS
                and points["eta"].nunique() > 1
            )

            if is_eta_curve and args.eta_style == "shape":
                ax.plot(points["error_freq_pct"], points["part_occupancy_rate_mean"],
                        color=color, alpha=0.65, linewidth=1.1, zorder=1)
                for _, row in points.iterrows():
                    eta = row["eta"]
                    ax.scatter(
                        row["error_freq_pct"], row["part_occupancy_rate_mean"],
                        marker=eta_to_marker[eta], s=38 * (fs / 9.0) ** 2,
                        facecolor=color, edgecolor=color, linewidth=0.8, zorder=3,
                    )
            elif is_eta_curve and args.eta_style == "size":
                ax.plot(points["error_freq_pct"], points["part_occupancy_rate_mean"],
                        color=color, alpha=0.65, linewidth=1.1, zorder=1)
                for _, row in points.iterrows():
                    rank = eta_rank[row["eta"]]
                    size = (18 + 42 * rank / max_rank) * (fs / 9.0) ** 2
                    alpha = 0.35 + 0.65 * rank / max_rank
                    ax.scatter(
                        row["error_freq_pct"], row["part_occupancy_rate_mean"],
                        marker=style["marker"], s=size, color=color, alpha=alpha,
                        edgecolor=color, linewidth=0.8, zorder=3,
                    )
            else:
                # Random / Proposed-GT はη形状を割り当てず、元の x / star を維持。
                ax.plot(
                    points["error_freq_pct"], points["part_occupancy_rate_mean"],
                    color=color, marker=style["marker"], linestyle="-",
                    linewidth=1.1, markersize=(5.5 if method != "Proposed-GT" else 9) * fs / 9.0,
                    zorder=3,
                )

            if not args.no_errorbar:
                ax.errorbar(
                    points["error_freq_pct"], points["part_occupancy_rate_mean"],
                    yerr=points["part_occupancy_rate_std_between"], fmt="none",
                    ecolor=color, alpha=0.35, capsize=2, zorder=2,
                )

            # 指定された Object × 手法に限り、主要なηを手法色で直接表示する。
            if method in annotations.get(obj, set()):
                selected = points[points["eta"].apply(
                    lambda value: pd.notna(value) and any(
                        abs(value - target) < 1e-9 for target in annotate_etas
                    )
                )]
                default_offsets = [(7, 0), (7, 0), (7, 0), (7, 0)]
                # Object Dでは近接するProposedの点を縦に分散。
                object_d_proposed_offsets = {
                    0.0: (8, 4), 0.25: (8, 2), 0.5: (8, -4), 0.75: (8, 4),
                }
                for label_index, (_, row) in enumerate(selected.sort_values("eta").iterrows()):
                    x = row["error_freq_pct"]
                    eta_value = float(row["eta"])
                    if obj == "Object D" and method == "Proposed":
                        dx, dy = next(
                            (offset for eta_key, offset in object_d_proposed_offsets.items()
                             if abs(eta_value - eta_key) < 1e-9),
                            default_offsets[label_index % len(default_offsets)],
                        )
                    else:
                        # Object EのPCD-DMなどは点の高さに沿ってラベルを並べる。
                        dx, dy = default_offsets[label_index % len(default_offsets)]
                    # :g で η=0.25 や η=0.75 を丸めずに表示する。
                    eta_label = f"η={eta_value:g}"
                    ax.annotate(
                        eta_label,
                        (x, row["part_occupancy_rate_mean"]),
                        xytext=(dx, dy), textcoords="offset points", ha="left",
                        va="center", fontsize=fs - 1, color=color,
                        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                              "edgecolor": "none", "alpha": 0.78},
                        zorder=5,
                    )

        ax.set_xlim(-5, 110)
        ax.set_ylim(-5, 110)
        ax.set_xlabel("Cutting error freq. [%]")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(names.get(obj, obj), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.annotate(
            "better", xy=(4, 100), xytext=(30, 80), fontsize=fs - 1.5,
            color="0.4", arrowprops={"arrowstyle": "->", "color": "0.5"},
        )

    axes[0][0].set_ylabel("Part occupancy rate [%]")

    # 手法凡例: η依存手法は色の線、η非依存手法は実際の x / star を示す。
    method_handles = []
    for method in methods:
        style = STYLE.get(method, {"color": "tab:green", "marker": "D"})
        if method in ETA_INDEPENDENT_METHODS:
            method_handles.append(Line2D(
                [], [], color=style["color"], marker=style["marker"],
                linewidth=1.1, markersize=7 if method == "Proposed-GT" else 5.5,
                label=method,
            ))
        else:
            method_handles.append(Line2D(
                [], [], color=style["color"], linewidth=1.8, label=method,
            ))
    fig.legend(
        handles=method_handles, loc="lower center", ncol=len(method_handles),
        frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.6,
        handletextpad=0.4, columnspacing=1.0,
    )

    bottom = 0.10
    if args.eta_style == "shape" and len(etas) > 1:
        eta_handles = [Line2D(
            [], [], color="0.35", marker=eta_to_marker[eta], linestyle="None",
            markerfacecolor="0.75", markeredgecolor="0.35", markersize=6,
            label=f"η = {eta:g}",
        ) for eta in etas]
        fig.legend(
            handles=eta_handles, loc="lower center", ncol=len(eta_handles),
            frameon=False, bbox_to_anchor=(0.5, 0.065),
            handletextpad=0.3, columnspacing=1.5,
        )
        bottom = 0.16

    fig.tight_layout(rect=(0, bottom, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"[OK] {args.out}")


if __name__ == "__main__":
    main()
