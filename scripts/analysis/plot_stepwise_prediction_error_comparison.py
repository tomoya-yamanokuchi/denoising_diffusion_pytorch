# scripts/analysis/plot_stepwise_prediction_error_comparison.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


DEFAULT_METRICS = [
    {
        "name": "rgb_mae",
        "mean_col": "rgb_mae_mean",
        "std_col": "rgb_mae_std",
        "ylabel": "RGB MAE",
        "ylim": None,
    },
    {
        "name": "target_iou",
        "mean_col": "target_iou_mean",
        "std_col": "target_iou_std",
        "ylabel": "Target IoU",
        "ylim": (-0.03, 1.03),
    },
    {
        "name": "target_observed_fraction",
        "mean_col": "target_observed_fraction_mean",
        "std_col": "target_observed_fraction_std",
        "ylabel": "Target observed fraction",
        "ylim": (-0.03, 1.03),
    },
    {
        "name": "sample_rgb_mae",
        "mean_col": "sample_rgb_mae_mean_mean",
        "std_col": "sample_rgb_mae_mean_std",
        "ylabel": "Sample RGB MAE",
        "ylim": None,
    },
    {
        "name": "sample_target_iou",
        "mean_col": "sample_target_iou_mean_mean",
        "std_col": "sample_target_iou_mean_std",
        "ylabel": "Sample target IoU",
        "ylim": (-0.03, 1.03),
    },
]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def safe_name(text: Any) -> str:
    text = str(text)
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^0-9a-zA-Z_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def clean_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"unknown", "nan", "none", ""}:
        return ""
    return text


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, path: Path, *, save_pdf: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    if save_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)


def read_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)

    candidate_cols = ["step_summary_csv", "summary_csv", "csv_path", "path"]
    found = [col for col in candidate_cols if col in manifest.columns]

    if not found:
        raise ValueError(
            "Manifest must contain one of these columns: "
            f"{candidate_cols}. Available columns: {list(manifest.columns)}"
        )

    manifest = manifest.copy()
    manifest["step_summary_csv"] = manifest[found[0]]

    if "enabled" in manifest.columns:
        manifest = manifest[manifest["enabled"].map(parse_bool)]

    if manifest.empty:
        raise RuntimeError(f"No enabled rows in manifest: {manifest_path}")

    return manifest


def load_step_summary_from_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = read_manifest(manifest_path)

    frames: list[pd.DataFrame] = []

    for _, spec in manifest.iterrows():
        csv_path = Path(str(spec["step_summary_csv"]))
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)

        for col in manifest.columns:
            if col == "step_summary_csv":
                continue

            value = spec[col]
            if col in df.columns:
                df[f"manifest_{col}"] = value
            else:
                df[col] = value

        if "method" not in df.columns:
            df["method"] = spec.get("method", "unknown")

        if "run_id" not in df.columns:
            df["run_id"] = spec.get("run_id", csv_path.parent.name)

        if "condition" not in df.columns:
            df["condition"] = spec.get("condition", "unknown")

        frames.append(df)

    if not frames:
        raise RuntimeError(f"No data loaded from manifest: {manifest_path}")

    out = pd.concat(frames, ignore_index=True)
    return out


def choose_metrics(df: pd.DataFrame, requested: list[str] | None) -> list[dict[str, Any]]:
    metrics = []

    for metric in DEFAULT_METRICS:
        if requested is not None and metric["name"] not in requested:
            continue

        if metric["mean_col"] not in df.columns:
            continue

        metrics.append(metric)

    if not metrics:
        raise ValueError(
            "No plottable metrics found. Available columns: "
            f"{list(df.columns)}"
        )

    return metrics


def method_label(method: Any, run_id: Any, *, use_run_id: bool) -> str:
    method_text = clean_label(method)
    run_text = clean_label(run_id)

    if use_run_id and run_text:
        if method_text and method_text != run_text:
            return f"{method_text} ({run_text})"
        return run_text

    if method_text:
        return method_text

    if run_text:
        return run_text

    return "unknown"


def plot_object_metric_comparison(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    metric: dict[str, Any],
    no_std_band: bool,
    use_run_id_in_legend: bool,
) -> None:
    plot_dir = out_dir / "plots" / "by_object_metric" / metric["name"]
    plot_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["condition", "case", "case_label"]
    optional_fixed_cols = ["axis", "pred_source", "eval_region"]

    for col in optional_fixed_cols:
        if col in df.columns and df[col].nunique(dropna=False) == 1:
            group_cols.append(col)

    for keys, group in df.groupby(group_cols, dropna=False):
        key = dict(zip(group_cols, keys))

        fig, ax = plt.subplots(figsize=(6.4, 3.6))

        plotted = False

        for (method, run_id), method_group in group.groupby(["method", "run_id"], dropna=False):
            method_group = method_group.sort_values("step_idx")

            x = method_group["step_idx"].to_numpy(dtype=float)
            y = method_group[metric["mean_col"]].to_numpy(dtype=float)

            if np.all(np.isnan(y)):
                continue

            if metric["std_col"] in method_group.columns:
                y_std = method_group[metric["std_col"]].to_numpy(dtype=float)
            else:
                y_std = np.zeros_like(y)

            label = method_label(
                method,
                run_id,
                use_run_id=use_run_id_in_legend,
            )

            line = ax.plot(
                x,
                y,
                linewidth=2.1,
                marker="o",
                markersize=5.2,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=label,
            )[0]

            if not no_std_band and not np.all(np.isnan(y_std)):
                color = line.get_color()
                ax.fill_between(
                    x,
                    y - y_std,
                    y + y_std,
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )

            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        condition_label = clean_label(key.get("condition", ""))
        case_label = str(key.get("case_label", key.get("case", "unknown")))

        title = f"{metric['ylabel']}: {case_label}"
        if condition_label:
            title += f" / {condition_label}"

        ax.set_title(title, pad=10)
        ax.set_xlabel("Planning step")
        ax.set_ylabel(metric["ylabel"])

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, axis="y")
        ax.grid(True, axis="x", alpha=0.25)

        if metric["ylim"] is not None:
            ax.set_ylim(metric["ylim"])

        ax.legend(
            frameon=False,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
        )

        despine(ax)
        fig.tight_layout()

        filename = (
            f"{safe_name(key.get('condition', 'condition'))}__"
            f"{safe_name(key.get('case', case_label))}__"
            f"{safe_name(metric['name'])}__comparison.png"
        )
        save_figure(fig, plot_dir / filename)
        plt.close(fig)


def plot_object_panel_comparison(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    metrics: list[dict[str, Any]],
    no_std_band: bool,
    use_run_id_in_legend: bool,
) -> None:
    """
    Save one multi-panel figure per object.

    Rows are metrics, lines are methods.
    """
    plot_dir = out_dir / "plots" / "by_object_panel"
    plot_dir.mkdir(parents=True, exist_ok=True)

    panel_metrics = [
        m for m in metrics
        if m["name"] in {"rgb_mae", "target_iou", "target_observed_fraction"}
    ]

    if not panel_metrics:
        return

    group_cols = ["condition", "case", "case_label"]

    for keys, group in df.groupby(group_cols, dropna=False):
        key = dict(zip(group_cols, keys))

        fig, axes = plt.subplots(
            len(panel_metrics),
            1,
            figsize=(6.4, 2.55 * len(panel_metrics)),
            sharex=True,
        )

        if len(panel_metrics) == 1:
            axes = [axes]

        legend_handles = []
        legend_labels = []

        for ax, metric in zip(axes, panel_metrics):
            for (method, run_id), method_group in group.groupby(["method", "run_id"], dropna=False):
                method_group = method_group.sort_values("step_idx")

                x = method_group["step_idx"].to_numpy(dtype=float)
                y = method_group[metric["mean_col"]].to_numpy(dtype=float)

                if np.all(np.isnan(y)):
                    continue

                if metric["std_col"] in method_group.columns:
                    y_std = method_group[metric["std_col"]].to_numpy(dtype=float)
                else:
                    y_std = np.zeros_like(y)

                label = method_label(
                    method,
                    run_id,
                    use_run_id=use_run_id_in_legend,
                )

                line = ax.plot(
                    x,
                    y,
                    linewidth=2.0,
                    marker="o",
                    markersize=4.8,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    label=label,
                )[0]

                if label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)

                if not no_std_band and not np.all(np.isnan(y_std)):
                    color = line.get_color()
                    ax.fill_between(
                        x,
                        y - y_std,
                        y + y_std,
                        color=color,
                        alpha=0.12,
                        linewidth=0,
                    )

            ax.set_ylabel(metric["ylabel"])
            if metric["ylim"] is not None:
                ax.set_ylim(metric["ylim"])

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, axis="y")
            ax.grid(True, axis="x", alpha=0.25)
            despine(ax)

        axes[-1].set_xlabel("Planning step")

        condition_label = clean_label(key.get("condition", ""))
        case_label = str(key.get("case_label", key.get("case", "unknown")))

        title = f"Step-wise prediction behavior: {case_label}"
        if condition_label:
            title += f" / {condition_label}"

        fig.suptitle(title, y=1.01, fontsize=12)

        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            ncol=max(1, len(legend_labels)),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
        )

        fig.tight_layout(rect=(0, 0.04, 1, 1.0))

        filename = (
            f"{safe_name(key.get('condition', 'condition'))}__"
            f"{safe_name(key.get('case', case_label))}__panel_comparison.png"
        )
        save_figure(fig, plot_dir / filename)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help=(
            "CSV manifest with columns: method, condition, step_summary_csv. "
            "Optional: run_id, enabled."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory to save comparison CSV and plots.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "Metric names to plot. Default: all available. "
            "Examples: rgb_mae target_iou target_observed_fraction"
        ),
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Optional condition filter, e.g., eta0p5.",
    )
    parser.add_argument(
        "--no_std_band",
        action="store_true",
        help="Disable shaded std bands.",
    )
    parser.add_argument(
        "--use_run_id_in_legend",
        action="store_true",
        help="Show run_id in legend labels.",
    )
    parser.add_argument(
        "--no_panel",
        action="store_true",
        help="Do not save multi-panel object figures.",
    )

    args = parser.parse_args()

    apply_publication_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_step_summary_from_manifest(args.manifest)

    if args.condition is not None:
        df = df[df["condition"].astype(str) == args.condition]

    if df.empty:
        raise RuntimeError("No rows remain after filtering.")

    metrics = choose_metrics(df, args.metrics)

    comparison_csv = args.out_dir / "stepwise_prediction_error_comparison_input.csv"
    df.to_csv(comparison_csv, index=False)
    print(f"[OK] Saved merged comparison input: {comparison_csv}")

    for metric in metrics:
        plot_object_metric_comparison(
            df,
            args.out_dir,
            metric=metric,
            no_std_band=args.no_std_band,
            use_run_id_in_legend=args.use_run_id_in_legend,
        )

    if not args.no_panel:
        plot_object_panel_comparison(
            df,
            args.out_dir,
            metrics=metrics,
            no_std_band=args.no_std_band,
            use_run_id_in_legend=args.use_run_id_in_legend,
        )

    print(f"[OK] Saved comparison plots under: {args.out_dir / 'plots'}")


if __name__ == "__main__":
    main()
