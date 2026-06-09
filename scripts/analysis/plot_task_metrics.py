# scripts/analysis/plot_task_metrics.py
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# METRIC_SPECS = [
#     {
#         "name"     : "cutting_error_volume",
#         "label"    : "Cutting Error Volume [voxels]",
#         "direction": "lower is better",
#     },
#     {
#         "name"     : "part_remaining_rate",
#         "label"    : "Part Remaining Rate [%]",
#         "direction": "higher is better",
#     },
#     {
#         "name"     : "part_occupancy_rate",
#         "label"    : "Part Occupancy Rate [%]",
#         "direction": "higher is better",
#     },
# ]


METRIC_SPECS = [
    {
        "name"     : "cutting_error_volume",
        "label"    : "Cut Err. Vol. [voxels]",
        "direction": "lower is better",
    },
    {
        "name"     : "part_remaining_rate",
        "label"    : "Part Remain. Rate [%]",
        "direction": "higher is better",
    },
    {
        "name"     : "part_occupancy_rate",
        "label"    : "Part Occ. Rate [%]",
        "direction": "higher is better",
    },
]

AXIS_LABELS = {
    "delta"             : "Maximum execution error Δ [voxels]",
    "eta"               : "Cutting-risk threshold (η)",
    "guidance_scale"    : "CFG guidance scale (w)",
    "sample_image_num"  : "Number of generated samples (M)",
    "sampling_timesteps": "DDIM sampling steps (S)",
}

METRIC_BOUNDS = {
    "cutting_error_volume": {
        "lower": 0.0,
        "upper": None,
    },
    "part_remaining_rate": {
        "lower": 0.0,
        "upper": 100.0,
    },
    "part_occupancy_rate": {
        "lower": 0.0,
        "upper": 100.0,
    },
}


def get_metric_bounds(metric_name: str) -> tuple[float | None, float | None]:
    bounds = METRIC_BOUNDS.get(metric_name, {})
    return bounds.get("lower"), bounds.get("upper")


def is_percent_metric(metric_name: str) -> bool:
    return metric_name in {
        "part_remaining_rate",
        "part_occupancy_rate",
    }

PERCENT_METRICS = {
    "part_remaining_rate",
    "part_occupancy_rate",
}


def is_percent_metric(metric_name: str) -> bool:
    return metric_name in PERCENT_METRICS


AXIS_STYLE_SPECS = {
    "delta": {
        "color": "tab:red",
        "marker": "o",
    },
    "eta": {
        "color": "tab:purple",
        "marker": "o",
    },
    "guidance_scale": {
        "color": "tab:blue",
        "marker": "o",
    },
    "sample_image_num": {
        "color": "tab:orange",
        "marker": "o",
    },
    "sampling_timesteps": {
        "color": "tab:green",
        "marker": "o",
    },
}


def get_axis_style(x_axis: str) -> dict[str, str]:
    return AXIS_STYLE_SPECS.get(
        x_axis,
        {
            "color": "tab:blue",
            "marker": "o",
        },
    )


def plot_metric(
    summary_df: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    title_suffix: str,
    out_path: Path,
    x_axis: str,
    group_by: str | None,
) -> None:
    mean_col = f"{metric_name}_mean"
    std_col = f"{metric_name}_std"

    if mean_col not in summary_df.columns:
        raise KeyError(f"Missing column: {mean_col}")

    if x_axis not in summary_df.columns:
        raise KeyError(f"Missing x_axis column: {x_axis}")

    if group_by is not None and group_by not in summary_df.columns:
        raise KeyError(f"Missing group_by column: {group_by}")

    # fig, ax = plt.subplots(figsize=(5.0, 4.0))
    fig, ax = plt.subplots(figsize=(3.7, 2.7))

    axis_style = get_axis_style(x_axis)

    if group_by is None:
        plot_groups = [(None, summary_df)]
    else:
        plot_groups = list(summary_df.groupby(group_by, dropna=False))

    print(f"\nPlotting metric '{metric_name}' vs '{x_axis}'")
    for group_value, group in plot_groups:

        group = group.sort_values(x_axis)

        x = group[x_axis].to_numpy()
        y = group[mean_col].to_numpy()

        # yerr = group[std_col].to_numpy() if std_col in group.columns else None
        yerr = build_yerr_for_plot(
            group=group,
            metric_name=metric_name,
            error_col=std_col,
            y=y,
        )

        label = None
        if group_by is not None:
            label = f"{AXIS_LABELS.get(group_by, group_by)} = {group_value:g}"

        errorbar_kwargs = {
            "yerr": yerr,
            "marker": axis_style["marker"],
            "capsize": 4,
            "label": label,
        }

        # For one-parameter sensitivity plots, use x-axis-specific color.
        # For grouped plots, keep Matplotlib's color cycle so each group remains distinguishable.
        if group_by is None:
            errorbar_kwargs["color"] = axis_style["color"]
            errorbar_kwargs["ecolor"] = axis_style["color"]

        ax.errorbar(
            x,
            y,
            **errorbar_kwargs,
        )

        set_reasonable_ylim(
            ax=ax,
            summary_df=summary_df,
            metric_name=metric_name,
            mean_col=mean_col,
            error_col=std_col,
        )


        print(f"group_value = {group_value} | x = {x} | y = {y} | yerr = {yerr}")
        # import ipdb; ipdb.set_trace()

    # ax.set_xscale("log")
    # ax.set_xlabel(AXIS_LABELS.get(x_axis, x_axis), fontsize=14)
    # ax.set_ylabel(ylabel, fontsize=14)

    ax.set_xlabel(AXIS_LABELS.get(x_axis, x_axis), fontsize=12.5)
    ax.set_ylabel(ylabel, fontsize=12.5)

    x_tick_values = sorted(summary_df[x_axis].dropna().unique())
    ax.set_xticks(x_tick_values)

    if x_axis in ["delta", "eta", "guidance_scale", "sample_image_num", "sampling_timesteps"]:
        ax.set_xlim(*get_padded_xlim(x_tick_values))

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ax.grid(True, alpha=0.3)

    if group_by is not None:
        ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    # fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def get_padded_xlim(
    x_tick_values: list[float],
    margin_ratio: float = 0.03,
    # min_margin: float = 0.5,
    min_margin: float = 0.1,
) -> tuple[float, float]:
    x_min = min(x_tick_values)
    x_max = max(x_tick_values)

    if x_min == x_max:
        return x_min - min_margin, x_max + min_margin

    x_range = x_max - x_min
    margin = max(x_range * margin_ratio, min_margin)

    return x_min - margin, x_max + margin


def build_yerr_for_plot(
    group: pd.DataFrame,
    metric_name: str,
    error_col: str,
    y: np.ndarray,
) -> np.ndarray | None:
    if error_col not in group.columns:
        return None

    err = group[error_col].to_numpy(dtype=float)

    if not is_percent_metric(metric_name):
        return err

    lower_err = np.minimum(err, np.maximum(y - 0.0, 0.0))
    upper_err = np.minimum(err, np.maximum(100.0 - y, 0.0))

    return np.vstack([lower_err, upper_err])


def set_reasonable_ylim(
    ax,
    summary_df: pd.DataFrame,
    metric_name: str,
    mean_col: str,
    error_col: str,
) -> None:
    y = summary_df[mean_col].to_numpy(dtype=float)

    if error_col in summary_df.columns:
        err = summary_df[error_col].to_numpy(dtype=float)
    else:
        err = np.zeros_like(y)

    if is_percent_metric(metric_name):
        lower_values = np.maximum(y - err, 0.0)
        upper_values = np.minimum(y + err, 100.0)

        y_min = float(np.nanmin(lower_values))
        y_max = float(np.nanmax(upper_values))

        y_range = max(y_max - y_min, 1.0)
        margin = max(y_range * 0.08, 1.0)

        lower = max(0.0, y_min - margin)
        upper = min(100.0, y_max + margin)

        # Keep a tiny drawing margin above 100 so markers at 100% are not clipped.
        # Tick labels are still restricted to <= 100.
        axis_upper = 100.5 if upper >= 99.5 else upper

        ax.set_ylim(lower, axis_upper)

        ticks = [
            tick for tick in ax.get_yticks()
            if lower <= tick <= 100.0
        ]

        if y_max >= 99.0 and not any(np.isclose(tick, 100.0) for tick in ticks):
            ticks.append(100.0)

        ax.set_yticks(sorted(ticks))
        return

    # Non-percentage metrics: normal padding.
    lower_values = y - err
    upper_values = y + err

    y_min = float(np.nanmin(lower_values))
    y_max = float(np.nanmax(upper_values))

    if np.isclose(y_min, y_max):
        margin = max(abs(y_min) * 0.08, 1.0)
    else:
        margin = max((y_max - y_min) * 0.08, 1.0)

    ax.set_ylim(y_min - margin, y_max + margin)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        type=Path,
        required=True,
        help="summary_metrics.csv generated by aggregate_task_metrics.py",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
    )
    parser.add_argument(
        "--x_axis",
        type=str,
        default="delta",
        help=(
            "Column to use as x-axis. "
            "Examples: delta, guidance_scale, sample_image_num, sampling_timesteps."
        ),
    )
    parser.add_argument(
        "--group_by",
        type=str,
        default=None,
        help=(
            "Optional column for grouping lines. "
            "Use eta for eta-delta sweep. Omit for one-parameter sensitivity."
        ),
    )


    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(args.summary_csv)


    required_cols = {args.x_axis}
    if args.group_by is not None:
        required_cols.add(args.group_by)

    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary_csv is missing columns: {missing}")


    for spec in METRIC_SPECS:
        metric_name = spec["name"]
        # out_path = args.out_dir / f"{metric_name}_vs_execution_error.{args.format}"
        out_path = args.out_dir / f"{metric_name}_vs_{args.x_axis}.{args.format}"

        plot_metric(
            summary_df=summary_df,
            metric_name=metric_name,
            ylabel=spec["label"],
            title_suffix=f"{spec['label']} ({spec['direction']})",
            out_path=out_path,
            x_axis=args.x_axis,
            group_by=args.group_by,
        )

        print(f"[OK] Saved figure: {out_path}")


if __name__ == "__main__":
    main()


'''
python scripts/analysis/plot_task_metrics.py \
    --summary_csv ./analysis/revise/sensitivity/guidance_scale/summary_metrics.csv \
    --out_dir ./analysis/revise/sensitivity/guidance_scale/figures_pdf \
    --x_axis guidance_scale \
    --format pdf
'''
