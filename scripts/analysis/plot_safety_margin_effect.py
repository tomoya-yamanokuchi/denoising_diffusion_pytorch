# scripts/analysis/plot_safety_margin_effect.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from aggregate_task_metrics import (
    STD_DDOF,
    aggregate_single_root,
    read_optional_float,
    read_optional_int,
    read_optional_str,
)


POLICY_VARIANT_ORDER = [
    "Standard (r=0)",
    "Safety margin (r=Delta)",
]

POLICY_STYLE = {
    "Standard (r=0)": {
        "marker": "o",
        "linestyle": "-",
        "color": "tab:blue",
    },
    "Safety margin (r=Delta)": {
        "marker": "s",
        "linestyle": "-",
        "color": "tab:orange",
    },
}

METRIC_SPECS = {
    "cutting_error_volume": {
        "mean": "cutting_error_volume_mean",
        "std": "cutting_error_volume_std",
        "label": "Cutting Error Volume [voxels]",
        "lower": 0.0,
        "upper": None,
    },
    "part_remaining_rate": {
        "mean": "part_remaining_rate_mean",
        "std": "part_remaining_rate_std",
        "label": "Part Remaining Rate [%]",
        "lower": 0.0,
        "upper": 100.0,
    },
    "part_occupancy_rate": {
        "mean": "part_occupancy_rate_mean",
        "std": "part_occupancy_rate_std",
        "label": "Part Occupancy Rate [%]",
        "lower": 0.0,
        "upper": 100.0,
    },
}


def get_nested(
    data: dict[str, Any] | None,
    keys: list[str],
    default: Any = None,
) -> Any:
    cur = data
    for key in keys:
        if cur is None or not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def load_condition_metadata(root: Path) -> dict[str, Any] | None:
    metadata_path = root / "condition_metadata.yaml"
    if not metadata_path.exists():
        return None

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    if metadata is None:
        return {}

    if not isinstance(metadata, dict):
        raise TypeError(f"Expected dict in {metadata_path}, got {type(metadata)}")

    return metadata


def validate_manifest_for_safety_margin(
    manifest: pd.DataFrame,
    manifest_path: Path,
) -> None:
    required_columns = {
        "rollout_root",
        "delta",
        "safety_margin_voxels",
    }

    missing = required_columns - set(manifest.columns)
    if missing:
        raise ValueError(
            "\n"
            "Invalid manifest for safety-margin visualization.\n"
            f"Manifest path: {manifest_path}\n"
            f"Missing columns: {sorted(missing)}\n"
            f"Available columns: {list(manifest.columns)}\n\n"
            "For this script, the manifest must explicitly specify the execution "
            "error and safety margin for each rollout root.\n\n"
            "Example header:\n"
            "rollout_root,eta,delta,safety_margin_voxels,condition\n\n"
            "Example rows:\n"
            "/path/to/object_A_delta_0_margin_0,0.5,0,0,Object_A_delta_0_margin_0\n"
            "/path/to/object_A_delta_1_margin_0,0.5,1,0,Object_A_delta_1_margin_0\n"
            "/path/to/object_A_delta_1_margin_1,0.5,1,1,Object_A_delta_1_margin_1\n"
        )

    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    if manifest["rollout_root"].isna().any():
        raise ValueError(
            f"Manifest contains empty rollout_root values: {manifest_path}"
        )

    if manifest["delta"].isna().any():
        raise ValueError(
            f"Manifest contains empty delta values: {manifest_path}"
        )

    if manifest["safety_margin_voxels"].isna().any():
        raise ValueError(
            f"Manifest contains empty safety_margin_voxels values: {manifest_path}"
        )

    try:
        manifest["delta"].astype(int)
    except ValueError as exc:
        raise ValueError(
            "Manifest column 'delta' must contain integer values, e.g., 0, 1, 2."
        ) from exc

    try:
        manifest["safety_margin_voxels"].astype(int)
    except ValueError as exc:
        raise ValueError(
            "Manifest column 'safety_margin_voxels' must contain integer values, "
            "e.g., 0, 1, 2."
        ) from exc


def resolve_safety_margin_voxels(
    *,
    root: Path,
    spec: pd.Series,
) -> int:
    # For this script, manifest values are treated as the primary source of truth.
    for key in ["safety_margin_voxels", "safety_margin", "margin", "r"]:
        if key in spec and not pd.isna(spec[key]):
            return int(spec[key])

    # This fallback is kept for future compatibility.
    metadata = load_condition_metadata(root)
    if metadata is not None:
        candidates = [
            metadata.get("safety_margin_voxels"),
            metadata.get("safety_margin"),
            get_nested(metadata, ["policy", "decision", "param", "safety_margin_voxels"]),
            get_nested(metadata, ["policy", "decision", "safety_margin_voxels"]),
            get_nested(metadata, ["eval", "policy", "decision", "param", "safety_margin_voxels"]),
        ]

        for value in candidates:
            if value is not None:
                return int(value)

    return 0


def build_policy_variant(delta: int, safety_margin_voxels: int) -> str:
    if safety_margin_voxels == 0:
        return "Standard (r=0)"

    if safety_margin_voxels == delta:
        return "Safety margin (r=Delta)"

    return f"Safety margin (r={safety_margin_voxels})"


def aggregate_from_manifest_with_margin(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)

    validate_manifest_for_safety_margin(
        manifest=manifest,
        manifest_path=manifest_path,
    )

    all_frames = []

    for _, spec in manifest.iterrows():
        root = Path(spec["rollout_root"])

        eta = read_optional_float(spec, "eta")
        delta = read_optional_int(spec, "delta")
        condition = read_optional_str(spec, "condition")

        df = aggregate_single_root(
            root=root,
            eta=eta,
            delta=delta,
            condition=condition,
        )

        safety_margin_voxels = resolve_safety_margin_voxels(
            root=root,
            spec=spec,
        )

        df["safety_margin_voxels"] = int(safety_margin_voxels)

        if "delta" not in df.columns or df["delta"].isna().all():
            if delta is None:
                raise ValueError(
                    "delta could not be resolved from metadata or manifest.\n"
                    f"rollout_root: {root}"
                )
            df["delta"] = int(delta)

        df["policy_variant"] = [
            build_policy_variant(
                delta=int(row_delta),
                safety_margin_voxels=safety_margin_voxels,
            )
            for row_delta in df["delta"].to_numpy()
        ]

        # For easier debugging. This does not affect plotting.
        df["condition_display"] = (
            df["condition"].astype(str)
            + "_r"
            + df["safety_margin_voxels"].astype(str)
        )

        for key in ["note", "tag", "variant_label"]:
            if key in spec and not pd.isna(spec[key]):
                df[key] = spec[key]

        all_frames.append(df)

    if len(all_frames) == 0:
        raise RuntimeError(f"No rows were aggregated from manifest: {manifest_path}")

    return pd.concat(all_frames, ignore_index=True)


def filter_to_standard_and_matched_margin(per_episode_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the main comparison:
      - Standard policy: r = 0
      - Safety-margin policy: r = Delta
    This is useful when the manifest contains a full grid such as r in {0, 1, 2}.
    """
    delta = per_episode_df["delta"].astype(int)
    margin = per_episode_df["safety_margin_voxels"].astype(int)

    keep = (margin == 0) | ((delta > 0) & (margin == delta))

    return per_episode_df[keep].copy()


def validate_recognized_policy_variants(per_episode_df: pd.DataFrame) -> None:
    matched_margin = (
        (per_episode_df["policy_variant"] == "Safety margin (r=Delta)")
        & (per_episode_df["delta"].astype(int) > 0)
    )

    if not matched_margin.any():
        raise RuntimeError(
            "No rows were recognized as 'Safety margin (r=Delta)'.\n"
            "Please check that the manifest contains rows with "
            "safety_margin_voxels equal to delta, such as delta=1,r=1 "
            "and delta=2,r=2."
        )


def summarize(
    per_episode_df: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows = []

    for keys, group in per_episode_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["num_episodes"] = int(len(group))
        row["num_cases"] = int(group["case"].nunique()) if "case" in group.columns else 0

        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=STD_DDOF))
                if len(values) > 1
                else 0.0
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def add_zero_baseline_for_margin(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate the no-error standard point as the starting point of the
    safety-margin curve when explicit r=Delta results at Delta=0 are absent.
    Since Delta=0 implies r=0, this is only for visualization.
    """
    if "policy_variant" not in summary_df.columns:
        return summary_df

    has_margin_zero = (
        (summary_df["delta"] == 0)
        & (summary_df["policy_variant"] == "Safety margin (r=Delta)")
    ).any()

    if has_margin_zero:
        return summary_df

    standard_zero = summary_df[
        (summary_df["delta"] == 0)
        & (summary_df["policy_variant"] == "Standard (r=0)")
    ].copy()

    if standard_zero.empty:
        return summary_df

    standard_zero["policy_variant"] = "Safety margin (r=Delta)"
    standard_zero["safety_margin_voxels"] = 0

    return pd.concat([summary_df, standard_zero], ignore_index=True)


def get_padded_xlim(values: list[float] | np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return -0.5, 0.5

    x_min = float(values.min())
    x_max = float(values.max())

    if np.isclose(x_min, x_max):
        return x_min - 0.5, x_max + 0.5

    margin = max((x_max - x_min) * 0.08, 0.25)
    return x_min - margin, x_max + margin


def build_bounded_yerr(
    y: np.ndarray,
    yerr: np.ndarray | None,
    lower: float | None,
    upper: float | None,
) -> np.ndarray | None:
    if yerr is None:
        return None

    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    lower_err = yerr.copy()
    upper_err = yerr.copy()

    if lower is not None:
        lower_err = np.minimum(lower_err, np.maximum(y - lower, 0.0))

    if upper is not None:
        upper_err = np.minimum(upper_err, np.maximum(upper - y, 0.0))

    return np.vstack([lower_err, upper_err])


def set_metric_ylim(ax, summary_df: pd.DataFrame, metric_name: str) -> None:
    spec = METRIC_SPECS[metric_name]

    # Rate metrics are shown on a common percentage scale.
    # We keep tick labels in [0, 100], but add small visual padding
    # so markers/error bars at 0% or 100% are not cramped.
    if spec["lower"] == 0.0 and spec["upper"] == 100.0:
        padding = 5.0
        ax.set_ylim(-padding, 100.0 + padding)
        ax.set_yticks(np.arange(0, 101, 20))
        return

    mean_col = spec["mean"]
    std_col = spec["std"]

    y = summary_df[mean_col].to_numpy(dtype=float)
    err = (
        summary_df[std_col].to_numpy(dtype=float)
        if std_col in summary_df.columns
        else np.zeros_like(y)
    )

    lower_bound = spec["lower"]
    upper_bound = spec["upper"]

    lower_values = y - err
    upper_values = y + err

    if lower_bound is not None:
        lower_values = np.maximum(lower_values, lower_bound)

    if upper_bound is not None:
        upper_values = np.minimum(upper_values, upper_bound)

    y_min = float(np.nanmin(lower_values))
    y_max = float(np.nanmax(upper_values))

    if lower_bound == 0.0:
        if np.isclose(y_min, y_max):
            axis_lower = 0.0
            axis_upper = 1.0 if y_max <= 1.0 else y_max * 1.1
        else:
            margin = max((y_max - y_min) * 0.08, 1.0)
            axis_lower = 0.0
            axis_upper = y_max + margin
    else:
        margin = max((y_max - y_min) * 0.08, 1.0)
        axis_lower = y_min - margin
        axis_upper = y_max + margin

    ax.set_ylim(axis_lower, axis_upper)


def plot_metric_lines(
    ax,
    summary_df: pd.DataFrame,
    metric_name: str,
    *,
    title: str | None = None,
    show_ylabel: bool = True,
    axis_label_fontsize: int = 13,
    tick_fontsize: int = 11,
) -> None:
    spec = METRIC_SPECS[metric_name]
    mean_col = spec["mean"]
    std_col = spec["std"]

    variants = [
        variant
        for variant in POLICY_VARIANT_ORDER
        if variant in set(summary_df["policy_variant"])
    ]

    extra_variants = [
        variant
        for variant in sorted(summary_df["policy_variant"].dropna().unique())
        if variant not in variants
    ]
    variants.extend(extra_variants)

    for variant in variants:
        group = summary_df[summary_df["policy_variant"] == variant].sort_values("delta")
        if group.empty:
            continue

        style = POLICY_STYLE.get(
            variant,
            {
                "marker": "o",
                "linestyle": "-",
                "color": None,
            },
        )

        x = group["delta"].to_numpy(dtype=float)
        y = group[mean_col].to_numpy(dtype=float)
        raw_yerr = (
            group[std_col].to_numpy(dtype=float)
            if std_col in group.columns
            else None
        )
        yerr = build_bounded_yerr(
            y=y,
            yerr=raw_yerr,
            lower=spec["lower"],
            upper=spec["upper"],
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            ecolor=style["color"],
            capsize=4,
            label=variant,
        )

    ax.set_xlabel(
        "Maximum execution error $\\Delta$ [voxels]",
        fontsize=axis_label_fontsize,
    )

    if show_ylabel:
        ax.set_ylabel(spec["label"], fontsize=axis_label_fontsize)

    if title is not None:
        ax.set_title(title, fontsize=axis_label_fontsize)

    x_tick_values = sorted(summary_df["delta"].dropna().unique())
    ax.set_xticks(x_tick_values)
    ax.set_xlim(*get_padded_xlim(x_tick_values))

    set_metric_ylim(ax, summary_df, metric_name)

    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)


def plot_main_figure(
    overall_summary: pd.DataFrame,
    out_path: Path,
    *,
    axis_label_fontsize: int,
    tick_fontsize: int,
    legend_fontsize: int,
) -> None:
    plot_df = add_zero_baseline_for_margin(overall_summary)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))

    plot_metric_lines(
        axes[0],
        plot_df,
        "part_remaining_rate",
        title="Target part preservation",
        axis_label_fontsize=axis_label_fontsize,
        tick_fontsize=tick_fontsize,
    )
    plot_metric_lines(
        axes[1],
        plot_df,
        "part_occupancy_rate",
        title="Part occupancy",
        axis_label_fontsize=axis_label_fontsize,
        tick_fontsize=tick_fontsize,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=legend_fontsize,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_objectwise_figure(
    object_summary: pd.DataFrame,
    out_path: Path,
    *,
    axis_label_fontsize: int,
    tick_fontsize: int,
    legend_fontsize: int,
) -> None:
    plot_df = add_zero_baseline_for_margin(object_summary)
    cases = sorted(plot_df["case"].dropna().unique())

    if len(cases) == 0:
        raise ValueError("No case column was found for object-wise plotting.")

    metrics = [
        "cutting_error_volume",
        "part_remaining_rate",
        "part_occupancy_rate",
    ]

    fig, axes = plt.subplots(
        len(metrics),
        len(cases),
        figsize=(3.6 * len(cases), 8.0),
        squeeze=False,
        sharex=False,
    )

    for col_idx, case in enumerate(cases):
        case_df = plot_df[plot_df["case"] == case]

        for row_idx, metric in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            plot_metric_lines(
                ax,
                case_df,
                metric,
                title=str(case) if row_idx == 0 else None,
                show_ylabel=(col_idx == 0),
                axis_label_fontsize=axis_label_fontsize,
                tick_fontsize=tick_fontsize,
            )

            if ax.get_legend() is not None:
                ax.get_legend().remove()

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=legend_fontsize,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def bounded_error_for_scatter(
    value: float,
    std: float,
    lower: float = 0.0,
    upper: float = 100.0,
) -> tuple[float, float]:
    lower_err = min(std, max(value - lower, 0.0))
    upper_err = min(std, max(upper - value, 0.0))
    return lower_err, upper_err


def plot_tradeoff_figure(
    overall_summary: pd.DataFrame,
    out_path: Path,
    *,
    axis_label_fontsize: int,
    tick_fontsize: int,
    legend_fontsize: int,
) -> None:
    plot_df = add_zero_baseline_for_margin(overall_summary)
    deltas = sorted([d for d in plot_df["delta"].dropna().unique() if int(d) > 0])

    if len(deltas) == 0:
        raise ValueError("No Delta > 0 rows were found for trade-off plotting.")

    fig, axes = plt.subplots(
        1,
        len(deltas),
        figsize=(4.2 * len(deltas), 3.8),
        squeeze=False,
    )
    axes = axes[0]

    for ax, delta in zip(axes, deltas):
        delta_df = plot_df[plot_df["delta"] == delta]

        points = {}

        for variant in POLICY_VARIANT_ORDER:
            group = delta_df[delta_df["policy_variant"] == variant]
            if group.empty:
                continue

            row = group.iloc[0]
            style = POLICY_STYLE.get(variant, {"marker": "o", "color": None})

            x = float(row["part_occupancy_rate_mean"])
            y = float(row["part_remaining_rate_mean"])
            x_std = float(row["part_occupancy_rate_std"])
            y_std = float(row["part_remaining_rate_std"])

            xerr = np.asarray(bounded_error_for_scatter(x, x_std)).reshape(2, 1)
            yerr = np.asarray(bounded_error_for_scatter(y, y_std)).reshape(2, 1)

            ax.errorbar(
                [x],
                [y],
                xerr=xerr,
                yerr=yerr,
                marker=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                linestyle="None",
                capsize=4,
                label=variant,
            )

            points[variant] = (x, y)

        standard = points.get("Standard (r=0)")
        margin = points.get("Safety margin (r=Delta)")
        if standard is not None and margin is not None:
            ax.annotate(
                "",
                xy=margin,
                xytext=standard,
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

        ax.set_title(f"$\\Delta={int(delta)}$", fontsize=axis_label_fontsize)
        ax.set_xlabel("Part Occupancy Rate [%]", fontsize=axis_label_fontsize)
        ax.set_ylabel("Part Remaining Rate [%]", fontsize=axis_label_fontsize)
        ax.set_xlim(0.0, 100.5)
        ax.set_ylim(0.0, 100.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=legend_fontsize,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def print_recognized_conditions(per_episode_df: pd.DataFrame) -> None:
    print("\n[INFO] Conditions recognized for safety-margin visualization:")

    debug_cols = [
        "condition",
        "condition_display",
        "case",
        "eta",
        "delta",
        "safety_margin_voxels",
        "policy_variant",
    ]
    available_debug_cols = [
        col for col in debug_cols
        if col in per_episode_df.columns
    ]

    print(
        per_episode_df[available_debug_cols]
        .drop_duplicates()
        .sort_values(["case", "delta", "safety_margin_voxels"])
        .to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"])

    parser.add_argument(
        "--eta_filter",
        type=float,
        default=None,
        help="Optional eta value to keep, e.g., 0.5.",
    )
    parser.add_argument(
        "--case_filter",
        type=str,
        default=None,
        help="Optional comma-separated case names to keep, e.g., Object_A,Object_B,Object_C.",
    )

    parser.add_argument(
        "--include_all_margins",
        action="store_true",
        help=(
            "If specified, plot all safety_margin_voxels values in the manifest. "
            "By default, only r=0 and r=Delta are kept."
        ),
    )

    parser.add_argument("--plot_objectwise", action="store_true")
    parser.add_argument("--plot_tradeoff", action="store_true")

    parser.add_argument("--axis_label_fontsize", type=int, default=13)
    parser.add_argument("--tick_fontsize", type=int, default=11)
    parser.add_argument("--legend_fontsize", type=int, default=10)

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_episode_df = aggregate_from_manifest_with_margin(args.manifest)

    if args.eta_filter is not None:
        per_episode_df = per_episode_df[
            np.isclose(per_episode_df["eta"].astype(float), float(args.eta_filter))
        ].copy()

    if args.case_filter is not None:
        keep_cases = [x.strip() for x in args.case_filter.split(",") if x.strip()]
        per_episode_df = per_episode_df[per_episode_df["case"].isin(keep_cases)].copy()

    if not args.include_all_margins:
        per_episode_df = filter_to_standard_and_matched_margin(per_episode_df)

    if per_episode_df.empty:
        raise RuntimeError("No rows remained after applying filters.")

    print_recognized_conditions(per_episode_df)
    validate_recognized_policy_variants(per_episode_df)

    metrics = [
        "cutting_error_volume",
        "part_remaining_rate",
        "part_occupancy_rate",
    ]

    overall_summary = summarize(
        per_episode_df=per_episode_df,
        group_cols=["delta", "policy_variant", "safety_margin_voxels"],
        metrics=metrics,
    )

    object_summary = summarize(
        per_episode_df=per_episode_df,
        group_cols=["case", "delta", "policy_variant", "safety_margin_voxels"],
        metrics=metrics,
    )

    per_episode_path = args.out_dir / "per_episode_metrics_with_safety_margin.csv"
    overall_summary_path = args.out_dir / "summary_safety_margin_overall.csv"
    object_summary_path = args.out_dir / "summary_safety_margin_objectwise.csv"

    per_episode_df.to_csv(per_episode_path, index=False)
    overall_summary.to_csv(overall_summary_path, index=False)
    object_summary.to_csv(object_summary_path, index=False)

    main_fig_path = args.out_dir / f"safety_margin_effect_main.{args.format}"
    plot_main_figure(
        overall_summary,
        main_fig_path,
        axis_label_fontsize=args.axis_label_fontsize,
        tick_fontsize=args.tick_fontsize,
        legend_fontsize=args.legend_fontsize,
    )

    print(f"\n[OK] Saved per-episode metrics: {per_episode_path}")
    print(f"[OK] Saved overall summary    : {overall_summary_path}")
    print(f"[OK] Saved object-wise summary: {object_summary_path}")
    print(f"[OK] Saved main figure        : {main_fig_path}")

    if args.plot_objectwise:
        objectwise_fig_path = args.out_dir / f"safety_margin_effect_objectwise.{args.format}"
        plot_objectwise_figure(
            object_summary,
            objectwise_fig_path,
            axis_label_fontsize=args.axis_label_fontsize,
            tick_fontsize=args.tick_fontsize,
            legend_fontsize=args.legend_fontsize,
        )
        print(f"[OK] Saved object-wise figure : {objectwise_fig_path}")

    if args.plot_tradeoff:
        tradeoff_fig_path = args.out_dir / f"safety_margin_tradeoff.{args.format}"
        plot_tradeoff_figure(
            overall_summary,
            tradeoff_fig_path,
            axis_label_fontsize=args.axis_label_fontsize,
            tick_fontsize=args.tick_fontsize,
            legend_fontsize=args.legend_fontsize,
        )
        print(f"[OK] Saved trade-off figure   : {tradeoff_fig_path}")


if __name__ == "__main__":
    main()


'''
python scripts/analysis/plot_safety_margin_effect.py \
  --manifest ./analysis/revise/safety_margin/manifest.csv \
  --out_dir ./analysis/revise/safety_margin/figures_pdf \
  --eta_filter 0.5 \
  --format pdf
'''

'''
python scripts/analysis/plot_safety_margin_effect.py \
  --manifest ./analysis/revise/safety_margin/manifest.csv \
  --out_dir ./analysis/revise/safety_margin/figures_pdf \
  --eta_filter 0.5 \
  --format pdf \
  --plot_objectwise \
  --plot_tradeoff

'''
