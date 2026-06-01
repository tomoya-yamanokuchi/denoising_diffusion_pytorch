# scripts/analysis/aggregate_task_metrics_by_object.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aggregate_task_metrics import (
    PAPER_METRIC_COLUMNS,
    CONDITION_COLUMNS,
    aggregate_single_root,
    read_optional_float,
    read_optional_int,
    read_optional_str,
)
from reporting.summary_console_reporter import SummaryConsoleReporter


def humanize_case_label(case: str) -> str:
    """
    Convert config case names into paper-style labels.

    Examples:
        Object_A -> Object A
        Object_D -> Object D
        Object_1 -> Object_1
    """
    text = str(case)

    if text.startswith("Object_"):
        suffix = text.replace("Object_", "", 1)
        if len(suffix) == 1 and suffix.isalpha():
            return f"Object {suffix}"

    return text


def aggregate_from_manifest_with_manifest_columns(
    manifest_path: Path,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)

    if "rollout_root" not in manifest.columns:
        raise ValueError(
            "Manifest must contain a 'rollout_root' column. "
            f"Available columns: {list(manifest.columns)}"
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

        # Keep manifest-level metadata for downstream grouping/debugging.
        for col in manifest.columns:
            if col in df.columns:
                df[f"manifest_{col}"] = spec[col]
            else:
                df[col] = spec[col]

        all_frames.append(df)

    if not all_frames:
        raise RuntimeError(f"No rows were aggregated from manifest: {manifest_path}")

    return pd.concat(all_frames, ignore_index=True)


def metric_mean_std_sem(
    group: pd.DataFrame,
    metric: str,
) -> dict[str, float]:
    values = group[metric].to_numpy(dtype=float)

    return {
        f"{metric}_mean": float(np.mean(values)),
        f"{metric}_std": (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ),
        f"{metric}_sem": (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        ),
        f"{metric}_min": float(np.min(values)),
        f"{metric}_max": float(np.max(values)),
    }


def build_summary(
    per_episode_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []

    for keys, group in per_episode_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))

        row["num_episodes"] = int(len(group))
        row["num_cases"] = int(group["case"].nunique())

        if "case" in group_cols:
            row["case_label"] = humanize_case_label(str(row["case"]))

        if "cutting_error_volume" in group.columns:
            cut_values = group["cutting_error_volume"].to_numpy(dtype=float)
            row["num_success_no_cut_error"] = int(np.sum(cut_values == 0.0))

        for metric in PAPER_METRIC_COLUMNS:
            row.update(metric_mean_std_sem(group, metric))

        # Diagnostics, if available.
        for diagnostic in [
            "num_steps_with_shift",
            "mean_abs_sampled_shift",
            "mean_abs_applied_shift",
            "max_abs_sampled_shift",
            "max_abs_applied_shift",
        ]:
            if diagnostic in group.columns:
                values = group[diagnostic].to_numpy(dtype=float)
                row[f"{diagnostic}_mean"] = float(np.mean(values))

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    sort_cols = [
        col for col in [
            "eta",
            "delta",
            "guidance_scale",
            "sample_image_num",
            "sampling_timesteps",
            "condition",
            "case",
        ]
        if col in summary_df.columns
    ]

    if sort_cols:
        summary_df = summary_df.sort_values(sort_cols)

    return summary_df


def build_overall_summary(per_episode_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["condition"] + [
        col for col in CONDITION_COLUMNS
        if col in per_episode_df.columns
    ]
    return build_summary(per_episode_df, group_cols)


def build_object_summary(per_episode_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["condition"] + [
        col for col in CONDITION_COLUMNS
        if col in per_episode_df.columns
    ] + ["case"]

    return build_summary(per_episode_df, group_cols)


def print_paper_like_summary(
    summary_df: pd.DataFrame,
    title: str,
    decimals: int = 2,
) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    display_cols = [
        col for col in [
            "condition",
            "case_label",
            "case",
            "num_episodes",
            "num_success_no_cut_error",
            "cutting_error_volume_mean",
            "cutting_error_volume_std",
            "part_remaining_rate_mean",
            "part_remaining_rate_std",
            "part_occupancy_rate_mean",
            "part_occupancy_rate_std",
        ]
        if col in summary_df.columns
    ]

    display_df = summary_df[display_cols].copy()

    rename_map = {
        "case_label": "Object",
        "case": "case",
        "num_episodes": "N",
        "num_success_no_cut_error": "NoCut",
        "cutting_error_volume_mean": "Cut Err. mean",
        "cutting_error_volume_std": "Cut Err. std",
        "part_remaining_rate_mean": "Remain mean",
        "part_remaining_rate_std": "Remain std",
        "part_occupancy_rate_mean": "Occ mean",
        "part_occupancy_rate_std": "Occ std",
    }
    display_df = display_df.rename(columns=rename_map)

    formatters = {}
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            formatters[col] = (
                lambda x, decimals=decimals:
                "" if pd.isna(x) else f"{x:.{decimals}f}"
            )

    print(display_df.to_string(index=False, formatters=formatters))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "One rollout root containing condition_metadata.yaml and "
            "case/episode rollout_data.pickle files."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "CSV manifest. Minimal format: one column 'rollout_root'. "
            "Optional columns: eta, delta, condition, method, etc."
        ),
    )

    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--delta", type=int, default=None)
    parser.add_argument("--condition", type=str, default=None)

    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory to save aggregated CSV files.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimals for console output.",
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.root is None and args.manifest is None:
        raise ValueError("Specify either --root or --manifest.")

    if args.root is not None and args.manifest is not None:
        raise ValueError("Specify only one of --root or --manifest.")

    if args.manifest is not None:
        per_episode_df = aggregate_from_manifest_with_manifest_columns(args.manifest)
    else:
        per_episode_df = aggregate_single_root(
            root=args.root,
            eta=args.eta,
            delta=args.delta,
            condition=args.condition,
        )

    overall_summary_df = build_overall_summary(per_episode_df)
    object_summary_df = build_object_summary(per_episode_df)

    per_episode_path = args.out_dir / "per_episode_metrics.csv"
    overall_summary_path = args.out_dir / "summary_metrics.csv"
    object_summary_path = args.out_dir / "summary_metrics_by_object.csv"

    per_episode_df.to_csv(per_episode_path, index=False)
    overall_summary_df.to_csv(overall_summary_path, index=False)
    object_summary_df.to_csv(object_summary_path, index=False)

    print(f"[OK] Saved per-episode metrics      : {per_episode_path}")
    print(f"[OK] Saved overall summary metrics : {overall_summary_path}")
    print(f"[OK] Saved object summary metrics  : {object_summary_path}")

    print("\nOverall Summary:")
    SummaryConsoleReporter(decimals=args.decimals).print(overall_summary_df)

    print_paper_like_summary(
        object_summary_df,
        title="Object-wise Summary",
        decimals=args.decimals,
    )


if __name__ == "__main__":
    main()
