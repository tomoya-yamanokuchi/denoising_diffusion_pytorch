# scripts/analysis/aggregate_task_metrics.py
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PAPER_METRIC_COLUMNS = [
    "cutting_error_volume",
    "part_remaining_rate",
    "part_occupancy_rate",
]


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, but got {type(data)}")

    return data


def get_array(
    data: dict[str, Any],
    primary_key: str,
    fallback_key: str | None = None,
) -> np.ndarray:
    if primary_key in data:
        return np.asarray(data[primary_key])

    if fallback_key is not None and fallback_key in data:
        return np.asarray(data[fallback_key])

    raise KeyError(
        f"Neither '{primary_key}' nor fallback '{fallback_key}' was found. "
        f"Available keys: {list(data.keys())}"
    )


def extract_case_and_episode(rollout_path: Path) -> tuple[str, int]:
    """
    Expected path pattern:
        .../<case_name>/episode_<idx>/rollout_data.pickle
    """
    episode_dir = rollout_path.parent
    case_dir = episode_dir.parent

    case_name = case_dir.name

    episode_name = episode_dir.name
    if episode_name.startswith("episode_"):
        episode_idx = int(episode_name.replace("episode_", ""))
    else:
        episode_idx = -1

    return case_name, episode_idx


def summarize_execution_error_infos(
    execution_error_infos: list[dict[str, Any]] | None,
) -> dict[str, float]:
    if not execution_error_infos:
        return {
            "num_steps_with_shift": 0,
            "mean_abs_sampled_shift": 0.0,
            "mean_abs_applied_shift": 0.0,
            "max_abs_sampled_shift": 0.0,
            "max_abs_applied_shift": 0.0,
        }

    sampled = np.asarray(
        [float(info.get("sampled_shift", 0.0)) for info in execution_error_infos]
    )
    applied = np.asarray(
        [float(info.get("applied_shift", 0.0)) for info in execution_error_infos]
    )

    return {
        "num_steps_with_shift": int(np.count_nonzero(applied)),
        "mean_abs_sampled_shift": float(np.mean(np.abs(sampled))),
        "mean_abs_applied_shift": float(np.mean(np.abs(applied))),
        "max_abs_sampled_shift": float(np.max(np.abs(sampled))),
        "max_abs_applied_shift": float(np.max(np.abs(applied))),
    }


def summarize_rollout(
    rollout_path: Path,
    eta: float | None,
    delta: int | None,
    condition: str | None,
) -> dict[str, Any]:
    data = load_pickle(rollout_path)

    cutting_error_volumes = get_array(
        data,
        primary_key="cutting_error_volumes",
        fallback_key="rewards",
    )
    part_remaining_rates = get_array(
        data,
        primary_key="part_remaining_rates",
        fallback_key=None,
    )
    part_occupancy_rates = get_array(
        data,
        primary_key="part_occupancy_rates",
        fallback_key="removal_performance",
    )

    case_name, episode_idx = extract_case_and_episode(rollout_path)

    execution_error_summary = summarize_execution_error_infos(
        data.get("execution_error_infos")
    )

    row = {
        "condition": condition,
        "eta": eta,
        "delta": delta,
        "case": case_name,
        "episode": episode_idx,
        "rollout_path": str(rollout_path),

        # Paper metrics
        "cutting_error_volume": float(np.sum(cutting_error_volumes)),
        "part_remaining_rate": float(part_remaining_rates[-1]),
        "part_occupancy_rate": float(part_occupancy_rates[-1]),

        # Optional diagnostics
        "num_steps": int(len(cutting_error_volumes)),
        "step_cutting_error_volumes": cutting_error_volumes.tolist(),
        "step_part_remaining_rates": part_remaining_rates.tolist(),
        "step_part_occupancy_rates": part_occupancy_rates.tolist(),
    }

    row.update(execution_error_summary)
    return row


def find_rollout_files(root: Path) -> list[Path]:
    rollout_files = sorted(root.rglob("rollout_data.pickle"))
    if len(rollout_files) == 0:
        raise FileNotFoundError(f"No rollout_data.pickle found under: {root}")
    return rollout_files


def aggregate_single_root(
    root: Path,
    eta: float | None,
    delta: int | None,
    condition: str | None,
) -> pd.DataFrame:
    rows = []
    for rollout_path in find_rollout_files(root):
        rows.append(
            summarize_rollout(
                rollout_path=rollout_path,
                eta=eta,
                delta=delta,
                condition=condition,
            )
        )
    return pd.DataFrame(rows)


def aggregate_from_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)

    required = {"rollout_root", "eta", "delta"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    all_rows = []

    for _, spec in manifest.iterrows():
        root = Path(spec["rollout_root"])
        eta = float(spec["eta"])
        delta = int(spec["delta"])

        if "condition" in manifest.columns and not pd.isna(spec["condition"]):
            condition = str(spec["condition"])
        else:
            condition = f"eta_{eta:g}_delta_{delta}"

        df = aggregate_single_root(
            root=root,
            eta=eta,
            delta=delta,
            condition=condition,
        )
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)


def build_summary(per_episode_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["eta", "delta"]

    if "condition" in per_episode_df.columns:
        group_cols = ["condition", "eta", "delta"]

    summary_rows = []

    for keys, group in per_episode_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["num_episodes"] = int(len(group))

        for metric in PAPER_METRIC_COLUMNS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_sem"] = (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )

        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values(["eta", "delta"])


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory that contains rollout_data.pickle files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV manifest with columns: rollout_root, eta, delta, optional condition.",
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

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest is None and args.root is None:
        raise ValueError("Specify either --root or --manifest.")

    if args.manifest is not None:
        per_episode_df = aggregate_from_manifest(args.manifest)
    else:
        per_episode_df = aggregate_single_root(
            root=args.root,
            eta=args.eta,
            delta=args.delta,
            condition=args.condition,
        )

    summary_df = build_summary(per_episode_df)

    per_episode_path = args.out_dir / "per_episode_metrics.csv"
    summary_path = args.out_dir / "summary_metrics.csv"

    per_episode_df.to_csv(per_episode_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] Saved per-episode metrics: {per_episode_path}")
    print(f"[OK] Saved summary metrics    : {summary_path}")

    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
