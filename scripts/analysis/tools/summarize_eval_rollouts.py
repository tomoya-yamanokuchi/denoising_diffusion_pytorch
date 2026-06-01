#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, but got {type(data)}")

    return data


def natural_key(text: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def as_float_array(data: dict[str, Any], keys: list[str]) -> np.ndarray:
    for key in keys:
        if key in data:
            arr = np.asarray(data[key], dtype=float).reshape(-1)
            return arr
    return np.asarray([], dtype=float)


def final_or_nan(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(arr[-1])


def sum_or_nan(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.sum(arr))


def parse_episode_index(episode_dir_name: str) -> int | None:
    m = re.search(r"episode[_-](\d+)", episode_dir_name, re.IGNORECASE)
    if m is None:
        return None
    return int(m.group(1))


def find_condition_roots(root: Path) -> list[Path]:
    """
    Accept either:
      - condition root: .../epsilon_greedy_00
      - experiment root: .../simple_B_..._proposed
      - higher root containing multiple experiments

    A condition root is recognized by having condition_metadata.yaml
    or config_resolved.yaml and case directories below it.
    """
    candidates = []

    if (root / "condition_metadata.yaml").exists() or (root / "config_resolved.yaml").exists():
        candidates.append(root)

    for p in root.rglob("condition_metadata.yaml"):
        candidates.append(p.parent)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)

    return sorted(out, key=lambda x: str(x))


def iter_rollout_paths(condition_root: Path):
    """
    Expected structure:
      condition_root/
        object_7/episode_0/rollout_data.pickle
        Object_7/episode_0/rollout_data.pickle
    """
    for rollout_path in sorted(condition_root.rglob("rollout_data.pickle")):
        episode_dir = rollout_path.parent
        case_dir = episode_dir.parent

        # Skip files not under case/episode directory.
        if not episode_dir.name.lower().startswith("episode"):
            continue

        yield case_dir, episode_dir, rollout_path


def summarize_one_rollout(condition_root: Path, case_dir: Path, episode_dir: Path, rollout_path: Path) -> dict[str, Any]:
    data = load_pickle(rollout_path)

    # New format
    step_cut = as_float_array(data, ["cutting_error_volumes", "rewards"])
    step_remain = as_float_array(data, ["part_remaining_rates"])
    step_occ = as_float_array(data, ["part_occupancy_rates", "removal_performance"])

    # Fallback for very old format:
    # old "infos" sometimes stored target_removal_rate, not part_remaining_rate.
    infos = as_float_array(data, ["infos"])
    if step_remain.size == 0 and infos.size > 0:
        # Treat infos as target_removal_rate only for legacy format.
        step_remain = 100.0 - infos

    episode_idx = parse_episode_index(episode_dir.name)

    return {
        "condition_root": str(condition_root),
        "case_dir": case_dir.name,
        "case_norm": case_dir.name.lower(),
        "episode_dir": episode_dir.name,
        "episode": episode_idx,
        "rollout_path": str(rollout_path),

        "cutting_error_volume": sum_or_nan(step_cut),
        "part_remaining_rate": final_or_nan(step_remain),
        "part_occupancy_rate": final_or_nan(step_occ),
        "num_steps": int(step_cut.size),

        "step_cutting_error_volumes": json.dumps(step_cut.tolist(), ensure_ascii=False),
        "step_part_remaining_rates": json.dumps(step_remain.tolist(), ensure_ascii=False),
        "step_part_occupancy_rates": json.dumps(step_occ.tolist(), ensure_ascii=False),

        "has_cutting_error": bool(np.nansum(step_cut) > 0) if step_cut.size else None,
    }


def build_episode_table(root: Path) -> pd.DataFrame:
    condition_roots = find_condition_roots(root)
    if not condition_roots:
        raise FileNotFoundError(
            f"No condition roots found under {root}. "
            "Please pass either an epsilon_greedy_00 directory or a parent eval directory."
        )

    rows = []
    for condition_root in condition_roots:
        for case_dir, episode_dir, rollout_path in iter_rollout_paths(condition_root):
            try:
                rows.append(summarize_one_rollout(condition_root, case_dir, episode_dir, rollout_path))
            except Exception as e:
                rows.append({
                    "condition_root": str(condition_root),
                    "case_dir": case_dir.name,
                    "case_norm": case_dir.name.lower(),
                    "episode_dir": episode_dir.name,
                    "episode": parse_episode_index(episode_dir.name),
                    "rollout_path": str(rollout_path),
                    "error": repr(e),
                })

    if not rows:
        raise FileNotFoundError(f"No rollout_data.pickle found under {root}")

    df = pd.DataFrame(rows)

    # ---- stable sort keys ----
    df["_condition_sort"] = df["condition_root"].astype(str)
    df["_case_prefix_sort"] = df["case_norm"].astype(str).str.replace(
        r"\d+$", "", regex=True
    )
    df["_case_number_sort"] = (
        df["case_norm"]
        .astype(str)
        .str.extract(r"(\d+)$")[0]
        .astype(float)
        .fillna(1e18)
    )
    df["_episode_sort"] = pd.to_numeric(df["episode"], errors="coerce").fillna(-1)

    df = df.sort_values(
        by=[
            "_condition_sort",
            "_case_prefix_sort",
            "_case_number_sort",
            "_episode_sort",
        ]
    ).drop(
        columns=[
            "_condition_sort",
            "_case_prefix_sort",
            "_case_number_sort",
            "_episode_sort",
        ]
    )

    return df


def build_case_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["cutting_error_volume", "part_remaining_rate", "part_occupancy_rate"]
    for col in metric_cols:
        if col not in df.columns:
            df[col] = np.nan

    summary = (
        df.groupby(["condition_root", "case_norm", "case_dir"], dropna=False)
        .agg(
            n_episodes=("rollout_path", "count"),
            n_success_no_cut_error=("cutting_error_volume", lambda x: int(np.sum(np.asarray(x, dtype=float) == 0.0))),
            cut_err_mean=("cutting_error_volume", "mean"),
            cut_err_std=("cutting_error_volume", "std"),
            cut_err_min=("cutting_error_volume", "min"),
            cut_err_max=("cutting_error_volume", "max"),
            remain_mean=("part_remaining_rate", "mean"),
            remain_std=("part_remaining_rate", "std"),
            remain_min=("part_remaining_rate", "min"),
            occ_mean=("part_occupancy_rate", "mean"),
            occ_std=("part_occupancy_rate", "std"),
            occ_max=("part_occupancy_rate", "max"),
        )
        .reset_index()
    )

    # Recommended champion order:
    # 1. More no-cut-error episodes
    # 2. Lower mean cutting error
    # 3. Higher mean remaining rate
    # 4. Higher mean occupancy rate
    summary = summary.sort_values(
        by=[
            "condition_root",
            "n_success_no_cut_error",
            "cut_err_mean",
            "remain_mean",
            "occ_mean",
        ],
        ascending=[True, False, True, False, False],
    )

    # Rank within each condition.
    summary["champion_rank"] = (
        summary.groupby("condition_root")
        .cumcount()
        + 1
    )

    return summary


def print_summary(summary: pd.DataFrame) -> None:
    cols = [
        "champion_rank",
        "case_dir",
        "n_episodes",
        "n_success_no_cut_error",
        "cut_err_mean",
        "cut_err_std",
        "remain_mean",
        "remain_std",
        "occ_mean",
        "occ_std",
        "occ_max",
    ]

    for condition_root, sub in summary.groupby("condition_root", sort=False):
        print("\n" + "=" * 120)
        print(f"condition_root: {condition_root}")
        print("=" * 120)
        print(sub[cols].to_string(index=False))

        champion = sub.iloc[0]
        print("\n[champion]")
        print(
            f"{champion['case_dir']} "
            f"(rank={int(champion['champion_rank'])}, "
            f"success={int(champion['n_success_no_cut_error'])}/{int(champion['n_episodes'])}, "
            f"cut_err_mean={champion['cut_err_mean']:.6g}, "
            f"remain_mean={champion['remain_mean']:.6g}, "
            f"occ_mean={champion['occ_mean']:.6g})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help=(
            "Evaluation directory. You can pass either "
            ".../epsilon_greedy_00, an experiment directory, or a higher eval root."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write per_episode_metrics.csv and case_summary.csv.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional filename prefix, e.g. simple_B_",
    )
    args = parser.parse_args()

    df = build_episode_table(args.root)
    summary = build_case_summary(df)

    print_summary(summary)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.root if args.root.is_dir() else args.root.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    episode_csv = out_dir / f"{args.prefix}per_episode_metrics.csv"
    summary_csv = out_dir / f"{args.prefix}case_summary.csv"

    df.to_csv(episode_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print("\nSaved:")
    print(f"  {episode_csv}")
    print(f"  {summary_csv}")


if __name__ == "__main__":
    main()
