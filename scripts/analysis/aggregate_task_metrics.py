# scripts/analysis/aggregate_task_metrics.py
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PAPER_METRIC_COLUMNS = [
    "cutting_error_volume",
    "part_remaining_rate",
    "part_occupancy_rate",
]

CONDITION_COLUMNS = [
    "eta",
    "delta",
    "guidance_scale",
    "sample_image_num",
    "sampling_timesteps",
]


def get_nested(
    data: dict[str, Any] | None,
    keys: list[str],
    default: Any = None,
) -> Any:
    cur = data
    for key in keys:
        if cur is None:
            return default
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, but got {type(data)}")

    return data


def load_condition_metadata(root: Path) -> dict[str, Any] | None:
    metadata_path = root / "condition_metadata.yaml"

    if not metadata_path.exists():
        return None

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    if metadata is None:
        return {}

    if not isinstance(metadata, dict):
        raise TypeError(
            f"Expected dict in condition_metadata.yaml, but got {type(metadata)}: "
            f"{metadata_path}"
        )

    return metadata


def resolve_condition_info(
    root: Path,
    eta: float | None = None,
    delta: int | None = None,
    condition: str | None = None,
) -> dict[str, Any]:
    """
    Resolve eta/delta/condition for one rollout root.

    Priority:
      1. <root>/condition_metadata.yaml
      2. explicit arguments / manifest columns
      3. fallback condition name from root directory
    """
    metadata = load_condition_metadata(root)

    if metadata is not None:
        resolved_eta = float(metadata["eta"])
        resolved_delta = int(metadata["delta"])
        resolved_condition = str(
            metadata.get(
                "condition",
                build_condition_name(resolved_eta, resolved_delta),
            )
        )

        return {
            "condition": resolved_condition,
            "eta": resolved_eta,
            "delta": resolved_delta,
            "metadata_path": str(root / "condition_metadata.yaml"),
            "metadata": metadata,
        }

    if eta is None or delta is None:
        raise ValueError(
            "No condition_metadata.yaml was found and eta/delta were not provided.\n"
            f"root: {root}\n"
            "Either save condition_metadata.yaml under the rollout root, "
            "or provide eta and delta through CLI/manifest."
        )

    resolved_eta = float(eta)
    resolved_delta = int(delta)
    resolved_condition = condition or build_condition_name(resolved_eta, resolved_delta)

    return {
        "condition": resolved_condition,
        "eta": resolved_eta,
        "delta": resolved_delta,
        "guidance_scale": float(
            metadata.get(
                "guidance_scale",
                get_nested(metadata, ["policy", "inference", "guidance_scale"], np.nan),
            )
        ),
        "sample_image_num": int(
            metadata.get(
                "sample_image_num",
                get_nested(metadata, ["policy", "inference", "sample_image_num"], -1),
            )
        ),
        "sampling_timesteps": int(
            metadata.get(
                "sampling_timesteps",
                get_nested(metadata, ["policy", "inference", "sampling_timesteps"], -1),
            )
        ),
        "metadata_path": str(root / "condition_metadata.yaml"),
        "metadata": metadata,
    }


def build_condition_name(eta: float, delta: int) -> str:
    return f"eta_{format_eta_label(eta)}_delta_{delta}"


def format_eta_label(eta: float) -> str:
    text = f"{eta:.3f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text.replace(".", "p").replace("-", "m")


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
        <rollout_root>/<case_name>/episode_<idx>/rollout_data.pickle
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
) -> dict[str, float | int]:
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
    condition_info: dict[str, Any],
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
    episode_cumulative_normalized_cutting_error_rate = \
        float(data["episode_cumulative_normalized_cutting_error_rate"])

    case_name, episode_idx = extract_case_and_episode(rollout_path)

    execution_error_summary = summarize_execution_error_infos(
        data.get("execution_error_infos")
    )

    row = {
        "condition"         : condition_info["condition"],
        "eta"               : condition_info["eta"],
        "delta"             : condition_info["delta"],
        "guidance_scale"    : condition_info["guidance_scale"],
        "sample_image_num"  : condition_info["sample_image_num"],
        "sampling_timesteps": condition_info["sampling_timesteps"],
        "metadata_path"     : condition_info["metadata_path"],

        "case"        : case_name,
        "episode"     : episode_idx,
        "rollout_path": str(rollout_path),

        # Paper metrics
        "cutting_error_volume": float(np.sum(cutting_error_volumes)),
        "episode_cumulative_normalized_cutting_error_rate": (
            episode_cumulative_normalized_cutting_error_rate
        ),
        "part_remaining_rate": float(part_remaining_rates[-1]),
        "part_occupancy_rate": float(part_occupancy_rates[-1]),

        # Diagnostics
        "num_steps": int(len(cutting_error_volumes)),
        "step_cutting_error_volumes": cutting_error_volumes.tolist(),
        "step_part_remaining_rates": part_remaining_rates.tolist(),
        "step_part_occupancy_rates": part_occupancy_rates.tolist(),
    }

    metadata = condition_info.get("metadata")
    if metadata is not None:
        row.update(
            {
                "execution_error_enabled": metadata.get("execution_error", {}).get(
                    "enabled"
                ),
                "execution_error_mode": metadata.get("execution_error", {}).get("mode"),
                "execution_error_seed": metadata.get("execution_error", {}).get("seed"),
                "cases_name": metadata.get("eval", {}).get("cases_name"),
                "num_episodes_config": metadata.get("eval", {}).get("num_episodes"),
                "task_step_config": metadata.get("eval", {}).get("task_step"),
                "epoch": metadata.get("eval", {}).get("epoch"),
                "infer_model": metadata.get("eval", {}).get("infer_model"),
                "control_mode": metadata.get("policy", {}).get("control_mode"),
                "guidance_scale_config": get_nested(
                    metadata, ["policy", "inference", "guidance_scale"]
                ),
                "sample_image_num_config": get_nested(
                    metadata, ["policy", "inference", "sample_image_num"]
                ),
                "sampling_timesteps_config": get_nested(
                    metadata, ["policy", "inference", "sampling_timesteps"]
                ),
            }
        )

    row.update(execution_error_summary)
    return row


def find_rollout_files(root: Path) -> list[Path]:
    rollout_files = sorted(root.rglob("rollout_data.pickle"))

    if len(rollout_files) == 0:
        raise FileNotFoundError(f"No rollout_data.pickle found under: {root}")

    return rollout_files


def aggregate_single_root(
    root: Path,
    eta: float | None = None,
    delta: int | None = None,
    condition: str | None = None,
) -> pd.DataFrame:
    condition_info = resolve_condition_info(
        root=root,
        eta=eta,
        delta=delta,
        condition=condition,
    )

    rows = []
    for rollout_path in find_rollout_files(root):
        rows.append(
            summarize_rollout(
                rollout_path=rollout_path,
                condition_info=condition_info,
            )
        )

    return pd.DataFrame(rows)


def aggregate_from_manifest(manifest_path: Path) -> pd.DataFrame:
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
        all_frames.append(df)

    if len(all_frames) == 0:
        raise RuntimeError(f"No rows were aggregated from manifest: {manifest_path}")

    return pd.concat(all_frames, ignore_index=True)


def read_optional_float(row: pd.Series, key: str) -> float | None:
    if key not in row or pd.isna(row[key]):
        return None
    return float(row[key])


def read_optional_int(row: pd.Series, key: str) -> int | None:
    if key not in row or pd.isna(row[key]):
        return None
    return int(row[key])


def read_optional_str(row: pd.Series, key: str) -> str | None:
    if key not in row or pd.isna(row[key]):
        return None
    return str(row[key])


def build_summary(per_episode_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["condition"] + [
        col for col in CONDITION_COLUMNS
        if col in per_episode_df.columns
    ]

    summary_rows = []

    for keys, group in per_episode_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["num_episodes"] = int(len(group))
        row["num_cases"] = int(group["case"].nunique())

        for metric in PAPER_METRIC_COLUMNS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric}_sem"] = (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )

            # import ipdb; ipdb.set_trace()

        # Diagnostics
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
        ]
        if col in summary_df.columns
    ]

    return summary_df.sort_values(sort_cols)


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
            "Optional old-style columns: eta, delta, condition."
        ),
    )

    # Fallback arguments for old results without condition_metadata.yaml.
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

    if args.root is None and args.manifest is None:
        raise ValueError("Specify either --root or --manifest.")

    if args.root is not None and args.manifest is not None:
        raise ValueError("Specify only one of --root or --manifest.")

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
