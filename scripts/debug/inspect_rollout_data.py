from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def summarize_value(name: str, value: Any) -> None:
    print(f"\n[{name}]")
    print(f"  type: {type(value)}")

    if isinstance(value, np.ndarray):
        print(f"  shape: {value.shape}")
        print(f"  dtype: {value.dtype}")

        if value.size > 0:
            flat = value.reshape(-1)
            print(f"  first values: {flat[:10]}")

        return

    if isinstance(value, list):
        print(f"  len: {len(value)}")

        if len(value) > 0:
            print(f"  first item type: {type(value[0])}")
            print(f"  first item: {value[0]}")

        return

    if isinstance(value, dict):
        print(f"  keys: {list(value.keys())}")

        if len(value) > 0:
            first_key = next(iter(value))
            print(f"  first item: {first_key} -> {value[first_key]}")

        return

    print(f"  value: {value}")


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, but got {type(data)}")

    return data


def find_latest_rollout(root: Path) -> Path:
    candidates = list(root.rglob("rollout_data.pickle"))

    if len(candidates) == 0:
        raise FileNotFoundError(f"No rollout_data.pickle found under: {root}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to rollout_data.pickle",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory to search latest rollout_data.pickle",
    )
    args = parser.parse_args()

    if args.path is None and args.root is None:
        raise ValueError("Specify either --path or --root")

    if args.path is not None:
        rollout_path = args.path
    else:
        rollout_path = find_latest_rollout(args.root)

    print("=" * 80)
    print(f"rollout_data.pickle: {rollout_path}")
    print("=" * 80)

    data = load_pickle(rollout_path)

    print("\nKeys:")
    for key in data.keys():
        print(f"  - {key}")

    print("\nSummary:")
    for key, value in data.items():
        summarize_value(key, value)

    print("\n" + "=" * 80)
    print("Execution error check")
    print("=" * 80)

    planned_actions = data.get("planned_actions")
    executed_actions = data.get("executed_actions")
    execution_error_infos = data.get("execution_error_infos")

    if planned_actions is not None:
        print(f"\nplanned_actions: {planned_actions}")

    if executed_actions is not None:
        print(f"\nexecuted_actions: {executed_actions}")

    if planned_actions is not None and executed_actions is not None:
        print(f"\naction difference: {np.asarray(executed_actions) - np.asarray(planned_actions)}")

    if execution_error_infos is not None:
        print("\nexecution_error_infos:")
        for i, info in enumerate(execution_error_infos):
            print(f"  step {i}: {info}")


if __name__ == "__main__":
    main()
