# scripts/analysis/inspect_cost_map_logs.py
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def to_array(value: Any) -> np.ndarray:
    return np.asarray(value)


def summarize_array(name: str, value: Any) -> None:
    arr = to_array(value)

    print(f"  {name}")
    print(f"    type : {type(value)}")
    print(f"    shape: {arr.shape}")
    print(f"    dtype: {arr.dtype}")

    if arr.size == 0:
        print("    empty")
        return

    flat = arr.reshape(-1)
    print(f"    min  : {np.nanmin(flat)}")
    print(f"    max  : {np.nanmax(flat)}")
    print(f"    mean : {np.nanmean(flat):.6f}")

    unique = np.unique(flat)
    if unique.size <= 20:
        print(f"    unique: {unique.tolist()}")
    else:
        print(f"    unique: {unique[:20].tolist()} ... ({unique.size} values)")


def get_color_obj(container: Any, color: str) -> Any:
    if hasattr(container, color):
        return getattr(container, color)

    if isinstance(container, dict) and color in container:
        return container[color]

    raise AttributeError(
        f"Could not find color={color!r} in object of type {type(container)}"
    )


def get_axis_value(axis_obj: Any, axis: str) -> Any:
    attr = f"{axis}_axis"

    if hasattr(axis_obj, attr):
        return getattr(axis_obj, attr)

    if isinstance(axis_obj, dict) and attr in axis_obj:
        return axis_obj[attr]

    raise AttributeError(
        f"Could not find axis attr={attr!r} in object of type {type(axis_obj)}"
    )


def inspect_cost_map_log(path: Path, color: str) -> None:
    data = load_pickle(path)

    print("=" * 100)
    print(f"file: {path}")
    print(f"top-level type: {type(data)}")

    if not isinstance(data, dict):
        print("Unexpected: cost_map_logs is not a dict.")
        return

    print(f"top-level keys: {list(data.keys())}")

    if "slice_candidate" in data:
        print(f"slice_candidate: {data['slice_candidate']}")

    if "slice_range" in data:
        print(f"slice_range: {data['slice_range']}")

    for key in ["cost_ensembles", "costs_decision"]:
        if key not in data:
            print(f"[WARN] missing key: {key}")
            continue

        print("-" * 100)
        print(key)
        obj = data[key]
        print(f"  type: {type(obj)}")

        color_obj = get_color_obj(obj, color)
        print(f"  {color} type: {type(color_obj)}")

        for axis in ["x", "y", "z"]:
            try:
                axis_value = get_axis_value(color_obj, axis)
            except AttributeError as exc:
                print(f"  [WARN] {exc}")
                continue

            summarize_array(f"{color}.{axis}_axis", axis_value)


def find_logs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]

    logs = sorted(root.glob("*_cost_map_logs.pickle"))
    if logs:
        return logs

    logs = sorted(root.rglob("*_cost_map_logs.pickle"))
    return logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to one episode directory or one *_cost_map_logs.pickle file.",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="blue",
        choices=["blue", "red", "yellow"],
        help="Color channel to inspect. Use blue for task-aligned target diagnostics.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=3,
        help="Maximum number of log files to inspect.",
    )

    args = parser.parse_args()

    logs = find_logs(args.path)

    if not logs:
        raise FileNotFoundError(f"No *_cost_map_logs.pickle found under: {args.path}")

    print(f"Found {len(logs)} cost_map_logs files.")
    print(f"Inspecting first {min(args.max_files, len(logs))} files.")

    for path in logs[: args.max_files]:
        inspect_cost_map_log(path, color=args.color)


if __name__ == "__main__":
    main()
