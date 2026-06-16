from __future__ import annotations

import argparse
import csv
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class OracleRegretRecord:
    series: str
    episode: int
    step: int
    regret: float
    selected_len: int
    oracle_len: int
    selected_has_target: bool
    selected_axis: str
    oracle_axis: str
    source: Path


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def parse_series_item(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError("--series must be formatted as LABEL=PATH")
    label, root = text.split("=", 1)
    label = label.strip()
    root = root.strip()
    if not label or not root:
        raise ValueError(f"Invalid --series item: {text!r}")
    return label, Path(root)


def find_cost_map_logs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    for pattern in ("*_cost_map_logs.pickle", "episode_*/*_cost_map_logs.pickle"):
        found = sorted(root.glob(pattern))
        if found:
            return found
    found = sorted(root.rglob("*_cost_map_logs.pickle"))
    if found:
        return found
    raise FileNotFoundError(f"No *_cost_map_logs.pickle files found under {root}")


def parse_episode_idx(path: Path) -> int:
    match = re.match(r"episode_(\d+)$", path.parent.name)
    return int(match.group(1)) if match else 0


def parse_step_idx(path: Path) -> int:
    match = re.match(r"(\d+)_cost_map_logs\.pickle$", path.name)
    if not match:
        raise ValueError(f"Could not parse step index from {path.name}")
    return int(match.group(1))


def load_gt_presence(logs: dict) -> dict[str, np.ndarray]:
    for key in ("risk_recall", "false_safe_rate", "brier_score"):
        metric = logs.get(key)
        if not isinstance(metric, dict):
            continue
        if not metric.get("available", False):
            continue
        gt = metric.get("ground_truth_presence")
        if gt is None:
            continue
        return {
            "x": np.asarray(gt["x"], dtype=bool).reshape(-1),
            "y": np.asarray(gt["y"], dtype=bool).reshape(-1),
            "z": np.asarray(gt["z"], dtype=bool).reshape(-1),
        }
    raise KeyError(
        "No ground_truth_presence was found in this cost-map log. "
        "Use logs produced after adding brier_score, false_safe_rate, or risk_recall."
    )


def axis_lengths(gt: dict[str, np.ndarray]) -> dict[str, int]:
    return {axis: int(values.size) for axis, values in gt.items()}


def global_to_axis_local(global_index: int, lengths: dict[str, int]) -> tuple[str, int]:
    idx = int(global_index)
    z_len = lengths["z"]
    x_len = lengths["x"]
    y_len = lengths["y"]
    if 0 <= idx < z_len:
        return "z", idx
    if z_len <= idx < z_len + x_len:
        return "x", idx - z_len
    if z_len + x_len <= idx < z_len + x_len + y_len:
        return "y", idx - z_len - x_len
    raise ValueError(f"global_index={idx} is out of range for axis lengths {lengths}")


def axis_local_to_global(axis: str, local_index: int, lengths: dict[str, int]) -> int:
    if axis == "z":
        return int(local_index)
    if axis == "x":
        return int(lengths["z"] + local_index)
    if axis == "y":
        return int(lengths["z"] + lengths["x"] + local_index)
    raise ValueError(f"Unknown axis: {axis}")


def action_axis_and_locals(global_indices: list[int], lengths: dict[str, int]) -> tuple[str, list[int]]:
    if not global_indices:
        return "none", []
    pairs = [global_to_axis_local(idx, lengths) for idx in global_indices]
    axes = {axis for axis, _ in pairs}
    if len(axes) != 1:
        raise ValueError(f"Selected action contains multiple axes: {global_indices}")
    axis = pairs[0][0]
    locals_ = [local for _, local in pairs]
    return axis, locals_


def longest_oracle_safe_action(gt: dict[str, np.ndarray]) -> tuple[str, list[int]]:
    lengths = axis_lengths(gt)
    best_axis = "none"
    best_locals: list[int] = []

    for axis in ("z", "x", "y"):
        present = np.asarray(gt[axis], dtype=bool).reshape(-1)
        k = int(present.size)
        risky = np.flatnonzero(present)
        if risky.size == 0:
            candidates = [list(range(k))]
        else:
            start = int(risky.min())
            end = int(risky.max())
            candidates = [list(range(0, start)), list(range(end + 1, k))]
        axis_best = max(candidates, key=len)
        if len(axis_best) > len(best_locals):
            best_axis = axis
            best_locals = axis_best

    return best_axis, best_locals


def compute_regret(oracle_len: int, selected_len: int, selected_has_target: bool, unsafe_policy: str) -> float:
    if oracle_len <= 0:
        return float("nan")
    if selected_has_target:
        if unsafe_policy == "one":
            return 1.0
        if unsafe_policy == "nan":
            return float("nan")
        if unsafe_policy == "raw":
            return float((oracle_len - selected_len) / oracle_len)
        raise ValueError(f"Unknown unsafe_policy={unsafe_policy!r}")
    return float((oracle_len - selected_len) / oracle_len)


def read_record(path: Path, series: str, unsafe_policy: str) -> OracleRegretRecord | None:
    logs = load_pickle(path)
    selected = logs.get("slice_range")
    if selected is None:
        return None
    selected_globals = [int(v) for v in selected]
    gt = load_gt_presence(logs)
    lengths = axis_lengths(gt)

    selected_axis, selected_locals = action_axis_and_locals(selected_globals, lengths)
    selected_has_target = False
    if selected_axis != "none":
        selected_has_target = bool(np.any(gt[selected_axis][selected_locals]))

    oracle_axis, oracle_locals = longest_oracle_safe_action(gt)
    selected_len = int(len(selected_locals))
    oracle_len = int(len(oracle_locals))
    regret = compute_regret(
        oracle_len=oracle_len,
        selected_len=selected_len,
        selected_has_target=selected_has_target,
        unsafe_policy=unsafe_policy,
    )

    return OracleRegretRecord(
        series=series,
        episode=parse_episode_idx(path),
        step=parse_step_idx(path),
        regret=regret,
        selected_len=selected_len,
        oracle_len=oracle_len,
        selected_has_target=selected_has_target,
        selected_axis=selected_axis,
        oracle_axis=oracle_axis,
        source=path,
    )


def collect_records(series_roots: Iterable[tuple[str, Path]], unsafe_policy: str) -> list[OracleRegretRecord]:
    records: list[OracleRegretRecord] = []
    for series, root in series_roots:
        for path in find_cost_map_logs(root):
            record = read_record(path, series, unsafe_policy)
            if record is not None:
                records.append(record)
    if not records:
        raise RuntimeError("No oracle regret records were found.")
    return sorted(records, key=lambda r: (r.series, r.episode, r.step))


def write_records_csv(records: list[OracleRegretRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "series",
                "episode",
                "step",
                "regret",
                "selected_len",
                "oracle_len",
                "selected_has_target",
                "selected_axis",
                "oracle_axis",
                "source",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "series": r.series,
                "episode": r.episode,
                "step": r.step,
                "regret": r.regret,
                "selected_len": r.selected_len,
                "oracle_len": r.oracle_len,
                "selected_has_target": int(r.selected_has_target),
                "selected_axis": r.selected_axis,
                "oracle_axis": r.oracle_axis,
                "source": str(r.source),
            })


def summarize_records(records: list[OracleRegretRecord]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, int], list[OracleRegretRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.series, record.step)].append(record)

    rows: list[dict[str, float | int | str]] = []
    for (series, step), items in sorted(grouped.items()):
        regrets = np.asarray([r.regret for r in items], dtype=float)
        regrets = regrets[~np.isnan(regrets)]
        unsafe_flags = np.asarray([r.selected_has_target for r in items], dtype=float)
        selected_lens = np.asarray([r.selected_len for r in items], dtype=float)
        oracle_lens = np.asarray([r.oracle_len for r in items], dtype=float)
        rows.append({
            "series": series,
            "step": step,
            "mean": float(np.mean(regrets)) if regrets.size else float("nan"),
            "std": float(np.std(regrets, ddof=0)) if regrets.size else float("nan"),
            "n_episodes": int(len(items)),
            "unsafe_selected_rate": float(np.mean(unsafe_flags)) if unsafe_flags.size else float("nan"),
            "selected_len_mean": float(np.mean(selected_lens)) if selected_lens.size else float("nan"),
            "oracle_len_mean": float(np.mean(oracle_lens)) if oracle_lens.size else float("nan"),
        })
    return rows


def write_summary_csv(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "series",
                "step",
                "mean",
                "std",
                "n_episodes",
                "unsafe_selected_rate",
                "selected_len_mean",
                "oracle_len_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, float | int | str]], out_path: Path, *, title: str | None, dpi: int) -> None:
    by_series: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        by_series[str(row["series"])].append(row)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for series, items in by_series.items():
        items = sorted(items, key=lambda item: int(item["step"]))
        steps = np.asarray([int(item["step"]) for item in items], dtype=int)
        means = np.asarray([float(item["mean"]) for item in items], dtype=float)
        stds = np.asarray([float(item["std"]) for item in items], dtype=float)
        ax.plot(steps, means, marker="o", label=series)
        ax.fill_between(steps, means - stds, means + stds, alpha=0.2)

    ax.set_xlabel("Task step")
    ax.set_ylabel("Oracle regret of selected action")
    ax.set_ylim(bottom=0.0)
    if title:
        ax.set_title(title)
    if len(by_series) > 1:
        ax.legend(frameon=False)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and plot oracle regret from existing cost-map logs.")
    parser.add_argument("--series", action="append", default=None, help="LABEL=PATH. Can be passed multiple times.")
    parser.add_argument("--eval_root", type=Path, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--figure_name", type=str, default="oracle_regret_transition.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--unsafe_policy",
        type=str,
        default="one",
        choices=["one", "nan", "raw"],
        help="How to score selected actions that intersect the target: one, nan, or raw.",
    )
    args = parser.parse_args()

    if args.series:
        series_roots = [parse_series_item(item) for item in args.series]
    else:
        if args.eval_root is None:
            raise ValueError("Pass either --series LABEL=PATH or --eval_root PATH.")
        label = args.label or args.eval_root.name
        series_roots = [(label, args.eval_root)]

    records = collect_records(series_roots, unsafe_policy=args.unsafe_policy)
    summary = summarize_records(records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_csv = args.out_dir / "oracle_regret_records.csv"
    summary_csv = args.out_dir / "oracle_regret_summary.csv"
    figure_path = args.out_dir / args.figure_name

    write_records_csv(records, records_csv)
    write_summary_csv(summary, summary_csv)
    plot_summary(summary, figure_path, title=args.title, dpi=args.dpi)

    print(f"[OK] saved records: {records_csv}")
    print(f"[OK] saved summary: {summary_csv}")
    print(f"[OK] saved figure : {figure_path}")


if __name__ == "__main__":
    main()
