from __future__ import annotations

import argparse
import csv
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Row:
    series: str
    episode: int
    step: int
    regret: float
    selected_len: int
    oracle_len: int
    selected_hits_target: bool
    selected_axis: str
    oracle_axis: str
    unavailable_count: int
    source: Path


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def parse_series(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError("--series must be LABEL=PATH")
    label, root = text.split("=", 1)
    if not label.strip() or not root.strip():
        raise ValueError(f"Invalid --series: {text!r}")
    return label.strip(), Path(root.strip())


def parse_int_set(text: str | None) -> set[int]:
    if not text:
        return set()
    return {int(v.strip()) for v in text.split(",") if v.strip()}


def find_logs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    for pattern in ("*_cost_map_logs.pickle", "episode_*/*_cost_map_logs.pickle"):
        found = sorted(root.glob(pattern))
        if found:
            return found
    found = sorted(root.rglob("*_cost_map_logs.pickle"))
    if not found:
        raise FileNotFoundError(f"No *_cost_map_logs.pickle under {root}")
    return found


def episode_idx(path: Path) -> int:
    match = re.match(r"episode_(\d+)$", path.parent.name)
    return int(match.group(1)) if match else 0


def step_idx(path: Path) -> int:
    match = re.match(r"(\d+)_cost_map_logs\.pickle$", path.name)
    if not match:
        raise ValueError(f"Could not parse step from {path.name}")
    return int(match.group(1))


def ground_truth_presence(logs: dict) -> dict[str, np.ndarray]:
    for key in ("risk_recall", "false_safe_rate", "brier_score"):
        metric = logs.get(key)
        if isinstance(metric, dict) and metric.get("available", False):
            gt = metric.get("ground_truth_presence")
            if gt is not None:
                return {
                    "x": np.asarray(gt["x"], dtype=bool).reshape(-1),
                    "y": np.asarray(gt["y"], dtype=bool).reshape(-1),
                    "z": np.asarray(gt["z"], dtype=bool).reshape(-1),
                }
    raise KeyError("ground_truth_presence was not found in this log.")


def lengths(gt: dict[str, np.ndarray]) -> dict[str, int]:
    return {axis: int(values.size) for axis, values in gt.items()}


def global_to_axis_local(index: int, lens: dict[str, int]) -> tuple[str, int]:
    if index < lens["z"]:
        return "z", int(index)
    if index < lens["z"] + lens["x"]:
        return "x", int(index - lens["z"])
    if index < lens["z"] + lens["x"] + lens["y"]:
        return "y", int(index - lens["z"] - lens["x"])
    raise ValueError(f"global index out of range: {index}")


def axis_local_to_global(axis: str, local: int, lens: dict[str, int]) -> int:
    if axis == "z":
        return int(local)
    if axis == "x":
        return int(lens["z"] + local)
    if axis == "y":
        return int(lens["z"] + lens["x"] + local)
    raise ValueError(axis)


def selected_axis_locals(global_indices: list[int], lens: dict[str, int]) -> tuple[str, list[int]]:
    pairs = [global_to_axis_local(i, lens) for i in global_indices]
    axes = {a for a, _ in pairs}
    if len(axes) != 1:
        raise ValueError(f"selected range spans multiple axes: {global_indices}")
    return pairs[0][0], [local for _, local in pairs]


def available_locals(axis: str, locals_iter, lens: dict[str, int], unavailable: set[int]) -> list[int]:
    out: list[int] = []
    for local in locals_iter:
        if axis_local_to_global(axis, int(local), lens) not in unavailable:
            out.append(int(local))
    return out


def oracle_best(gt: dict[str, np.ndarray], unavailable: set[int]) -> tuple[str, list[int]]:
    lens = lengths(gt)
    best_axis = "none"
    best_locals: list[int] = []
    for axis in ("z", "x", "y"):
        present = gt[axis]
        k = int(present.size)
        risky = np.flatnonzero(present)
        if risky.size == 0:
            candidates = [available_locals(axis, range(k), lens, unavailable)]
        else:
            start = int(risky.min())
            end = int(risky.max())
            top = available_locals(axis, range(0, start), lens, unavailable)
            bottom = available_locals(axis, range(k - 1, end, -1), lens, unavailable)
            candidates = [top, bottom]
        axis_best = max(candidates, key=len)
        if len(axis_best) > len(best_locals):
            best_axis = axis
            best_locals = axis_best
    return best_axis, best_locals


def regret_value(oracle_len: int, selected_len: int, selected_hits_target: bool, hit_policy: str) -> float:
    if oracle_len <= 0:
        return float("nan")
    if selected_hits_target:
        if hit_policy == "one":
            return 1.0
        if hit_policy == "nan":
            return float("nan")
        if hit_policy == "raw":
            return float((oracle_len - selected_len) / oracle_len)
        raise ValueError(hit_policy)
    return float((oracle_len - selected_len) / oracle_len)


def read_row(path: Path, series: str, unavailable_before: set[int], hit_policy: str) -> tuple[Row | None, list[int]]:
    logs = load_pickle(path)
    selected = logs.get("slice_range")
    if selected is None:
        return None, []
    selected_globals = [int(v) for v in selected]
    gt = ground_truth_presence(logs)
    lens = lengths(gt)

    sel_axis, sel_locals = selected_axis_locals(selected_globals, lens)
    sel_hits = bool(np.any(gt[sel_axis][sel_locals])) if sel_locals else False
    ora_axis, ora_locals = oracle_best(gt, set(unavailable_before))
    selected_len = int(len(sel_locals))
    oracle_len = int(len(ora_locals))

    return Row(
        series=series,
        episode=episode_idx(path),
        step=step_idx(path),
        regret=regret_value(oracle_len, selected_len, sel_hits, hit_policy),
        selected_len=selected_len,
        oracle_len=oracle_len,
        selected_hits_target=sel_hits,
        selected_axis=sel_axis,
        oracle_axis=ora_axis,
        unavailable_count=int(len(unavailable_before)),
        source=path,
    ), selected_globals


def collect(series_roots: list[tuple[str, Path]], initial_unavailable: set[int], hit_policy: str) -> list[Row]:
    rows: list[Row] = []
    for series, root in series_roots:
        paths = sorted(find_logs(root), key=lambda p: (episode_idx(p), step_idx(p)))
        unavailable_by_episode: dict[int, set[int]] = defaultdict(lambda: set(initial_unavailable))
        for path in paths:
            ep = episode_idx(path)
            unavailable = unavailable_by_episode[ep]
            row, selected_globals = read_row(path, series, unavailable, hit_policy)
            if row is not None:
                rows.append(row)
            unavailable.update(selected_globals)
    if not rows:
        raise RuntimeError("No oracle regret rows were found.")
    return sorted(rows, key=lambda r: (r.series, r.episode, r.step))


def write_records(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fields = [
            "series", "episode", "step", "regret", "selected_len", "oracle_len",
            "selected_hits_target", "selected_axis", "oracle_axis", "unavailable_count", "source",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "series": r.series,
                "episode": r.episode,
                "step": r.step,
                "regret": r.regret,
                "selected_len": r.selected_len,
                "oracle_len": r.oracle_len,
                "selected_hits_target": int(r.selected_hits_target),
                "selected_axis": r.selected_axis,
                "oracle_axis": r.oracle_axis,
                "unavailable_count": r.unavailable_count,
                "source": str(r.source),
            })


def summarize(rows: list[Row]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, int], list[Row]] = defaultdict(list)
    for row in rows:
        grouped[(row.series, row.step)].append(row)
    out: list[dict[str, float | int | str]] = []
    for (series, step), items in sorted(grouped.items()):
        regrets = np.asarray([r.regret for r in items], dtype=float)
        regrets = regrets[~np.isnan(regrets)]
        out.append({
            "series": series,
            "step": step,
            "mean": float(np.mean(regrets)) if regrets.size else float("nan"),
            "std": float(np.std(regrets, ddof=0)) if regrets.size else float("nan"),
            "n_episodes": int(len(items)),
            "target_hit_rate": float(np.mean([r.selected_hits_target for r in items])),
            "selected_len_mean": float(np.mean([r.selected_len for r in items])),
            "oracle_len_mean": float(np.mean([r.oracle_len for r in items])),
            "unavailable_count_mean": float(np.mean([r.unavailable_count for r in items])),
        })
    return out


def write_summary(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fields = [
            "series", "step", "mean", "std", "n_episodes", "target_hit_rate",
            "selected_len_mean", "oracle_len_mean", "unavailable_count_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | int | str]], path: Path, title: str | None, dpi: int) -> None:
    by_series: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        by_series[str(row["series"])].append(row)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for series, items in by_series.items():
        items = sorted(items, key=lambda x: int(x["step"]))
        steps = np.asarray([int(x["step"]) for x in items], dtype=int)
        means = np.asarray([float(x["mean"]) for x in items], dtype=float)
        stds = np.asarray([float(x["std"]) for x in items], dtype=float)
        ax.plot(steps, means, marker="o", label=series)
        ax.fill_between(steps, means - stds, means + stds, alpha=0.2)
    ax.set_xlabel("Task step")
    ax.set_ylabel("Oracle regret of selected action (available)")
    ax.set_ylim(bottom=0.0)
    if title:
        ax.set_title(title)
    if len(by_series) > 1:
        ax.legend(frameon=False)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot oracle regret using current available action space.")
    parser.add_argument("--series", action="append", default=None, help="LABEL=PATH. Can be passed multiple times.")
    parser.add_argument("--eval_root", type=Path, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--figure_name", type=str, default="oracle_regret_available_transition.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--hit_policy", type=str, default="one", choices=["one", "nan", "raw"])
    parser.add_argument("--initial_used_global_indices", type=str, default=None)
    args = parser.parse_args()

    if args.series:
        series_roots = [parse_series(item) for item in args.series]
    else:
        if args.eval_root is None:
            raise ValueError("Pass either --series LABEL=PATH or --eval_root PATH.")
        series_roots = [(args.label or args.eval_root.name, args.eval_root)]

    rows = collect(series_roots, parse_int_set(args.initial_used_global_indices), args.hit_policy)
    summary = summarize(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_csv = args.out_dir / "oracle_regret_available_records.csv"
    summary_csv = args.out_dir / "oracle_regret_available_summary.csv"
    figure_path = args.out_dir / args.figure_name
    write_records(rows, records_csv)
    write_summary(summary, summary_csv)
    plot(summary, figure_path, args.title, args.dpi)
    print(f"[OK] saved records: {records_csv}")
    print(f"[OK] saved summary: {summary_csv}")
    print(f"[OK] saved figure : {figure_path}")


if __name__ == "__main__":
    main()
