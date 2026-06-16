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
class FalseSafeRecord:
    series: str
    episode: int
    step: int
    overall: float
    x_axis: float
    y_axis: float
    z_axis: float
    false_safe_count: int
    feasible_count: int
    source: Path


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def parse_series_item(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(
            "--series must be formatted as LABEL=PATH, e.g. "
            "--series Proposed=/path/to/Object_A"
        )
    label, root = text.split("=", 1)
    label = label.strip()
    root = root.strip()
    if not label:
        raise ValueError(f"Empty series label in {text!r}")
    if not root:
        raise ValueError(f"Empty series path in {text!r}")
    return label, Path(root)


def find_cost_map_logs(root: Path) -> list[Path]:
    """
    Accept either an object/case directory containing episode_* directories or a
    single episode directory containing *_cost_map_logs.pickle files.
    """
    if not root.exists():
        raise FileNotFoundError(root)

    direct = sorted(root.glob("*_cost_map_logs.pickle"))
    if direct:
        return direct

    nested = sorted(root.glob("episode_*/*_cost_map_logs.pickle"))
    if nested:
        return nested

    recursive = sorted(root.rglob("*_cost_map_logs.pickle"))
    if recursive:
        return recursive

    raise FileNotFoundError(f"No *_cost_map_logs.pickle files found under {root}")


def parse_episode_idx(path: Path) -> int:
    match = re.match(r"episode_(\d+)$", path.parent.name)
    if match:
        return int(match.group(1))
    return 0


def parse_step_idx(path: Path) -> int:
    match = re.match(r"(\d+)_cost_map_logs\.pickle$", path.name)
    if not match:
        raise ValueError(f"Could not parse step index from {path.name}")
    return int(match.group(1))


def read_false_safe_record(path: Path, series: str) -> FalseSafeRecord | None:
    logs = load_pickle(path)
    metric = logs.get("false_safe_rate")
    if metric is None:
        return None
    if not metric.get("available", False):
        return None

    per_axis = metric.get("per_axis", {})
    return FalseSafeRecord(
        series=series,
        episode=parse_episode_idx(path),
        step=parse_step_idx(path),
        overall=float(metric["overall"]),
        x_axis=float(per_axis.get("x", np.nan)),
        y_axis=float(per_axis.get("y", np.nan)),
        z_axis=float(per_axis.get("z", np.nan)),
        false_safe_count=int(metric.get("false_safe_count", 0)),
        feasible_count=int(metric.get("feasible_count", 0)),
        source=path,
    )


def collect_records(series_roots: Iterable[tuple[str, Path]]) -> list[FalseSafeRecord]:
    records: list[FalseSafeRecord] = []
    for series, root in series_roots:
        for path in find_cost_map_logs(root):
            record = read_false_safe_record(path, series)
            if record is not None:
                records.append(record)
    if not records:
        raise RuntimeError(
            "No false-safe rate records were found. Re-run evaluation after the "
            "false_safe_rate logging change, then point this script to the new "
            "episode/object directory."
        )
    return sorted(records, key=lambda r: (r.series, r.episode, r.step))


def write_records_csv(records: list[FalseSafeRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "series",
                "episode",
                "step",
                "overall",
                "x_axis",
                "y_axis",
                "z_axis",
                "false_safe_count",
                "feasible_count",
                "source",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "series": r.series,
                "episode": r.episode,
                "step": r.step,
                "overall": r.overall,
                "x_axis": r.x_axis,
                "y_axis": r.y_axis,
                "z_axis": r.z_axis,
                "false_safe_count": r.false_safe_count,
                "feasible_count": r.feasible_count,
                "source": str(r.source),
            })


def summarize_records(records: list[FalseSafeRecord]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, int], list[FalseSafeRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.series, record.step)].append(record)

    rows: list[dict[str, float | int | str]] = []
    for (series, step), items in sorted(grouped.items()):
        values = np.asarray([r.overall for r in items], dtype=float)
        values = values[~np.isnan(values)]
        false_safe_counts = np.asarray([r.false_safe_count for r in items], dtype=float)
        feasible_counts = np.asarray([r.feasible_count for r in items], dtype=float)

        # Mean/std show episode-wise behavior, while pooled_rate aggregates all
        # feasible surfaces across episodes at the same step.
        if values.size == 0:
            mean = float("nan")
            std = float("nan")
        else:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))

        feasible_total = float(np.sum(feasible_counts))
        pooled_rate = float(np.sum(false_safe_counts) / feasible_total) if feasible_total > 0 else float("nan")

        rows.append({
            "series": series,
            "step": step,
            "mean": mean,
            "std": std,
            "pooled_rate": pooled_rate,
            "n_episodes": int(len(items)),
            "false_safe_count_total": int(np.sum(false_safe_counts)),
            "feasible_count_total": int(np.sum(feasible_counts)),
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
                "pooled_rate",
                "n_episodes",
                "false_safe_count_total",
                "feasible_count_total",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(
    rows: list[dict[str, float | int | str]],
    out_path: Path,
    *,
    title: str | None,
    dpi: int,
    use_pooled_rate: bool,
) -> None:
    by_series: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        by_series[str(row["series"])].append(row)

    y_key = "pooled_rate" if use_pooled_rate else "mean"

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for series, items in by_series.items():
        items = sorted(items, key=lambda item: int(item["step"]))
        steps = np.asarray([int(item["step"]) for item in items], dtype=int)
        means = np.asarray([float(item[y_key]) for item in items], dtype=float)
        ax.plot(steps, means, marker="o", label=series)
        if not use_pooled_rate:
            stds = np.asarray([float(item["std"]) for item in items], dtype=float)
            ax.fill_between(steps, means - stds, means + stds, alpha=0.2)

    ax.set_xlabel("Task step")
    ax.set_ylabel("False-safe rate")
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
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and plot false-safe rates saved in *_cost_map_logs.pickle files. "
            "A false-safe surface is a surface judged feasible by the UCB-threshold "
            "decision although the oracle target part intersects that surface."
        )
    )
    parser.add_argument(
        "--series",
        action="append",
        default=None,
        help=(
            "Series to plot, formatted as LABEL=PATH. Can be passed multiple "
            "times, e.g. --series Proposed=/.../Object_A "
            "--series Proposed-Nocond=/.../Object_A"
        ),
    )
    parser.add_argument(
        "--eval_root",
        type=Path,
        default=None,
        help="Single object/episode root. Used only when --series is not passed.",
    )
    parser.add_argument("--label", type=str, default=None, help="Label for --eval_root.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--figure_name", type=str, default="false_safe_rate_transition.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--use_pooled_rate",
        action="store_true",
        help=(
            "Plot pooled false_safe_count / feasible_count across episodes instead "
            "of the episode-wise mean. CSV files always contain both."
        ),
    )
    args = parser.parse_args()

    if args.series:
        series_roots = [parse_series_item(item) for item in args.series]
    else:
        if args.eval_root is None:
            raise ValueError("Pass either --series LABEL=PATH or --eval_root PATH.")
        label = args.label or args.eval_root.name
        series_roots = [(label, args.eval_root)]

    records = collect_records(series_roots)
    summary = summarize_records(records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_csv = args.out_dir / "false_safe_rate_records.csv"
    summary_csv = args.out_dir / "false_safe_rate_summary.csv"
    figure_path = args.out_dir / args.figure_name

    write_records_csv(records, records_csv)
    write_summary_csv(summary, summary_csv)
    plot_summary(
        summary,
        figure_path,
        title=args.title,
        dpi=args.dpi,
        use_pooled_rate=args.use_pooled_rate,
    )

    print(f"[OK] saved records: {records_csv}")
    print(f"[OK] saved summary: {summary_csv}")
    print(f"[OK] saved figure : {figure_path}")


if __name__ == "__main__":
    main()
