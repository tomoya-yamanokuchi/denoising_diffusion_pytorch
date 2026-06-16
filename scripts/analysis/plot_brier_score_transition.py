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
class BrierRecord:
    series: str
    episode: int
    step: int
    overall: float
    x_axis: float
    y_axis: float
    z_axis: float
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


def read_brier_record(path: Path, series: str) -> BrierRecord | None:
    logs = load_pickle(path)
    brier = logs.get("brier_score")
    if brier is None:
        return None
    if not brier.get("available", False):
        return None

    per_axis = brier.get("per_axis", {})
    return BrierRecord(
        series=series,
        episode=parse_episode_idx(path),
        step=parse_step_idx(path),
        overall=float(brier["overall"]),
        x_axis=float(per_axis.get("x", np.nan)),
        y_axis=float(per_axis.get("y", np.nan)),
        z_axis=float(per_axis.get("z", np.nan)),
        source=path,
    )


def collect_records(series_roots: Iterable[tuple[str, Path]]) -> list[BrierRecord]:
    records: list[BrierRecord] = []
    for series, root in series_roots:
        for path in find_cost_map_logs(root):
            record = read_brier_record(path, series)
            if record is not None:
                records.append(record)
    if not records:
        raise RuntimeError(
            "No Brier score records were found. Re-run evaluation after the "
            "brier_score logging change, then point this script to the new "
            "episode/object directory."
        )
    return sorted(records, key=lambda r: (r.series, r.episode, r.step))


def write_records_csv(records: list[BrierRecord], out_path: Path) -> None:
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
                "source": str(r.source),
            })


def summarize_records(records: list[BrierRecord]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, int], list[BrierRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.series, record.step)].append(record)

    rows: list[dict[str, float | int | str]] = []
    for (series, step), items in sorted(grouped.items()):
        values = np.asarray([r.overall for r in items], dtype=float)
        rows.append({
            "series": series,
            "step": step,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "n": int(values.size),
        })
    return rows


def write_summary_csv(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["series", "step", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(
    rows: list[dict[str, float | int | str]],
    out_path: Path,
    *,
    title: str | None,
    dpi: int,
) -> None:
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
    ax.set_ylabel("Target-surface Brier score")
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
            "Aggregate and plot target-surface Brier scores saved in "
            "*_cost_map_logs.pickle files."
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
    parser.add_argument("--figure_name", type=str, default="brier_score_transition.pdf")
    parser.add_argument("--dpi", type=int, default=300)
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
    records_csv = args.out_dir / "brier_score_records.csv"
    summary_csv = args.out_dir / "brier_score_summary.csv"
    figure_path = args.out_dir / args.figure_name

    write_records_csv(records, records_csv)
    write_summary_csv(summary, summary_csv)
    plot_summary(summary, figure_path, title=args.title, dpi=args.dpi)

    print(f"[OK] saved records: {records_csv}")
    print(f"[OK] saved summary: {summary_csv}")
    print(f"[OK] saved figure : {figure_path}")


if __name__ == "__main__":
    main()
