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
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


@dataclass(frozen=True)
class SoftIoURecord:
    series: str
    episode: int
    step: int
    soft_iou: float
    intersection: float
    union: float
    predicted_target_mass: float
    ground_truth_target_area: float
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


def read_record(path: Path, series: str) -> SoftIoURecord | None:
    logs = load_pickle(path)
    metric = logs.get("target_soft_iou")
    if metric is None or not metric.get("available", False):
        return None

    return SoftIoURecord(
        series=series,
        episode=parse_episode_idx(path),
        step=parse_step_idx(path),
        soft_iou=float(metric.get("soft_iou", metric.get("overall", np.nan))),
        intersection=float(metric.get("intersection", np.nan)),
        union=float(metric.get("union", np.nan)),
        predicted_target_mass=float(metric.get("predicted_target_mass", np.nan)),
        ground_truth_target_area=float(metric.get("ground_truth_target_area", np.nan)),
        source=path,
    )


def collect_records(series_roots: Iterable[tuple[str, Path]]) -> list[SoftIoURecord]:
    records: list[SoftIoURecord] = []
    for series, root in series_roots:
        for path in find_cost_map_logs(root):
            record = read_record(path, series)
            if record is not None:
                records.append(record)
    if not records:
        raise RuntimeError(
            "No target_soft_iou records were found. Re-run evaluation after the "
            "target_soft_iou logging change, then point this script to the new "
            "episode/object directory."
        )
    return sorted(records, key=lambda r: (r.series, r.episode, r.step))


def write_records_csv(records: list[SoftIoURecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "series",
                "episode",
                "step",
                "soft_iou",
                "intersection",
                "union",
                "predicted_target_mass",
                "ground_truth_target_area",
                "source",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "series": r.series,
                "episode": r.episode,
                "step": r.step,
                "soft_iou": r.soft_iou,
                "intersection": r.intersection,
                "union": r.union,
                "predicted_target_mass": r.predicted_target_mass,
                "ground_truth_target_area": r.ground_truth_target_area,
                "source": str(r.source),
            })


def summarize_records(records: list[SoftIoURecord]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, int], list[SoftIoURecord]] = defaultdict(list)
    for record in records:
        grouped[(record.series, record.step)].append(record)

    rows: list[dict[str, float | int | str]] = []
    for (series, step), items in sorted(grouped.items()):
        values = np.asarray([r.soft_iou for r in items], dtype=float)
        values = values[~np.isnan(values)]
        rows.append({
            "series": series,
            "step": step,
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "std": float(np.std(values, ddof=0)) if values.size else float("nan"),
            "n_episodes": int(len(items)),
            "predicted_target_mass_mean": float(np.mean([r.predicted_target_mass for r in items])),
            "ground_truth_target_area_mean": float(np.mean([r.ground_truth_target_area for r in items])),
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
                "predicted_target_mass_mean",
                "ground_truth_target_area_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(
    rows: list[dict[str, float | int | str]],
    out_path: Path,
    *,
    title: str | None,
    ylabel: str | None,
    dpi: int,
    legend_outside_top: bool,
    hide_legend: bool,
    y_tick_decimals: int | None,
    y_tick_interval: float | None,
) -> None:
    by_series: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        by_series[str(row["series"])].append(row)

    fig, ax = plt.subplots(figsize=(3, 3.4))
    for series, items in by_series.items():
        items = sorted(items, key=lambda item: int(item["step"]))
        steps = np.asarray([int(item["step"]) for item in items], dtype=int)
        means = np.asarray([float(item["mean"]) for item in items], dtype=float)
        stds = np.asarray([float(item["std"]) for item in items], dtype=float)
        ax.plot(steps, means, marker="o", label=series)
        ax.fill_between(steps, means - stds, means + stds, alpha=0.2)

    ax.set_xlabel("Task step", fontsize=12)
    ax.set_ylabel(ylabel or "Target soft IoU", fontsize=12)
    if y_tick_interval is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
    if y_tick_decimals is not None:
        ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{y_tick_decimals}f"))
    # ax.set_ylim(bottom=0.0, top=1.05)
    if title:
        ax.set_title(title)
    if len(by_series) > 1 and not hide_legend:
        if legend_outside_top:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(by_series), frameon=False)
        else:
            ax.legend(frameon=False)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and plot target soft IoU saved in *_cost_map_logs.pickle files. "
            "This metric compares the generated target-part probability map with "
            "the oracle target mask on the prediction image."
        )
    )
    parser.add_argument("--series", action="append", default=None, help="LABEL=PATH. Can be passed multiple times.")
    parser.add_argument("--eval_root", type=Path, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--ylabel", type=str, default=None)
    parser.add_argument("--figure_name", type=str, default="target_soft_iou_transition.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--legend_outside_top", action="store_true")
    parser.add_argument("--hide_legend", action="store_true")
    parser.add_argument("--y_tick_decimals", type=int, default=None)
    parser.add_argument("--y_tick_interval", type=float, default=None)
    args = parser.parse_args()

    if args.series:
        series_roots = [parse_series_item(item) for item in args.series]
    else:
        if args.eval_root is None:
            raise ValueError("Pass either --series LABEL=PATH or --eval_root PATH.")
        series_roots = [(args.label or args.eval_root.name, args.eval_root)]

    records = collect_records(series_roots)
    summary = summarize_records(records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_csv = args.out_dir / "target_soft_iou_records.csv"
    summary_csv = args.out_dir / "target_soft_iou_summary.csv"
    figure_path = args.out_dir / args.figure_name

    write_records_csv(records, records_csv)
    write_summary_csv(summary, summary_csv)
    plot_summary(
        summary,
        figure_path,
        title=args.title,
        ylabel=args.ylabel,
        dpi=args.dpi,
        legend_outside_top=args.legend_outside_top,
        hide_legend=args.hide_legend,
        y_tick_decimals=args.y_tick_decimals,
        y_tick_interval=args.y_tick_interval,
    )

    print(f"[OK] saved records: {records_csv}")
    print(f"[OK] saved summary: {summary_csv}")
    print(f"[OK] saved figure : {figure_path}")


if __name__ == "__main__":
    main()
