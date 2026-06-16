from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import plot_oracle_regret_available_transition as available_regret


@dataclass(frozen=True)
class EpisodeTotalRow:
    series: str
    episode: int
    metric: str
    value: float
    weighted_total_regret: float
    mean_step_regret: float
    cumulative_step_regret: float
    num_steps: int
    target_hit_rate: float
    selected_len_total: float
    oracle_len_total: float
    source: str


def parse_series_item(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError("--series must be formatted as LABEL=PATH")
    label, root = text.split("=", 1)
    label = label.strip()
    root = root.strip()
    if not label or not root:
        raise ValueError(f"Invalid --series item: {text!r}")
    return label, Path(root)


def read_step_records_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def collect_step_records_from_csv(paths: list[Path]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in paths:
        records.extend(read_step_records_csv(path))
    if not records:
        raise RuntimeError("No step records were found in the provided CSV files.")
    return records


def collect_step_records_from_logs(
    series_roots: list[tuple[str, Path]],
    initial_used_global_indices: set[int],
    hit_policy: str,
) -> list[dict[str, str]]:
    rows = available_regret.collect(
        series_roots=series_roots,
        initial_unavailable=initial_used_global_indices,
        hit_policy=hit_policy,
    )
    records: list[dict[str, str]] = []
    for row in rows:
        records.append({
            "series": row.series,
            "episode": str(row.episode),
            "step": str(row.step),
            "regret": str(row.regret),
            "selected_len": str(row.selected_len),
            "oracle_len": str(row.oracle_len),
            "selected_hits_target": str(int(row.selected_hits_target)),
            "source": str(row.source),
        })
    return records


def to_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def compute_episode_total_rows(
    step_records: list[dict[str, str]],
    metric: str,
) -> list[EpisodeTotalRow]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for rec in step_records:
        grouped[(rec["series"], to_int(rec["episode"], 0))].append(rec)

    rows: list[EpisodeTotalRow] = []
    for (series, episode), items in sorted(grouped.items()):
        items = sorted(items, key=lambda r: to_int(r.get("step", "0"), 0))
        regrets = np.asarray([to_float(r.get("regret", "nan")) for r in items], dtype=float)
        selected_lens = np.asarray([to_float(r.get("selected_len", "0"), 0.0) for r in items], dtype=float)
        oracle_lens = np.asarray([to_float(r.get("oracle_len", "0"), 0.0) for r in items], dtype=float)
        hit_flags = np.asarray([to_int(r.get("selected_hits_target", "0"), 0) for r in items], dtype=float)

        valid = ~np.isnan(regrets)
        valid_regrets = regrets[valid]
        valid_oracle_lens = oracle_lens[valid]

        cumulative = float(np.sum(valid_regrets)) if valid_regrets.size else float("nan")
        mean_step = float(np.mean(valid_regrets)) if valid_regrets.size else float("nan")
        oracle_total = float(np.sum(valid_oracle_lens)) if valid_oracle_lens.size else 0.0
        if oracle_total > 0.0 and valid_regrets.size:
            weighted = float(np.sum(valid_regrets * valid_oracle_lens) / oracle_total)
        else:
            weighted = float("nan")

        metric_values = {
            "weighted_total_regret": weighted,
            "mean_step_regret": mean_step,
            "cumulative_step_regret": cumulative,
        }
        if metric not in metric_values:
            raise ValueError(
                f"Unknown metric={metric!r}. Use weighted_total_regret, "
                "mean_step_regret, or cumulative_step_regret."
            )

        source = ""
        if items:
            first_source = items[0].get("source", "")
            source = str(Path(first_source).parent) if first_source else ""

        rows.append(EpisodeTotalRow(
            series=series,
            episode=episode,
            metric=metric,
            value=float(metric_values[metric]),
            weighted_total_regret=weighted,
            mean_step_regret=mean_step,
            cumulative_step_regret=cumulative,
            num_steps=int(len(items)),
            target_hit_rate=float(np.mean(hit_flags)) if hit_flags.size else float("nan"),
            selected_len_total=float(np.sum(selected_lens)),
            oracle_len_total=float(np.sum(oracle_lens)),
            source=source,
        ))

    return rows


def summarize_episode_rows(rows: list[EpisodeTotalRow]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[EpisodeTotalRow]] = defaultdict(list)
    for row in rows:
        grouped[row.series].append(row)

    summary: list[dict[str, float | int | str]] = []
    for series, items in sorted(grouped.items()):
        values = np.asarray([r.value for r in items], dtype=float)
        values = values[~np.isnan(values)]
        summary.append({
            "series": series,
            "metric": items[0].metric,
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "std": float(np.std(values, ddof=0)) if values.size else float("nan"),
            "n_episodes": int(len(items)),
            "target_hit_rate_mean": float(np.mean([r.target_hit_rate for r in items])),
            "selected_len_total_mean": float(np.mean([r.selected_len_total for r in items])),
            "oracle_len_total_mean": float(np.mean([r.oracle_len_total for r in items])),
            "weighted_total_regret_mean": float(np.mean([r.weighted_total_regret for r in items])),
            "mean_step_regret_mean": float(np.mean([r.mean_step_regret for r in items])),
            "cumulative_step_regret_mean": float(np.mean([r.cumulative_step_regret for r in items])),
        })
    return summary


def write_episode_rows(rows: list[EpisodeTotalRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = [
            "series",
            "episode",
            "metric",
            "value",
            "weighted_total_regret",
            "mean_step_regret",
            "cumulative_step_regret",
            "num_steps",
            "target_hit_rate",
            "selected_len_total",
            "oracle_len_total",
            "source",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "series": row.series,
                "episode": row.episode,
                "metric": row.metric,
                "value": row.value,
                "weighted_total_regret": row.weighted_total_regret,
                "mean_step_regret": row.mean_step_regret,
                "cumulative_step_regret": row.cumulative_step_regret,
                "num_steps": row.num_steps,
                "target_hit_rate": row.target_hit_rate,
                "selected_len_total": row.selected_len_total,
                "oracle_len_total": row.oracle_len_total,
                "source": row.source,
            })


def write_summary(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = [
            "series",
            "metric",
            "mean",
            "std",
            "n_episodes",
            "target_hit_rate_mean",
            "selected_len_total_mean",
            "oracle_len_total_mean",
            "weighted_total_regret_mean",
            "mean_step_regret_mean",
            "cumulative_step_regret_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_bar(summary: list[dict[str, float | int | str]], path: Path, title: str | None, ylabel: str | None, dpi: int) -> None:
    labels = [str(row["series"]) for row in summary]
    means = np.asarray([float(row["mean"]) for row in summary], dtype=float)
    stds = np.asarray([float(row["std"]) for row in summary], dtype=float)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel or "Episode-total oracle regret")
    ax.set_ylim(bottom=0.0)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot episode-total oracle regret. The script can either read the "
            "oracle_regret_available_records.csv generated by the step-wise script, "
            "or compute step records directly from cost-map logs."
        )
    )
    parser.add_argument("--records_csv", action="append", default=None, type=Path)
    parser.add_argument("--series", action="append", default=None, help="LABEL=PATH. Can be passed multiple times.")
    parser.add_argument("--eval_root", type=Path, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--ylabel", type=str, default=None)
    parser.add_argument("--figure_name", type=str, default="episode_total_oracle_regret.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--metric",
        type=str,
        default="weighted_total_regret",
        choices=["weighted_total_regret", "mean_step_regret", "cumulative_step_regret"],
    )
    parser.add_argument("--hit_policy", type=str, default="one", choices=["one", "nan", "raw"])
    parser.add_argument("--initial_used_global_indices", type=str, default=None)
    args = parser.parse_args()

    if args.records_csv:
        step_records = collect_step_records_from_csv(args.records_csv)
    else:
        if args.series:
            series_roots = [parse_series_item(item) for item in args.series]
        else:
            if args.eval_root is None:
                raise ValueError("Pass --records_csv, --series LABEL=PATH, or --eval_root PATH.")
            series_roots = [(args.label or args.eval_root.name, args.eval_root)]
        step_records = collect_step_records_from_logs(
            series_roots=series_roots,
            initial_used_global_indices=available_regret.parse_int_set(args.initial_used_global_indices),
            hit_policy=args.hit_policy,
        )

    episode_rows = compute_episode_total_rows(step_records, metric=args.metric)
    summary = summarize_episode_rows(episode_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / f"episode_total_oracle_regret_records_{args.metric}.csv"
    summary_path = args.out_dir / f"episode_total_oracle_regret_summary_{args.metric}.csv"
    figure_path = args.out_dir / args.figure_name

    write_episode_rows(episode_rows, records_path)
    write_summary(summary, summary_path)
    plot_bar(summary, figure_path, title=args.title, ylabel=args.ylabel, dpi=args.dpi)

    print(f"[OK] saved records: {records_path}")
    print(f"[OK] saved summary: {summary_path}")
    print(f"[OK] saved figure : {figure_path}")


if __name__ == "__main__":
    main()
