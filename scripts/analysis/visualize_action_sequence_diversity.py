# scripts/analysis/visualize_action_sequence_diversity.py
from __future__ import annotations

import argparse
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import copy

from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def safe_name(text: Any) -> str:
    text = str(text)
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = re.sub(r"[^0-9a-zA-Z_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def parse_episode_idx(path: Path) -> int:
    match = re.search(r"episode[_-](\d+)", path.parent.name)
    if match is None:
        return -1
    return int(match.group(1))


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, but got {type(data)}")

    return data


def load_condition_metadata(root: Path) -> dict[str, Any]:
    path = root / "condition_metadata.yaml"
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if isinstance(data, dict) else {}


def read_optional_str(row: pd.Series, key: str, default: str | None = None) -> str | None:
    if key not in row:
        return default

    value = row[key]
    if pd.isna(value):
        return default

    return str(value)


def infer_run_id(
    root: Path,
    metadata: dict[str, Any],
    manifest_row: pd.Series | None,
) -> str:
    if manifest_row is not None:
        for key in ["run_id", "label", "condition", "method"]:
            value = read_optional_str(manifest_row, key)
            if value:
                return safe_name(value)

    condition = metadata.get("condition")
    if condition:
        return safe_name(condition)

    return safe_name(root.name)


def infer_condition(
    metadata: dict[str, Any],
    manifest_row: pd.Series | None,
) -> str:
    if manifest_row is not None:
        value = read_optional_str(manifest_row, "condition")
        if value:
            return value

    value = metadata.get("condition")
    if value:
        return str(value)

    return "unknown"


def infer_method(
    metadata: dict[str, Any],
    manifest_row: pd.Series | None,
) -> str:
    if manifest_row is not None:
        value = read_optional_str(manifest_row, "method")
        if value:
            return value

    eval_meta = metadata.get("eval", {})
    if isinstance(eval_meta, dict):
        infer_model = eval_meta.get("infer_model")
        if infer_model:
            return str(infer_model)

    return "unknown"


def infer_object_label(case: str) -> str:
    text = str(case)
    if text.startswith("Object_"):
        suffix = text.replace("Object_", "", 1)
        if len(suffix) == 1 and suffix.isalpha():
            return f"Object {suffix}"
    return text


def find_rollout_files(root: Path) -> list[Path]:
    candidates = sorted(root.rglob("rollout_data.pickle"))

    if not candidates:
        raise FileNotFoundError(f"No rollout_data.pickle found under: {root}")

    return candidates


def scalarize(value: Any) -> int | float | str:
    if isinstance(value, np.generic):
        return value.item()

    return value


def flatten_numeric_values(value: Any) -> list[int]:
    """
    Best-effort conversion of an action object into global action indices.

    Supported examples:
      - ActionCandidates with .to_list()
      - ActionCandidates with .global_indices
      - list[int]
      - np.ndarray of ints
      - dict with "global_indices"
    """
    if value is None:
        return []

    if hasattr(value, "to_list"):
        return [int(v) for v in value.to_list()]

    if hasattr(value, "global_indices"):
        return [int(v) for v in value.global_indices]

    if isinstance(value, dict):
        for key in ["global_indices", "indices", "actions", "values"]:
            if key in value:
                return flatten_numeric_values(value[key])
        return []

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.ndim == 0:
                return flatten_numeric_values(value.item())

            values: list[int] = []
            for item in value.tolist():
                values.extend(flatten_numeric_values(item))
            return values

        return [int(v) for v in value.reshape(-1).tolist()]

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return []

        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in value):
            return [int(v) for v in value]

        values: list[int] = []
        for item in value:
            values.extend(flatten_numeric_values(item))
        return values

    if isinstance(value, (int, float, np.integer, np.floating)):
        return [int(value)]

    return []


def extract_axis(value: Any) -> str | None:
    if value is None:
        return None

    if hasattr(value, "axis"):
        try:
            return str(value.axis)
        except Exception:
            return None

    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.ndim == 0:
            return extract_axis(value.item())

    if isinstance(value, (list, tuple)) and value:
        return extract_axis(value[0])

    return None


def normalize_action_steps(action_value: Any) -> list[Any]:
    """
    Convert rollout_data[action_key] to a list where each item corresponds to one step.

    Usually action_value is an array/list of ActionCandidates. This function keeps
    each step object intact so flatten_numeric_values(step_obj) can extract the
    selected global indices.
    """
    if action_value is None:
        return []

    if isinstance(action_value, np.ndarray):
        if action_value.ndim == 0:
            return [action_value.item()]

        if action_value.dtype == object:
            return list(action_value.tolist())

        if action_value.ndim == 1:
            return [int(v) for v in action_value.tolist()]

        return [row for row in action_value]

    if isinstance(action_value, (list, tuple)):
        return list(action_value)

    return [action_value]


def action_signature(indices: list[int]) -> str:
    if not indices:
        return "none"

    sorted_indices = list(indices)
    if len(sorted_indices) == 1:
        return str(sorted_indices[0])

    return f"{sorted_indices[0]}-{sorted_indices[-1]}"


def action_row_from_step(
    *,
    run_id: str,
    method: str,
    condition: str,
    root: Path,
    case: str,
    episode_idx: int,
    step_idx: int,
    step_action: Any,
    action_key: str,
) -> dict[str, Any]:
    indices = flatten_numeric_values(step_action)
    axis = extract_axis(step_action)

    if indices:
        action_min = int(np.min(indices))
        action_max = int(np.max(indices))
        action_first = int(indices[0])
        action_last = int(indices[-1])
        action_center = float(np.mean(indices))
        action_len = int(len(indices))
    else:
        action_min = np.nan
        action_max = np.nan
        action_first = np.nan
        action_last = np.nan
        action_center = np.nan
        action_len = 0

    signature = action_signature(indices)

    return {
        "run_id": run_id,
        "method": method,
        "condition": condition,
        "rollout_root": str(root),
        "case": case,
        "case_label": infer_object_label(case),
        "episode_idx": int(episode_idx),
        "step_idx": int(step_idx),
        "action_key": action_key,
        "axis": axis,
        "action_signature": signature,
        "action_indices": ",".join(str(i) for i in indices),
        "action_first": action_first,
        "action_last": action_last,
        "action_min": action_min,
        "action_max": action_max,
        "action_center": action_center,
        "action_len": action_len,
    }


def collect_action_rows_for_root(
    *,
    root: Path,
    action_key: str,
    manifest_row: pd.Series | None = None,
) -> list[dict[str, Any]]:
    metadata = load_condition_metadata(root)

    run_id = infer_run_id(root, metadata, manifest_row)
    condition = infer_condition(metadata, manifest_row)
    method = infer_method(metadata, manifest_row)

    rows: list[dict[str, Any]] = []

    for rollout_path in find_rollout_files(root):
        case = rollout_path.parent.parent.name
        episode_idx = parse_episode_idx(rollout_path)

        data = load_pickle(rollout_path)
        action_steps = normalize_action_steps(data.get(action_key))

        for step_idx, step_action in enumerate(action_steps):
            rows.append(
                action_row_from_step(
                    run_id=run_id,
                    method=method,
                    condition=condition,
                    root=root,
                    case=case,
                    episode_idx=episode_idx,
                    step_idx=step_idx,
                    step_action=step_action,
                    action_key=action_key,
                )
            )

    return rows


def collect_from_manifest(
    manifest_path: Path,
    action_key: str,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)

    if "rollout_root" not in manifest.columns:
        raise ValueError(
            "Manifest must contain a 'rollout_root' column. "
            f"Available columns: {list(manifest.columns)}"
        )

    rows: list[dict[str, Any]] = []

    for _, spec in manifest.iterrows():
        if "enabled" in spec and not parse_bool(spec["enabled"]):
            continue

        root = Path(str(spec["rollout_root"]))
        root_rows = collect_action_rows_for_root(
            root=root,
            action_key=action_key,
            manifest_row=spec,
        )

        for row in root_rows:
            for col in manifest.columns:
                if col in row:
                    row[f"manifest_{col}"] = spec[col]
                else:
                    row[col] = spec[col]

        rows.extend(root_rows)

    if not rows:
        raise RuntimeError(f"No action rows were collected from manifest: {manifest_path}")

    return pd.DataFrame(rows)


def collect_from_root(
    root: Path,
    action_key: str,
    run_id: str | None = None,
    method: str | None = None,
    condition: str | None = None,
) -> pd.DataFrame:
    manifest_row = None

    if any(v is not None for v in [run_id, method, condition]):
        manifest_row = pd.Series(
            {
                "run_id": run_id,
                "method": method,
                "condition": condition,
                "rollout_root": str(root),
            }
        )

    rows = collect_action_rows_for_root(
        root=root,
        action_key=action_key,
        manifest_row=manifest_row,
    )

    if not rows:
        raise RuntimeError(f"No action rows were collected from root: {root}")

    return pd.DataFrame(rows)


def entropy_from_counts(counts: list[int]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0

    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)

    return float(entropy)


def build_episode_sequence_df(action_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    group_cols = [
        "run_id",
        "method",
        "condition",
        "rollout_root",
        "case",
        "case_label",
        "episode_idx",
    ]

    for keys, group in action_df.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, keys))
        group = group.sort_values("step_idx")

        signatures = group["action_signature"].astype(str).tolist()
        centers = group["action_center"].tolist()
        lengths = group["action_len"].tolist()

        rows.append(
            {
                **key_dict,
                "num_steps": int(group["step_idx"].nunique()),
                "sequence_signature": "|".join(signatures),
                "action_center_sequence": ",".join(
                    "" if pd.isna(v) else f"{float(v):.3f}" for v in centers
                ),
                "action_len_sequence": ",".join(str(int(v)) for v in lengths),
            }
        )

    return pd.DataFrame(rows)


def build_diversity_by_step(action_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    group_cols = [
        "run_id",
        "method",
        "condition",
        "case",
        "case_label",
        "step_idx",
    ]

    for keys, group in action_df.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, keys))

        signature_counts = Counter(group["action_signature"].astype(str).tolist())
        center_values = group["action_center"].dropna().to_numpy(dtype=float)
        len_values = group["action_len"].to_numpy(dtype=float)

        rows.append(
            {
                **key_dict,
                "num_episodes": int(group["episode_idx"].nunique()),
                "unique_action_ranges": int(len(signature_counts)),
                "action_range_entropy": entropy_from_counts(list(signature_counts.values())),
                "most_common_action_range": signature_counts.most_common(1)[0][0],
                "most_common_action_range_count": int(signature_counts.most_common(1)[0][1]),
                "action_center_mean": (
                    float(np.mean(center_values)) if len(center_values) > 0 else np.nan
                ),
                "action_center_std": (
                    float(np.std(center_values, ddof=1)) if len(center_values) > 1 else 0.0
                ),
                "action_len_mean": float(np.mean(len_values)) if len(len_values) > 0 else 0.0,
                "action_len_std": (
                    float(np.std(len_values, ddof=1)) if len(len_values) > 1 else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def normalized_hamming_distance(seq_a: list[str], seq_b: list[str]) -> float:
    n = max(len(seq_a), len(seq_b))
    if n == 0:
        return 0.0

    padded_a = seq_a + ["<missing>"] * (n - len(seq_a))
    padded_b = seq_b + ["<missing>"] * (n - len(seq_b))

    diff = sum(1 for a, b in zip(padded_a, padded_b) if a != b)
    return diff / n


def mean_pairwise_sequence_distance(sequences: list[str]) -> float:
    if len(sequences) <= 1:
        return 0.0

    split_sequences = [seq.split("|") if seq else [] for seq in sequences]

    distances = []
    for i in range(len(split_sequences)):
        for j in range(i + 1, len(split_sequences)):
            distances.append(
                normalized_hamming_distance(split_sequences[i], split_sequences[j])
            )

    return float(np.mean(distances)) if distances else 0.0


def build_diversity_by_object(
    episode_sequence_df: pd.DataFrame,
    diversity_by_step_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    group_cols = [
        "run_id",
        "method",
        "condition",
        "case",
        "case_label",
    ]

    for keys, group in episode_sequence_df.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, keys))
        sequences = group["sequence_signature"].astype(str).tolist()
        sequence_counts = Counter(sequences)

        step_group = diversity_by_step_df
        for col, value in key_dict.items():
            step_group = step_group[step_group[col] == value]

        if len(step_group) > 0:
            step_diverse = step_group["unique_action_ranges"].to_numpy(dtype=float) > 1
            diverse_step_fraction = float(np.mean(step_diverse))
            mean_step_entropy = float(np.mean(step_group["action_range_entropy"]))
            max_step_entropy = float(np.max(step_group["action_range_entropy"]))
        else:
            diverse_step_fraction = 0.0
            mean_step_entropy = 0.0
            max_step_entropy = 0.0

        rows.append(
            {
                **key_dict,
                "num_episodes": int(group["episode_idx"].nunique()),
                "num_steps": int(group["num_steps"].max()),
                "unique_sequences": int(len(sequence_counts)),
                "sequence_entropy": entropy_from_counts(list(sequence_counts.values())),
                "most_common_sequence": sequence_counts.most_common(1)[0][0],
                "most_common_sequence_count": int(sequence_counts.most_common(1)[0][1]),
                "mean_pairwise_sequence_distance": mean_pairwise_sequence_distance(sequences),
                "diverse_step_fraction": diverse_step_fraction,
                "mean_step_action_entropy": mean_step_entropy,
                "max_step_action_entropy": max_step_entropy,
            }
        )

    return pd.DataFrame(rows)


def build_action_occupancy_df(action_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in action_df.iterrows():
        indices_text = str(row.get("action_indices", ""))
        if indices_text.strip() == "":
            continue

        indices = [int(x) for x in indices_text.split(",") if x != ""]

        for action_idx in indices:
            rows.append(
                {
                    "run_id": row["run_id"],
                    "method": row["method"],
                    "condition": row["condition"],
                    "case": row["case"],
                    "case_label": row["case_label"],
                    "episode_idx": int(row["episode_idx"]),
                    "step_idx": int(row["step_idx"]),
                    "action_index": int(action_idx),
                }
            )

    occupancy_long = pd.DataFrame(rows)

    if occupancy_long.empty:
        return occupancy_long

    return (
        occupancy_long
        .groupby(
            [
                "run_id",
                "method",
                "condition",
                "case",
                "case_label",
                "step_idx",
                "action_index",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )


def apply_publication_style() -> None:
    """
    Clean matplotlib style for paper/supplementary figures.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_label(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    if text.lower() in {"unknown", "nan", "none", ""}:
        return ""

    return text


def make_plot_title(
    *,
    title: str,
    case_label: str,
    method: str,
    condition: str,
) -> str:
    method_label = clean_label(method)
    condition_label = clean_label(condition)

    first_line = f"{title}: {case_label}"
    if method_label:
        first_line += f" / {method_label}"

    lines = [first_line]
    if condition_label:
        lines.append(condition_label)

    return "\n".join(lines)


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, path: Path, *, save_pdf: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    if save_pdf:
        fig.savefig(
            path.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=0.04,
        )


def save_sequence_heatmaps(
    action_df: pd.DataFrame,
    out_dir: Path,
    *,
    value_column: str = "action_center",
) -> None:
    plot_dir = out_dir / "plots" / "sequence_heatmap"
    plot_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["run_id", "method", "condition", "case", "case_label"]

    cmap = copy.copy(plt.cm.cividis)
    cmap.set_bad("#F2F2F2")

    for keys, group in action_df.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, keys))

        pivot = (
            group
            .pivot_table(
                index="episode_idx",
                columns="step_idx",
                values=value_column,
                aggfunc="mean",
            )
            .sort_index()
        )

        if pivot.empty:
            continue

        data = pivot.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(data)

        fig_width = max(6.4, pivot.shape[1] * 0.72)
        fig_height = max(3.0, pivot.shape[0] * 0.42)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        image = ax.imshow(
            masked,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
        )

        cbar = fig.colorbar(
            image,
            ax=ax,
            pad=0.012,
            fraction=0.045,
        )
        cbar.set_label("Action center")
        cbar.outline.set_visible(False)

        ax.set_title(
            make_plot_title(
                title="Action sequence",
                case_label=str(key_dict["case_label"]),
                method=str(key_dict["method"]),
                condition=str(key_dict["condition"]),
            ),
            pad=10,
        )

        ax.set_xlabel("Planning step")
        ax.set_ylabel("Episode")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])

        # Thin cell separators.
        ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)

        despine(ax)
        fig.tight_layout()

        filename = (
            f"{safe_name(key_dict['run_id'])}__"
            f"{safe_name(key_dict['case'])}__sequence_heatmap.png"
        )
        save_figure(fig, plot_dir / filename)
        plt.close(fig)

def save_action_occupancy_maps(
    occupancy_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Save polished action occupancy bubble plots.

    x-axis: global action index
    y-axis: planning step
    marker size/color: number of episodes selecting that action index
    """
    if occupancy_df.empty:
        return

    plot_dir = out_dir / "plots" / "action_occupancy"
    plot_dir.mkdir(parents=True, exist_ok=True)

    global_max_count = max(1.0, float(occupancy_df["count"].max()))
    cmap = copy.copy(plt.cm.cividis)

    group_cols = ["run_id", "method", "condition", "case", "case_label"]

    for keys, group in occupancy_df.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, keys))

        group = group.sort_values(["step_idx", "action_index"]).copy()

        x = group["action_index"].to_numpy(dtype=float)
        y = group["step_idx"].to_numpy(dtype=float)
        count = group["count"].to_numpy(dtype=float)

        min_action = int(np.nanmin(x))
        max_action = int(np.nanmax(x))
        min_step = int(np.nanmin(y))
        max_step = int(np.nanmax(y))

        # Sparse occupancy looks better as bubbles than as an imshow heatmap.
        marker_size = 35.0 + 320.0 * np.power(count / global_max_count, 0.85)

        fig_width = max(6.8, min(11.0, 4.5 + 0.12 * (max_action - min_action + 1)))
        fig_height = max(3.2, 2.2 + 0.36 * (max_step - min_step + 1))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Light step guide lines.
        for step in range(min_step, max_step + 1):
            ax.axhline(
                step,
                color="#EEEEEE",
                linewidth=0.8,
                zorder=0,
            )

        scatter = ax.scatter(
            x,
            y,
            s=marker_size,
            c=count,
            cmap=cmap,
            norm=Normalize(vmin=0.0, vmax=global_max_count),
            edgecolors="white",
            linewidths=0.65,
            alpha=0.96,
            zorder=3,
        )

        # Annotate only the most frequent selections to avoid clutter.
        annotate_threshold = max(2.0, global_max_count * 0.75)
        if len(group) <= 80:
            for xi, yi, ci in zip(x, y, count):
                if ci >= annotate_threshold:
                    ax.text(
                        xi,
                        yi,
                        f"{int(ci)}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#111111",
                        zorder=4,
                    )

        cbar = fig.colorbar(
            scatter,
            ax=ax,
            pad=0.012,
            fraction=0.045,
        )
        cbar.set_label("Episodes")
        cbar.outline.set_visible(False)

        ax.set_title(
            make_plot_title(
                title="Action occupancy",
                case_label=str(key_dict["case_label"]),
                method=str(key_dict["method"]),
                condition=str(key_dict["condition"]),
            ),
            pad=10,
        )

        ax.set_xlabel("Global action index")
        ax.set_ylabel("Planning step")

        ax.set_xlim(min_action - 1.0, max_action + 1.0)
        ax.set_ylim(max_step + 0.65, min_step - 0.65)  # invert: step 0 on top

        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.tick_params(axis="x", rotation=0)
        ax.grid(
            axis="x",
            color="#F0F0F0",
            linewidth=0.55,
            alpha=0.8,
        )

        despine(ax)
        fig.tight_layout()

        filename = (
            f"{safe_name(key_dict['run_id'])}__"
            f"{safe_name(key_dict['case'])}__action_occupancy.png"
        )
        save_figure(fig, plot_dir / filename)
        plt.close(fig)

def save_diversity_by_step_plots(
    diversity_by_step_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    plot_dir = out_dir / "plots" / "diversity_by_step"
    plot_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["run_id", "method", "condition", "case", "case_label"]

    for keys, group in diversity_by_step_df.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, keys))
        group = group.sort_values("step_idx")

        x = group["step_idx"].to_numpy(dtype=int)
        y = group["unique_action_ranges"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(6.4, 3.2))

        baseline = np.ones_like(y)

        ax.fill_between(
            x,
            baseline,
            y,
            where=y >= baseline,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            x,
            y,
            linewidth=2.0,
            marker="o",
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )

        ax.set_title(
            make_plot_title(
                title="Action diversity by step",
                case_label=str(key_dict["case_label"]),
                method=str(key_dict["method"]),
                condition=str(key_dict["condition"]),
            ),
            pad=10,
        )

        ax.set_xlabel("Planning step")
        ax.set_ylabel("Unique action ranges")

        y_max = max(2.0, float(np.nanmax(y)) + 0.5)
        ax.set_ylim(0.8, y_max)
        ax.set_xlim(float(np.min(x)) - 0.25, float(np.max(x)) + 0.25)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.grid(True, axis="y")
        ax.grid(True, axis="x", alpha=0.25)

        despine(ax)
        fig.tight_layout()

        filename = (
            f"{safe_name(key_dict['run_id'])}__"
            f"{safe_name(key_dict['case'])}__diversity_by_step.png"
        )
        save_figure(fig, plot_dir / filename)
        plt.close(fig)

def print_summary(diversity_by_object_df: pd.DataFrame) -> None:
    display_cols = [
        "run_id",
        "method",
        "condition",
        "case_label",
        "num_episodes",
        "num_steps",
        "unique_sequences",
        "sequence_entropy",
        "mean_pairwise_sequence_distance",
        "diverse_step_fraction",
        "mean_step_action_entropy",
    ]
    display_cols = [c for c in display_cols if c in diversity_by_object_df.columns]

    display_df = diversity_by_object_df[display_cols].copy()

    print("\nAction sequence diversity summary")
    print("=" * 120)
    if display_df.empty:
        print("(empty)")
        return

    formatters = {}
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            formatters[col] = lambda x: "" if pd.isna(x) else f"{x:.3f}"

    print(display_df.to_string(index=False, formatters=formatters))


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
            "Optional columns: run_id, method, condition, enabled."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory to save CSV files and plots.",
    )
    parser.add_argument(
        "--action_key",
        type=str,
        default="executed_actions",
        choices=["executed_actions", "planned_actions", "actions"],
        help=(
            "Which action sequence to analyze. "
            "Use planned_actions for policy output; executed_actions includes execution errors."
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Only save CSV files.",
    )

    args = parser.parse_args()

    if args.root is None and args.manifest is None:
        raise ValueError("Specify either --root or --manifest.")

    if args.root is not None and args.manifest is not None:
        raise ValueError("Specify only one of --root or --manifest.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest is not None:
        action_df = collect_from_manifest(
            manifest_path=args.manifest,
            action_key=args.action_key,
        )
    else:
        action_df = collect_from_root(
            root=args.root,
            action_key=args.action_key,
            run_id=args.run_id,
            method=args.method,
            condition=args.condition,
        )

    action_df = action_df.sort_values(
        ["run_id", "case", "episode_idx", "step_idx"],
        kind="stable",
    )

    episode_sequence_df = build_episode_sequence_df(action_df)
    diversity_by_step_df = build_diversity_by_step(action_df)
    diversity_by_object_df = build_diversity_by_object(
        episode_sequence_df=episode_sequence_df,
        diversity_by_step_df=diversity_by_step_df,
    )
    occupancy_df = build_action_occupancy_df(action_df)

    action_long_path = args.out_dir / "action_sequence_long.csv"
    episode_sequence_path = args.out_dir / "action_sequence_by_episode.csv"
    diversity_by_step_path = args.out_dir / "action_diversity_by_step.csv"
    diversity_by_object_path = args.out_dir / "action_diversity_by_object.csv"
    occupancy_path = args.out_dir / "action_occupancy_long.csv"

    action_df.to_csv(action_long_path, index=False)
    episode_sequence_df.to_csv(episode_sequence_path, index=False)
    diversity_by_step_df.to_csv(diversity_by_step_path, index=False)
    diversity_by_object_df.to_csv(diversity_by_object_path, index=False)
    occupancy_df.to_csv(occupancy_path, index=False)

    print(f"[OK] Saved action sequence long      : {action_long_path}")
    print(f"[OK] Saved action sequence episodes  : {episode_sequence_path}")
    print(f"[OK] Saved diversity by step         : {diversity_by_step_path}")
    print(f"[OK] Saved diversity by object       : {diversity_by_object_path}")
    print(f"[OK] Saved action occupancy long     : {occupancy_path}")

    if not args.no_plots:
        apply_publication_style()
        save_sequence_heatmaps(action_df, args.out_dir)
        save_action_occupancy_maps(occupancy_df, args.out_dir)
        save_diversity_by_step_plots(diversity_by_step_df, args.out_dir)
        print(f"[OK] Saved plots under              : {args.out_dir / 'plots'}")

    print_summary(diversity_by_object_df)


if __name__ == "__main__":
    main()
