# scripts/analysis/aggregate_stepwise_prediction_error.py
from __future__ import annotations

import argparse
import copy
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import MaxNLocator
from PIL import Image


SIMPLE_PALETTE = {
    "blue": {
        "lb": [-0.05, 0.55, 0.55],
        "ub": [0.45, 1.05, 1.05],
    },
    "red": {
        "lb": [0.70, 0.10, 0.10],
        "ub": [1.00, 0.40, 0.40],
    },
    "yellow": {
        "lb": [0.70, 0.70, 0.10],
        "ub": [1.00, 1.00, 0.80],
    },
}

COMPLEX_PALETTE = {
    "blue": {
        "lb": [-0.10, -0.10, 0.90],
        "ub": [0.10, 0.10, 1.00],
    },
    "red": {
        "lb": [0.90, -0.10, -0.10],
        "ub": [1.00, 0.10, 0.10],
    },
    "yellow": {
        "lb": [-0.10, 0.90, -0.10],
        "ub": [0.10, 1.00, 0.10],
    },
}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def safe_name(text: Any) -> str:
    text = str(text)
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^0-9a-zA-Z_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def parse_episode_idx(path: Path) -> int:
    match = re.search(r"episode[_-](\d+)", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def parse_step_idx_from_ensemble(path: Path, axis: str) -> int | None:
    pattern = rf"^(\d+)_ensemble_{axis}_axis\d+_0\.png$"
    match = re.match(pattern, path.name)
    if match is None:
        return None
    return int(match.group(1))


def parse_step_idx_from_raw_dir(path: Path) -> int | None:
    match = re.match(r"^step_(\d+)$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_condition_metadata(root: Path) -> dict[str, Any]:
    return load_yaml(root / "condition_metadata.yaml")


def read_optional_str(row: pd.Series | None, key: str, default: str | None = None) -> str | None:
    if row is None:
        return default
    if key not in row:
        return default
    value = row[key]
    if pd.isna(value):
        return default
    return str(value)


def infer_run_id(root: Path, metadata: dict[str, Any], manifest_row: pd.Series | None) -> str:
    for key in ["run_id", "label", "condition", "method"]:
        value = read_optional_str(manifest_row, key)
        if value:
            return safe_name(value)

    condition = metadata.get("condition")
    if condition:
        return safe_name(condition)

    return safe_name(root.name)


def infer_condition(metadata: dict[str, Any], manifest_row: pd.Series | None) -> str:
    value = read_optional_str(manifest_row, "condition")
    if value:
        return value
    value = metadata.get("condition")
    if value:
        return str(value)
    return "unknown"


def infer_method(metadata: dict[str, Any], manifest_row: pd.Series | None) -> str:
    value = read_optional_str(manifest_row, "method")
    if value:
        return value

    eval_meta = metadata.get("eval", {})
    if isinstance(eval_meta, dict):
        infer_model = eval_meta.get("infer_model")
        if infer_model:
            return str(infer_model)

    return "unknown"


def infer_case_label(case: str) -> str:
    text = str(case)
    if text.startswith("Object_"):
        suffix = text.replace("Object_", "", 1)
        if len(suffix) == 1 and suffix.isalpha():
            return f"Object {suffix}"
    return text


def get_palette(name: str) -> dict[str, dict[str, list[float]]] | None:
    if name == "simple":
        return SIMPLE_PALETTE
    if name == "complex":
        return COMPLEX_PALETTE
    if name == "none":
        return None
    raise ValueError(f"Unknown palette: {name}")


def load_rgb_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def ensure_same_shape(a: np.ndarray, b: np.ndarray, *, a_name: str, b_name: str) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Image shape mismatch: {a_name}={a.shape}, {b_name}={b.shape}"
        )


def non_background_mask(
    image: np.ndarray,
    *,
    background_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    tolerance: float = 0.08,
) -> np.ndarray:
    bg = np.asarray(background_rgb, dtype=np.float32).reshape(1, 1, 3)
    dist = np.linalg.norm(image - bg, axis=2)
    return dist > tolerance


def color_range_mask(
    image: np.ndarray,
    *,
    lb: list[float],
    ub: list[float],
) -> np.ndarray:
    lb_arr = np.asarray(lb, dtype=np.float32).reshape(1, 1, 3)
    ub_arr = np.asarray(ub, dtype=np.float32).reshape(1, 1, 3)
    return np.all((image >= lb_arr) & (image <= ub_arr), axis=2)


def palette_union_mask(
    image: np.ndarray,
    palette: dict[str, dict[str, list[float]]] | None,
    *,
    background_tolerance: float,
) -> np.ndarray:
    if palette is None:
        return non_background_mask(image, tolerance=background_tolerance)

    union = np.zeros(image.shape[:2], dtype=bool)
    for spec in palette.values():
        union |= color_range_mask(image, lb=spec["lb"], ub=spec["ub"])
    return union


def build_eval_mask(
    *,
    oracle_img: np.ndarray,
    seq_img: np.ndarray | None,
    oracle_target_mask: np.ndarray,
    seq_target_mask: np.ndarray | None,
    eval_region: str,
    background_tolerance: float,
) -> np.ndarray:
    if eval_region == "all":
        return np.ones(oracle_img.shape[:2], dtype=bool)

    if eval_region == "oracle_foreground":
        return non_background_mask(
            oracle_img,
            tolerance=background_tolerance,
        )

    if eval_region == "target_union":
        return oracle_target_mask.copy()

    if eval_region == "target_unobserved":
        if seq_target_mask is None:
            return oracle_target_mask.copy()
        return oracle_target_mask & (~seq_target_mask)

    raise ValueError(f"Unknown eval_region: {eval_region}")


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def safe_std(values: np.ndarray) -> float:
    values = np.asarray(values)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(num / den)


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    tp = float(np.sum(pred & gt))
    fp = float(np.sum(pred & (~gt)))
    fn = float(np.sum((~pred) & gt))
    tn = float(np.sum((~pred) & (~gt)))

    return {
        "target_iou": safe_div(tp, tp + fp + fn),
        "target_precision": safe_div(tp, tp + fp),
        "target_recall": safe_div(tp, tp + fn),
        "target_fp_pixels": fp,
        "target_fn_pixels": fn,
        "target_tp_pixels": tp,
        "target_tn_pixels": tn,
        "target_error_pixels": fp + fn,
    }


def rgb_error_metrics(
    pred_img: np.ndarray,
    oracle_img: np.ndarray,
    eval_mask: np.ndarray,
) -> dict[str, float]:
    ensure_same_shape(pred_img, oracle_img, a_name="pred", b_name="oracle")

    if eval_mask.sum() == 0:
        return {
            "rgb_mae": float("nan"),
            "rgb_mse": float("nan"),
            "rgb_rmse": float("nan"),
        }

    diff = pred_img - oracle_img
    diff_eval = diff[eval_mask]

    mse = float(np.mean(diff_eval ** 2))
    mae = float(np.mean(np.abs(diff_eval)))
    rmse = float(math.sqrt(mse))

    return {
        "rgb_mae": mae,
        "rgb_mse": mse,
        "rgb_rmse": rmse,
    }


def read_raw_prediction_samples(
    episode_dir: Path,
    step_idx: int,
) -> list[np.ndarray]:
    raw_dir = episode_dir / "raw_pred_image" / f"step_{step_idx}"
    if not raw_dir.exists():
        return []

    files = sorted(raw_dir.glob("ensemble_z_*.png"))
    return [load_rgb_image(path) for path in files]


def read_saved_ensemble_prediction(
    episode_dir: Path,
    step_idx: int,
    axis: str,
) -> np.ndarray | None:
    path = episode_dir / f"{step_idx}_ensemble_{axis}_axis{step_idx}_0.png"
    if not path.exists():
        return None
    return load_rgb_image(path)


def read_seq_observation(
    episode_dir: Path,
    step_idx: int,
    axis: str,
) -> np.ndarray | None:
    path = episode_dir / f"{step_idx}_seq_obs_cast_{axis}_axis{step_idx}_0.png"
    if not path.exists():
        return None
    return load_rgb_image(path)


def read_oracle_observation(
    episode_dir: Path,
    axis: str,
) -> np.ndarray | None:
    path = episode_dir / f"oracle_obs_cast_{axis}_axis0.png"
    if not path.exists():
        return None
    return load_rgb_image(path)


def available_step_indices(
    episode_dir: Path,
    *,
    axis: str,
    pred_source: str,
) -> list[int]:
    steps: set[int] = set()

    if pred_source in {"ensemble_image", "auto"}:
        for path in episode_dir.glob(f"*_ensemble_{axis}_axis*_0.png"):
            step_idx = parse_step_idx_from_ensemble(path, axis)
            if step_idx is not None:
                steps.add(step_idx)

    if pred_source in {"raw_mean", "auto"}:
        raw_root = episode_dir / "raw_pred_image"
        if raw_root.exists():
            for path in raw_root.glob("step_*"):
                if path.is_dir():
                    step_idx = parse_step_idx_from_raw_dir(path)
                    if step_idx is not None:
                        steps.add(step_idx)

    return sorted(steps)


def choose_prediction_image(
    *,
    episode_dir: Path,
    step_idx: int,
    axis: str,
    pred_source: str,
) -> tuple[np.ndarray | None, list[np.ndarray], str]:
    raw_samples: list[np.ndarray] = []
    pred_img: np.ndarray | None = None

    if pred_source in {"raw_mean", "auto"} and axis == "z":
        raw_samples = read_raw_prediction_samples(episode_dir, step_idx)
        if raw_samples:
            pred_img = np.mean(np.stack(raw_samples, axis=0), axis=0)
            return pred_img, raw_samples, "raw_mean"

    if pred_source in {"ensemble_image", "auto"}:
        pred_img = read_saved_ensemble_prediction(episode_dir, step_idx, axis)
        if pred_img is not None:
            return pred_img, raw_samples, "ensemble_image"

    return None, raw_samples, "missing"


def compute_sample_level_metrics(
    *,
    samples: list[np.ndarray],
    oracle_img: np.ndarray,
    eval_mask: np.ndarray,
    palette: dict[str, dict[str, list[float]]] | None,
    oracle_target_mask: np.ndarray,
    background_tolerance: float,
) -> dict[str, float]:
    if not samples:
        return {
            "sample_rgb_mae_mean": float("nan"),
            "sample_rgb_mae_std": float("nan"),
            "sample_target_iou_mean": float("nan"),
            "sample_target_iou_std": float("nan"),
        }

    mae_values = []
    iou_values = []

    for sample in samples:
        rgb_metrics = rgb_error_metrics(sample, oracle_img, eval_mask)
        mae_values.append(rgb_metrics["rgb_mae"])

        sample_target_mask = palette_union_mask(
            sample,
            palette,
            background_tolerance=background_tolerance,
        )
        bin_metrics = binary_metrics(
            pred=sample_target_mask & eval_mask,
            gt=oracle_target_mask & eval_mask,
        )
        iou_values.append(bin_metrics["target_iou"])

    mae_arr = np.asarray(mae_values, dtype=float)
    iou_arr = np.asarray(iou_values, dtype=float)

    return {
        "sample_rgb_mae_mean": float(np.nanmean(mae_arr)),
        "sample_rgb_mae_std": safe_std(mae_arr[~np.isnan(mae_arr)]),
        "sample_target_iou_mean": float(np.nanmean(iou_arr)),
        "sample_target_iou_std": safe_std(iou_arr[~np.isnan(iou_arr)]),
    }


def compute_color_metrics(
    *,
    pred_img: np.ndarray,
    oracle_img: np.ndarray,
    eval_mask: np.ndarray,
    palette: dict[str, dict[str, list[float]]] | None,
) -> dict[str, float]:
    if palette is None:
        return {}

    rows: dict[str, float] = {}

    for color_name, spec in palette.items():
        pred_mask = color_range_mask(pred_img, lb=spec["lb"], ub=spec["ub"])
        oracle_mask = color_range_mask(oracle_img, lb=spec["lb"], ub=spec["ub"])

        metrics = binary_metrics(
            pred=pred_mask & eval_mask,
            gt=oracle_mask & eval_mask,
        )

        for key, value in metrics.items():
            rows[f"{color_name}_{key}"] = value

    return rows


def compute_step_metrics(
    *,
    oracle_img: np.ndarray,
    seq_img: np.ndarray | None,
    pred_img: np.ndarray,
    raw_samples: list[np.ndarray],
    palette: dict[str, dict[str, list[float]]] | None,
    eval_region: str,
    background_tolerance: float,
) -> dict[str, float]:
    ensure_same_shape(pred_img, oracle_img, a_name="pred", b_name="oracle")

    if seq_img is not None:
        ensure_same_shape(seq_img, oracle_img, a_name="seq", b_name="oracle")

    oracle_target_mask = palette_union_mask(
        oracle_img,
        palette,
        background_tolerance=background_tolerance,
    )

    pred_target_mask = palette_union_mask(
        pred_img,
        palette,
        background_tolerance=background_tolerance,
    )

    seq_target_mask = (
        palette_union_mask(
            seq_img,
            palette,
            background_tolerance=background_tolerance,
        )
        if seq_img is not None
        else None
    )

    eval_mask = build_eval_mask(
        oracle_img=oracle_img,
        seq_img=seq_img,
        oracle_target_mask=oracle_target_mask,
        seq_target_mask=seq_target_mask,
        eval_region=eval_region,
        background_tolerance=background_tolerance,
    )

    rgb_metrics = rgb_error_metrics(pred_img, oracle_img, eval_mask)

    binary = binary_metrics(
        pred=pred_target_mask & eval_mask,
        gt=oracle_target_mask & eval_mask,
    )

    color_metrics = compute_color_metrics(
        pred_img=pred_img,
        oracle_img=oracle_img,
        eval_mask=eval_mask,
        palette=palette,
    )

    sample_metrics = compute_sample_level_metrics(
        samples=raw_samples,
        oracle_img=oracle_img,
        eval_mask=eval_mask,
        palette=palette,
        oracle_target_mask=oracle_target_mask,
        background_tolerance=background_tolerance,
    )

    oracle_target_pixels = float(np.sum(oracle_target_mask))

    if seq_target_mask is not None and oracle_target_pixels > 0:
        target_observed_fraction = float(
            np.sum(seq_target_mask & oracle_target_mask) / oracle_target_pixels
        )
    else:
        target_observed_fraction = float("nan")

    if seq_img is not None:
        observed_non_background_fraction = float(
            np.mean(
                non_background_mask(
                    seq_img,
                    tolerance=background_tolerance,
                )
            )
        )
    else:
        observed_non_background_fraction = float("nan")

    return {
        **rgb_metrics,
        **binary,
        **color_metrics,
        **sample_metrics,
        "eval_pixel_count": int(np.sum(eval_mask)),
        "oracle_target_pixels": int(oracle_target_pixels),
        "pred_target_pixels": int(np.sum(pred_target_mask)),
        "target_observed_fraction": target_observed_fraction,
        "observed_non_background_fraction": observed_non_background_fraction,
    }


def find_episode_dirs(root: Path) -> list[Path]:
    rollout_files = sorted(root.rglob("rollout_data.pickle"))
    if not rollout_files:
        raise FileNotFoundError(f"No rollout_data.pickle found under: {root}")
    return [path.parent for path in rollout_files]


def collect_rows_for_root(
    *,
    root: Path,
    axis: str,
    pred_source: str,
    palette_name: str,
    eval_region: str,
    background_tolerance: float,
    manifest_row: pd.Series | None = None,
) -> list[dict[str, Any]]:
    metadata = load_condition_metadata(root)

    run_id = infer_run_id(root, metadata, manifest_row)
    method = infer_method(metadata, manifest_row)
    condition = infer_condition(metadata, manifest_row)

    palette = get_palette(palette_name)

    rows: list[dict[str, Any]] = []

    for episode_dir in find_episode_dirs(root):
        case = episode_dir.parent.name
        episode_idx = parse_episode_idx(episode_dir)
        oracle_img = read_oracle_observation(episode_dir, axis)

        if oracle_img is None:
            print(f"[WARN] Missing oracle image: {episode_dir}")
            continue

        step_indices = available_step_indices(
            episode_dir,
            axis=axis,
            pred_source=pred_source,
        )

        for step_idx in step_indices:
            pred_img, raw_samples, actual_pred_source = choose_prediction_image(
                episode_dir=episode_dir,
                step_idx=step_idx,
                axis=axis,
                pred_source=pred_source,
            )

            if pred_img is None:
                print(f"[WARN] Missing prediction: {episode_dir}, step={step_idx}")
                continue

            seq_img = read_seq_observation(episode_dir, step_idx, axis)

            try:
                metrics = compute_step_metrics(
                    oracle_img=oracle_img,
                    seq_img=seq_img,
                    pred_img=pred_img,
                    raw_samples=raw_samples,
                    palette=palette,
                    eval_region=eval_region,
                    background_tolerance=background_tolerance,
                )
            except ValueError as exc:
                print(f"[WARN] Skipped {episode_dir}, step={step_idx}: {exc}")
                continue

            rows.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "condition": condition,
                    "rollout_root": str(root),
                    "case": case,
                    "case_label": infer_case_label(case),
                    "episode_idx": episode_idx,
                    "step_idx": step_idx,
                    "axis": axis,
                    "pred_source": actual_pred_source,
                    "palette": palette_name,
                    "eval_region": eval_region,
                    "num_raw_samples": len(raw_samples),
                    **metrics,
                }
            )

    return rows


def collect_from_manifest(
    *,
    manifest_path: Path,
    axis: str,
    pred_source: str,
    palette_name: str,
    eval_region: str,
    background_tolerance: float,
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
        root_rows = collect_rows_for_root(
            root=root,
            axis=axis,
            pred_source=pred_source,
            palette_name=palette_name,
            eval_region=eval_region,
            background_tolerance=background_tolerance,
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
        raise RuntimeError(f"No prediction-error rows were collected from {manifest_path}")

    return pd.DataFrame(rows)


def collect_from_root(
    *,
    root: Path,
    axis: str,
    pred_source: str,
    palette_name: str,
    eval_region: str,
    background_tolerance: float,
    run_id: str | None,
    method: str | None,
    condition: str | None,
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

    rows = collect_rows_for_root(
        root=root,
        axis=axis,
        pred_source=pred_source,
        palette_name=palette_name,
        eval_region=eval_region,
        background_tolerance=background_tolerance,
        manifest_row=manifest_row,
    )

    if not rows:
        raise RuntimeError(f"No prediction-error rows were collected from {root}")

    return pd.DataFrame(rows)


def metric_mean_std_sem(group: pd.DataFrame, metric: str) -> dict[str, float]:
    values = group[metric].to_numpy(dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return {
            f"{metric}_mean": float("nan"),
            f"{metric}_std": float("nan"),
            f"{metric}_sem": float("nan"),
        }

    return {
        f"{metric}_mean": float(np.mean(values)),
        f"{metric}_std": safe_std(values),
        f"{metric}_sem": float(safe_std(values) / math.sqrt(len(values))),
    }


def build_summary_by_step(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rgb_mae",
        "rgb_mse",
        "rgb_rmse",
        "target_iou",
        "target_precision",
        "target_recall",
        "target_error_pixels",
        "target_observed_fraction",
        "observed_non_background_fraction",
        "sample_rgb_mae_mean",
        "sample_target_iou_mean",
    ]

    group_cols = [
        "run_id",
        "method",
        "condition",
        "case",
        "case_label",
        "step_idx",
        "axis",
        "pred_source",
        "palette",
        "eval_region",
    ]

    rows: list[dict[str, Any]] = []

    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["num_episodes"] = int(group["episode_idx"].nunique())

        for metric in metrics:
            if metric in group.columns:
                row.update(metric_mean_std_sem(group, metric))

        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.sort_values(["run_id", "case", "step_idx"], kind="stable")


def nan_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if np.sum(mask) < 3:
        return float("nan")
    if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def build_summary_by_object(step_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "run_id",
        "method",
        "condition",
        "case",
        "case_label",
        "axis",
        "pred_source",
        "palette",
        "eval_region",
    ]

    rows: list[dict[str, Any]] = []

    for keys, group in step_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        group = group.sort_values("step_idx")

        row["num_steps"] = int(group["step_idx"].nunique())
        row["num_episodes"] = int(group["num_episodes"].max())

        for metric in [
            "rgb_mae_mean",
            "target_iou_mean",
            "target_observed_fraction_mean",
            "sample_rgb_mae_mean_mean",
            "sample_target_iou_mean_mean",
        ]:
            if metric in group.columns:
                values = group[metric].to_numpy(dtype=float)
                values = values[~np.isnan(values)]
                row[f"{metric}_over_steps"] = float(np.mean(values)) if len(values) else float("nan")

        first = group.iloc[0]
        last = group.iloc[-1]

        for metric in [
            "rgb_mae_mean",
            "target_iou_mean",
            "target_observed_fraction_mean",
            "observed_non_background_fraction_mean",
        ]:
            if metric in group.columns:
                row[f"{metric}_first"] = float(first[metric])
                row[f"{metric}_last"] = float(last[metric])
                row[f"{metric}_last_minus_first"] = float(last[metric] - first[metric])

        if "target_observed_fraction_mean" in group.columns and "rgb_mae_mean" in group.columns:
            row["corr_observed_fraction_vs_rgb_mae"] = nan_corr(
                group["target_observed_fraction_mean"].to_numpy(dtype=float),
                group["rgb_mae_mean"].to_numpy(dtype=float),
            )

        if "target_observed_fraction_mean" in group.columns and "target_iou_mean" in group.columns:
            row["corr_observed_fraction_vs_target_iou"] = nan_corr(
                group["target_observed_fraction_mean"].to_numpy(dtype=float),
                group["target_iou_mean"].to_numpy(dtype=float),
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["run_id", "case"], kind="stable")


def apply_publication_style() -> None:
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


def make_title(title: str, case_label: str, method: str, condition: str) -> str:
    method_label = clean_label(method)
    condition_label = clean_label(condition)

    first = f"{title}: {case_label}"
    if method_label:
        first += f" / {method_label}"

    lines = [first]
    if condition_label:
        lines.append(condition_label)

    return "\n".join(lines)


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, path: Path, *, save_pdf: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    if save_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)


def plot_metric_by_step(
    step_summary: pd.DataFrame,
    out_dir: Path,
    *,
    metric_mean_col: str,
    metric_std_col: str,
    ylabel: str,
    plot_name: str,
) -> None:
    plot_dir = out_dir / "plots" / plot_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["run_id", "method", "condition", "case", "case_label"]

    for keys, group in step_summary.groupby(group_cols, dropna=False):
        key = dict(zip(group_cols, keys))
        group = group.sort_values("step_idx")

        if metric_mean_col not in group.columns:
            continue

        x = group["step_idx"].to_numpy(dtype=float)
        y = group[metric_mean_col].to_numpy(dtype=float)

        if np.all(np.isnan(y)):
            continue

        if metric_std_col in group.columns:
            y_std = group[metric_std_col].to_numpy(dtype=float)
        else:
            y_std = np.zeros_like(y)

        fig, ax = plt.subplots(figsize=(6.2, 3.3))

        ax.plot(
            x,
            y,
            linewidth=2.0,
            marker="o",
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )

        if not np.all(np.isnan(y_std)):
            ax.fill_between(
                x,
                y - y_std,
                y + y_std,
                alpha=0.14,
                linewidth=0,
            )

        ax.set_title(
            make_title(
                title=ylabel,
                case_label=str(key["case_label"]),
                method=str(key["method"]),
                condition=str(key["condition"]),
            ),
            pad=10,
        )
        ax.set_xlabel("Planning step")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, axis="y")
        ax.grid(True, axis="x", alpha=0.25)

        despine(ax)
        fig.tight_layout()

        filename = (
            f"{safe_name(key['run_id'])}__"
            f"{safe_name(key['case'])}__{plot_name}.png"
        )
        save_figure(fig, plot_dir / filename)
        plt.close(fig)



def plot_metric_by_step_lineplot(
    step_summary: pd.DataFrame,
    out_dir: Path,
    *,
    metric_mean_col: str,
    metric_std_col: str,
    ylabel: str,
    plot_name: str,
) -> None:
    """
    Save ordinary line plots.

    One figure is created for each run/method/condition.
    Lines correspond to cases / objects.
    """
    plot_dir = out_dir / "plots" / plot_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["run_id", "method", "condition", "axis", "pred_source", "eval_region"]

    for keys, group in step_summary.groupby(group_cols, dropna=False):
        key = dict(zip(group_cols, keys))

        if metric_mean_col not in group.columns:
            continue

        fig, ax = plt.subplots(figsize=(6.6, 3.8))

        plotted = False

        for case_label, case_group in group.groupby("case_label", dropna=False):
            case_group = case_group.sort_values("step_idx")

            x = case_group["step_idx"].to_numpy(dtype=float)
            y = case_group[metric_mean_col].to_numpy(dtype=float)

            if np.all(np.isnan(y)):
                continue

            if metric_std_col in case_group.columns:
                y_std = case_group[metric_std_col].to_numpy(dtype=float)
            else:
                y_std = np.zeros_like(y)

            line = ax.plot(
                x,
                y,
                linewidth=2.0,
                marker="o",
                markersize=5.0,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=str(case_label),
            )[0]

            if not np.all(np.isnan(y_std)):
                color = line.get_color()
                ax.fill_between(
                    x,
                    y - y_std,
                    y + y_std,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )

            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        method_label = clean_label(key["method"])
        condition_label = clean_label(key["condition"])

        title = ylabel
        if method_label:
            title += f" / {method_label}"
        if condition_label:
            title += f" / {condition_label}"

        ax.set_title(title, pad=10)
        ax.set_xlabel("Planning step")
        ax.set_ylabel(ylabel)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, axis="y")
        ax.grid(True, axis="x", alpha=0.25)

        if "IoU" in ylabel:
            ax.set_ylim(-0.03, 1.03)
        if "fraction" in ylabel.lower():
            ax.set_ylim(-0.03, 1.03)

        ax.legend(
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
        )

        despine(ax)
        fig.tight_layout()

        filename = (
            f"{safe_name(key['run_id'])}__"
            f"{safe_name(plot_name)}__lineplot.png"
        )
        save_figure(fig, plot_dir / filename)
        plt.close(fig)


def plot_observed_vs_error_scatter(step_df: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "plots" / "observed_vs_error"
    plot_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["run_id", "method", "condition", "case", "case_label"]

    for keys, group in step_df.groupby(group_cols, dropna=False):
        key = dict(zip(group_cols, keys))

        if "target_observed_fraction" not in group.columns or "rgb_mae" not in group.columns:
            continue

        group = group.sort_values(["step_idx", "episode_idx"])
        x = group["target_observed_fraction"].to_numpy(dtype=float)
        y = group["rgb_mae"].to_numpy(dtype=float)
        c = group["step_idx"].to_numpy(dtype=float)

        mask = (~np.isnan(x)) & (~np.isnan(y))
        if np.sum(mask) == 0:
            continue

        fig, ax = plt.subplots(figsize=(5.4, 4.0))

        scatter = ax.scatter(
            x[mask],
            y[mask],
            c=c[mask],
            s=46,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.6,
            cmap=copy.copy(plt.cm.cividis),
        )

        cbar = fig.colorbar(scatter, ax=ax, pad=0.012, fraction=0.045)
        cbar.set_label("Step")
        cbar.outline.set_visible(False)

        ax.set_title(
            make_title(
                title="Observed fraction vs prediction error",
                case_label=str(key["case_label"]),
                method=str(key["method"]),
                condition=str(key["condition"]),
            ),
            pad=10,
        )
        ax.set_xlabel("Target observed fraction")
        ax.set_ylabel("RGB MAE")
        ax.set_xlim(-0.03, 1.03)
        ax.grid(True, alpha=0.45)

        despine(ax)
        fig.tight_layout()

        filename = (
            f"{safe_name(key['run_id'])}__"
            f"{safe_name(key['case'])}__observed_vs_rgb_mae.png"
        )
        save_figure(fig, plot_dir / filename)
        plt.close(fig)


def print_summary(object_summary: pd.DataFrame) -> None:
    cols = [
        "run_id",
        "method",
        "condition",
        "case_label",
        "num_episodes",
        "num_steps",
        "rgb_mae_mean_first",
        "rgb_mae_mean_last",
        "rgb_mae_mean_last_minus_first",
        "target_iou_mean_first",
        "target_iou_mean_last",
        "target_iou_mean_last_minus_first",
        "target_observed_fraction_mean_first",
        "target_observed_fraction_mean_last",
        "corr_observed_fraction_vs_rgb_mae",
    ]
    cols = [c for c in cols if c in object_summary.columns]
    display = object_summary[cols].copy()

    print("\nStep-wise prediction error summary")
    print("=" * 120)

    if display.empty:
        print("(empty)")
        return

    formatters = {}
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            formatters[col] = lambda x: "" if pd.isna(x) else f"{x:.4f}"

    print(display.to_string(index=False, formatters=formatters))


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
            "CSV manifest. Minimal format: rollout_root. "
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
        "--axis",
        type=str,
        default="z",
        choices=["x", "y", "z"],
        help="Axis image to evaluate.",
    )
    parser.add_argument(
        "--pred_source",
        type=str,
        default="raw_mean",
        choices=["raw_mean", "ensemble_image", "auto"],
        help=(
            "Prediction image source. raw_mean averages raw_pred_image/step_*/ensemble_z_*.png. "
            "ensemble_image uses saved *_ensemble_z_axis*.png. auto tries raw_mean first."
        ),
    )
    parser.add_argument(
        "--palette",
        type=str,
        default="simple",
        choices=["simple", "complex", "none"],
        help=(
            "Color ranges for converting images into target masks. "
            "Use simple for Object A/B/C experiments."
        ),
    )
    parser.add_argument(
        "--eval_region",
        type=str,
        default="target_union",
        choices=["all", "oracle_foreground", "target_union", "target_unobserved"],
        help=(
            "Region where RGB prediction error is computed. "
            "target_union is recommended for paper-style internal-part prediction error."
        ),
    )
    parser.add_argument(
        "--background_tolerance",
        type=float,
        default=0.08,
        help="Distance threshold from white background for foreground detection.",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument("--no_plots", action="store_true")

    args = parser.parse_args()

    if args.root is None and args.manifest is None:
        raise ValueError("Specify either --root or --manifest.")

    if args.root is not None and args.manifest is not None:
        raise ValueError("Specify only one of --root or --manifest.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest is not None:
        df = collect_from_manifest(
            manifest_path=args.manifest,
            axis=args.axis,
            pred_source=args.pred_source,
            palette_name=args.palette,
            eval_region=args.eval_region,
            background_tolerance=args.background_tolerance,
        )
    else:
        df = collect_from_root(
            root=args.root,
            axis=args.axis,
            pred_source=args.pred_source,
            palette_name=args.palette,
            eval_region=args.eval_region,
            background_tolerance=args.background_tolerance,
            run_id=args.run_id,
            method=args.method,
            condition=args.condition,
        )

    df = df.sort_values(
        ["run_id", "case", "episode_idx", "step_idx"],
        kind="stable",
    )

    step_summary = build_summary_by_step(df)
    object_summary = build_summary_by_object(step_summary)

    long_path = args.out_dir / "stepwise_prediction_error_long.csv"
    step_path = args.out_dir / "stepwise_prediction_error_by_step.csv"
    object_path = args.out_dir / "stepwise_prediction_error_by_object.csv"

    df.to_csv(long_path, index=False)
    step_summary.to_csv(step_path, index=False)
    object_summary.to_csv(object_path, index=False)

    print(f"[OK] Saved step-wise prediction error long : {long_path}")
    print(f"[OK] Saved summary by step                : {step_path}")
    print(f"[OK] Saved summary by object              : {object_path}")

    if not args.no_plots:
        apply_publication_style()

        plot_metric_by_step_lineplot(
            step_summary,
            args.out_dir,
            metric_mean_col="rgb_mae_mean",
            metric_std_col="rgb_mae_std",
            ylabel="RGB MAE",
            plot_name="rgb_mae_by_step",
        )
        plot_metric_by_step_lineplot(
            step_summary,
            args.out_dir,
            metric_mean_col="target_iou_mean",
            metric_std_col="target_iou_std",
            ylabel="Target IoU",
            plot_name="target_iou_by_step",
        )
        plot_metric_by_step_lineplot(
            step_summary,
            args.out_dir,
            metric_mean_col="target_observed_fraction_mean",
            metric_std_col="target_observed_fraction_std",
            ylabel="Target observed fraction",
            plot_name="target_observed_fraction_by_step",
        )
        plot_observed_vs_error_scatter(df, args.out_dir)

        print(f"[OK] Saved plots under                    : {args.out_dir / 'plots'}")

    print_summary(object_summary)


if __name__ == "__main__":
    main()
