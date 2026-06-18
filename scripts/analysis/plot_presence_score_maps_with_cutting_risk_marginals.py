from __future__ import annotations

import argparse
import csv
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from denoising_diffusion_pytorch.cost.types import AxisCost, AxisDecisionCost
from denoising_diffusion_pytorch.policy.decision.decision_rules import (
    clip_ucb_raw,
    compute_clip_ucb_scores,
)
from plot_presence_score_maps import (
    DEFAULT_COLOR_RANGES,
    MethodSpec,
    ViewSpec,
    apply_crop,
    apply_subplot_spacing,
    build_presence_colormap,
    crop_bounds_from_masks,
    discover_cases,
    load_external_shape_mask,
    ordered_method_labels,
    parse_case_filter,
    parse_optional_int,
    parse_view_specs,
    project_volume,
    read_oracle_score_map,
    read_raw_pred_score_map,
    read_shape_mask,
    render_score_panel,
    resize_mask_to_shape,
    resolve_episode_dir,
    resolve_method_spec_for_case,
    resolve_step_dir,
    select_external_shape_mask_for_view,
)
from plot_safety_margin_decision_maps import (
    load_pickle,
    parse_radii,
    remap_axis_scores_to_presence_coords,
    select_axis_cost_ensemble,
    to_risk,
)


@dataclass(frozen=True)
class MethodRiskSpec(MethodSpec):
    cost_map_log: Path | None = None


@dataclass
class PanelData:
    score: np.ndarray
    mask: np.ndarray | None
    is_ground_truth: bool
    coords: dict[str, np.ndarray]


TargetColorName = str


def parse_manifest(path: Path) -> list[MethodRiskSpec]:
    """Parse the presence-map manifest plus an optional cost_map_log column.

    Required columns are the same as plot_presence_score_maps.py:
      rollout_root[, method|label|name, source, case, episode, step]

    Optional column:
      cost_map_log: explicit path to *_cost_map_logs.pickle. If omitted, this
      script tries episode_dir/<resolved_step>_cost_map_logs.pickle for non-GT
      rows. Relative cost_map_log paths are first interpreted from the current
      working directory, then from the manifest directory.
    """
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    if "rollout_root" not in rows[0]:
        raise ValueError(f"Manifest must contain a rollout_root column. Available columns: {list(rows[0].keys())}")

    specs: list[MethodRiskSpec] = []
    for idx, row in enumerate(rows):
        row_no = idx + 2
        root_text = (row.get("rollout_root") or "").strip()
        if not root_text:
            raise ValueError(f"Empty rollout_root at manifest row {row_no}: {path}")
        label = str(row.get("method") or row.get("label") or row.get("name") or Path(root_text).name).strip()
        if not label:
            raise ValueError(f"Empty method label at manifest row {row_no}: {path}")
        source = str(row.get("source") or "").strip().lower()
        if not source:
            lower = label.lower()
            source = "oracle" if ("ground" in lower or "gt" in lower or "oracle" in lower) else "raw_pred"
        if source not in {"raw_pred", "oracle"}:
            raise ValueError(f"Unsupported source={source!r} at manifest row {row_no}. Use raw_pred or oracle.")

        cost_map_text = (row.get("cost_map_log") or "").strip()
        cost_map_log = None
        if cost_map_text:
            candidate = Path(cost_map_text)
            if not candidate.is_absolute() and not candidate.exists():
                manifest_relative = path.parent / candidate
                candidate = manifest_relative if manifest_relative.exists() else candidate
            cost_map_log = candidate

        specs.append(
            MethodRiskSpec(
                label=label,
                root=Path(root_text),
                source=source,
                case=(row.get("case") or "").strip() or None,
                episode=parse_optional_int(row, "episode", row_no=row_no, manifest_path=path),
                step=parse_optional_int(row, "step", row_no=row_no, manifest_path=path),
                cost_map_log=cost_map_log,
            )
        )
    return specs


def infer_target_from_presence_color(target_color: str) -> TargetColorName:
    for candidate in ("blue", "red", "yellow"):
        if target_color.endswith(candidate) or candidate in target_color:
            return candidate
    raise ValueError(
        f"Could not infer cutting-risk target from --target_color={target_color!r}. "
        "Pass --target explicitly, e.g. --target blue."
    )


def parse_ylim(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--risk_ylim must be given as 'low,high', e.g. 0,1")
    low, high = float(parts[0]), float(parts[1])
    if high <= low:
        raise ValueError(f"Invalid --risk_ylim={text!r}: high must be greater than low.")
    return low, high


def infer_step_from_step_dir(step_dir: Path) -> int:
    match = re.search(r"step_(-?\d+)$", step_dir.name)
    if match is None:
        raise ValueError(f"Could not infer step index from step directory: {step_dir}")
    return int(match.group(1))


def resolve_raw_prediction_step(episode_dir: Path, step: int | None) -> int:
    if step is not None and step >= 0:
        return step
    return infer_step_from_step_dir(resolve_step_dir(episode_dir / "raw_pred_image", None))


def resolve_cost_map_log(spec: MethodRiskSpec, episode_dir: Path, resolved_step: int) -> Path | None:
    if spec.cost_map_log is not None:
        return spec.cost_map_log
    if spec.source == "oracle":
        return None
    return episode_dir / f"{resolved_step}_cost_map_logs.pickle"


def visible_axis_coordinate_maps(
    *,
    volume_shape: tuple[int, int, int],
    projection_axis: str,
    rot90: int,
) -> dict[str, np.ndarray]:
    """Return displayed pixel-to-axis-index maps for a projected 3D volume."""
    x_len, y_len, z_len = volume_shape
    if projection_axis == "x":
        row_axis, col_axis = "y", "z"
        row_len, col_len = y_len, z_len
    elif projection_axis == "y":
        row_axis, col_axis = "x", "z"
        row_len, col_len = x_len, z_len
    elif projection_axis == "z":
        row_axis, col_axis = "x", "y"
        row_len, col_len = x_len, y_len
    else:
        raise ValueError(f"Unknown projection_axis={projection_axis!r}.")

    row_values = np.arange(row_len, dtype=int)[:, None]
    col_values = np.arange(col_len, dtype=int)[None, :]
    coords = {
        row_axis: np.broadcast_to(row_values, (row_len, col_len)).copy(),
        col_axis: np.broadcast_to(col_values, (row_len, col_len)).copy(),
    }
    if rot90:
        coords = {name: np.rot90(values, rot90) for name, values in coords.items()}
    return coords


def apply_crop_to_coords(
    coords: dict[str, np.ndarray],
    bounds: tuple[int, int, int, int] | None,
) -> dict[str, np.ndarray]:
    if bounds is None:
        return coords
    y0, y1, x0, x1 = bounds
    return {name: values[y0:y1, x0:x1] for name, values in coords.items()}


def pad_panel_to_square(panel: PanelData) -> None:
    h, w = panel.score.shape[:2]
    size = max(h, w)
    if h == size and w == size:
        return
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    pad_2d = ((pad_top, pad_bottom), (pad_left, pad_right))
    panel.score = np.pad(panel.score, pad_2d, mode="constant", constant_values=0.0)
    panel.mask = None if panel.mask is None else np.pad(panel.mask, pad_2d, mode="constant", constant_values=False)
    panel.coords = {
        name: np.pad(values, pad_2d, mode="constant", constant_values=-1)
        for name, values in panel.coords.items()
    }


def axis_values(axis_scores: AxisCost, axis_name: str) -> np.ndarray:
    if axis_name == "x":
        return np.asarray(axis_scores.x_axis, dtype=float).reshape(-1)
    if axis_name == "y":
        return np.asarray(axis_scores.y_axis, dtype=float).reshape(-1)
    if axis_name == "z":
        return np.asarray(axis_scores.z_axis, dtype=float).reshape(-1)
    raise ValueError(f"Unknown axis_name={axis_name!r}.")


def collapse_coordinate_by_column(values: np.ndarray) -> np.ndarray | None:
    height, width = values.shape
    profile = np.full(width, -1, dtype=int)
    for col in range(width):
        valid = values[:, col]
        valid = valid[valid >= 0]
        if valid.size == 0:
            continue
        unique = np.unique(valid)
        if unique.size != 1:
            return None
        profile[col] = int(unique[0])
    valid_profile = profile[profile >= 0]
    if valid_profile.size > 1 and np.unique(valid_profile).size > 1:
        return profile
    return None


def collapse_coordinate_by_row(values: np.ndarray) -> np.ndarray | None:
    height, width = values.shape
    profile = np.full(height, -1, dtype=int)
    for row in range(height):
        valid = values[row, :]
        valid = valid[valid >= 0]
        if valid.size == 0:
            continue
        unique = np.unique(valid)
        if unique.size != 1:
            return None
        profile[row] = int(unique[0])
    valid_profile = profile[profile >= 0]
    if valid_profile.size > 1 and np.unique(valid_profile).size > 1:
        return profile
    return None


def infer_display_axis_profiles(coords: dict[str, np.ndarray]) -> tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]:
    """Infer horizontal/vertical risk axis indices after projection, crop, and padding.

    Returns:
      (horizontal_axis_name, horizontal_indices_by_pixel_column),
      (vertical_axis_name, vertical_indices_by_pixel_row)

    Indices equal to -1 denote padded pixels and are plotted as NaN.
    """
    if not coords:
        raise ValueError("No coordinate maps were provided.")
    first = next(iter(coords.values()))
    height, width = first.shape

    horizontal: tuple[str, np.ndarray] | None = None
    vertical: tuple[str, np.ndarray] | None = None
    for name, values in coords.items():
        if values.shape != (height, width):
            raise ValueError("All coordinate maps must have the same shape.")
        col_profile = collapse_coordinate_by_column(values)
        if col_profile is not None:
            horizontal = (name, col_profile)
        row_profile = collapse_coordinate_by_row(values)
        if row_profile is not None:
            vertical = (name, row_profile)

    if horizontal is None or vertical is None:
        raise ValueError(
            "Could not infer horizontal/vertical display axes from coordinate maps. "
            "This can happen if the cropped view is one pixel wide/high."
        )
    return horizontal, vertical


def clipped_profile(values: np.ndarray, *, clip: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return np.clip(values, 0.0, 1.0) if clip else values


def profile_from_axis_risk(
    risk: AxisCost,
    axis_name: str,
    indices_by_pixel: np.ndarray,
    *,
    clip: bool,
) -> np.ndarray:
    axis_profile = axis_values(risk, axis_name)
    out = np.full(indices_by_pixel.shape, np.nan, dtype=float)
    valid = (indices_by_pixel >= 0) & (indices_by_pixel < axis_profile.size)
    out[valid] = axis_profile[indices_by_pixel[valid]]
    return clipped_profile(out, clip=clip)


def compute_cutting_risk_by_radius(
    *,
    cost_ensemble,
    radii: list[int],
    score_mode: str,
    ucb_lb: float,
) -> dict[int, AxisCost]:
    risks: dict[int, AxisCost] = {}
    for radius in radii:
        scores = compute_clip_ucb_scores(
            cost_ensemble=cost_ensemble,
            safety_margin_voxels=radius,
            ucb_beta=1.0,
        )
        decision: AxisDecisionCost | None = None
        if score_mode == "decision":
            decision = clip_ucb_raw(
                cost_ensemble=cost_ensemble,
                ucb_lb=ucb_lb,
                safety_margin_voxels=radius,
            )
        risks[radius] = remap_axis_scores_to_presence_coords(
            to_risk(scores, mode=score_mode, decision=decision),
        )
    return risks


def load_cutting_risks(
    *,
    cost_map_log: Path,
    target: TargetColorName,
    radii: list[int],
    score_mode: str,
    ucb_lb: float,
) -> dict[int, AxisCost]:
    logs = load_pickle(cost_map_log)
    if "cost_ensembles" not in logs:
        raise KeyError(f"cost_map_log does not contain 'cost_ensembles': {cost_map_log}")
    cost_ensemble = select_axis_cost_ensemble(logs["cost_ensembles"], target)  # type: ignore[arg-type]
    return compute_cutting_risk_by_radius(
        cost_ensemble=cost_ensemble,
        radii=radii,
        score_mode=score_mode,
        ucb_lb=ucb_lb,
    )


def fill_base_for_profile(*, risk_ylim: tuple[float, float], risk_threshold: float, mode: str) -> float:
    if mode == "from_zero":
        return risk_ylim[0]
    if mode == "above_threshold":
        return risk_threshold
    raise ValueError(f"Unknown profile_fill_mode={mode!r}. Use above_threshold or from_zero.")


def style_marginal_axis(ax, *, show_ticks: bool, show_axis_labels: bool) -> None:
    ax.patch.set_alpha(0.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if not show_axis_labels:
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.tick_params(length=2.0, width=0.6)


def add_marginal_cutting_risk_profiles(
    *,
    ax_main,
    panel: PanelData,
    risks: dict[int, AxisCost] | None,
    radii: list[int],
    line_colors: list,
    risk_ylim: tuple[float, float],
    clip_risk_for_display: bool,
    profile_height_ratio: float,
    profile_width_ratio: float,
    profile_pad: float,
    profile_linewidth: float,
    profile_alpha: float,
    fill_profile_area: bool,
    profile_fill_alpha: float,
    profile_fill_mode: str,
    risk_threshold: float,
    show_threshold_line: bool,
    show_profile_axis_labels: bool,
    show_profile_ticks: bool,
    label_fontsize: int,
) -> list:
    divider = make_axes_locatable(ax_main)
    ax_top = divider.append_axes(
        "top",
        size=f"{100.0 * profile_height_ratio:.1f}%",
        pad=profile_pad,
        sharex=ax_main,
    )
    ax_right = divider.append_axes(
        "right",
        size=f"{100.0 * profile_width_ratio:.1f}%",
        pad=profile_pad,
        sharey=ax_main,
    )

    height, width = panel.score.shape
    ax_top.set_xlim(-0.5, width - 0.5)
    ax_top.set_ylim(*risk_ylim)
    ax_right.set_xlim(*risk_ylim)
    ax_right.set_ylim(height - 0.5, -0.5)

    if not risks:
        ax_top.set_axis_off()
        ax_right.set_axis_off()
        return [ax_top, ax_right]

    (h_axis_name, h_indices), (v_axis_name, v_indices) = infer_display_axis_profiles(panel.coords)
    x_pixels = np.arange(width)
    y_pixels = np.arange(height)
    fill_base = fill_base_for_profile(
        risk_ylim=risk_ylim,
        risk_threshold=risk_threshold,
        mode=profile_fill_mode,
    )

    if show_threshold_line:
        ax_top.axhline(risk_threshold, color="0.55", linewidth=0.6, linestyle=(0, (3, 2)), alpha=0.7, zorder=1)
        ax_right.axvline(risk_threshold, color="0.55", linewidth=0.6, linestyle=(0, (3, 2)), alpha=0.7, zorder=1)

    for color, radius in zip(line_colors, radii):
        risk = risks.get(radius)
        if risk is None:
            continue
        h_profile = profile_from_axis_risk(risk, h_axis_name, h_indices, clip=clip_risk_for_display)
        v_profile = profile_from_axis_risk(risk, v_axis_name, v_indices, clip=clip_risk_for_display)
        h_valid = ~np.isnan(h_profile)
        v_valid = ~np.isnan(v_profile)
        if fill_profile_area:
            h_where = h_valid & (h_profile >= fill_base if profile_fill_mode == "above_threshold" else np.ones_like(h_profile, dtype=bool))
            v_where = v_valid & (v_profile >= fill_base if profile_fill_mode == "above_threshold" else np.ones_like(v_profile, dtype=bool))
            ax_top.fill_between(
                x_pixels,
                fill_base,
                h_profile,
                where=h_where,
                color=color,
                alpha=profile_fill_alpha,
                linewidth=0,
                zorder=2,
            )
            ax_right.fill_betweenx(
                y_pixels,
                fill_base,
                v_profile,
                where=v_where,
                color=color,
                alpha=profile_fill_alpha,
                linewidth=0,
                zorder=2,
            )
        ax_top.plot(x_pixels, h_profile, linewidth=profile_linewidth, alpha=profile_alpha, color=color, label=f"r={radius}", zorder=3)
        ax_right.plot(v_profile, y_pixels, linewidth=profile_linewidth, alpha=profile_alpha, color=color, label=f"r={radius}", zorder=3)

    if show_profile_ticks:
        ax_top.set_yticks([risk_ylim[0], risk_threshold, risk_ylim[1]])
        ax_top.tick_params(axis="y", labelsize=max(label_fontsize - 2, 6))
        ax_right.set_xticks([risk_ylim[0], risk_threshold, risk_ylim[1]])
        ax_right.tick_params(axis="x", labelsize=max(label_fontsize - 2, 6))
    if show_profile_axis_labels:
        ax_top.set_ylabel(f"s({h_axis_name})", fontsize=max(label_fontsize - 2, 6))
        ax_right.set_xlabel(f"s({v_axis_name})", fontsize=max(label_fontsize - 2, 6))

    style_marginal_axis(ax_top, show_ticks=show_profile_ticks, show_axis_labels=show_profile_axis_labels)
    style_marginal_axis(ax_right, show_ticks=show_profile_ticks, show_axis_labels=show_profile_axis_labels)
    return [ax_top, ax_right]


def add_outside_colorbar(fig, axes_for_layout: list, presence_cmap, args) -> None:
    fig.tight_layout(rect=[0.0, 0.0, args.colorbar_layout_right, 1.0])
    apply_subplot_spacing(fig, args)
    positions = [ax.get_position() for ax in axes_for_layout]
    right_edge = max(pos.x1 for pos in positions)
    bottom_edge = min(pos.y0 for pos in positions)
    top_edge = max(pos.y1 for pos in positions)
    total_height = top_edge - bottom_edge
    cbar_height = total_height * args.colorbar_shrink
    cbar_bottom = bottom_edge + (total_height - cbar_height) / 2.0
    cbar_left = min(right_edge + args.colorbar_pad, 0.98 - args.colorbar_width)
    cax = fig.add_axes([cbar_left, cbar_bottom, args.colorbar_width, cbar_height])
    sm = plt.cm.ScalarMappable(cmap=presence_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Presence score", fontsize=args.label_fontsize)
    cbar.ax.tick_params(labelsize=max(args.label_fontsize - 1, 6))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot presence-score heatmaps and add translucent marginal profiles "
            "of axis-wise cutting risk s from *_cost_map_logs.pickle."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--case_filter", type=str, default=None)
    parser.add_argument("--episode", type=int, default=0, help="Fallback episode index used when a manifest row has no episode column/value.")
    parser.add_argument("--step", type=int, default=-1, help="Fallback step index used when a manifest row has no step column/value. Negative means the last available step.")
    parser.add_argument("--target_color", type=str, default="complex_blue", choices=sorted(DEFAULT_COLOR_RANGES))
    parser.add_argument("--target", type=str, default=None, choices=["blue", "red", "yellow"], help="Target color key in cost_ensembles. Defaults to inference from --target_color.")
    parser.add_argument("--side_length", type=int, default=None)
    parser.add_argument("--view_specs", type=str, default=None)
    parser.add_argument("--presence_cmap", type=str, default="jet_bright")
    parser.add_argument("--background_mode", type=str, default="white", choices=["white", "low_score", "light_gray"])
    parser.add_argument("--ground_truth_background_mode", type=str, default="white", choices=["white", "low_score", "light_gray"])
    parser.add_argument("--ground_truth_style", type=str, default="structure", choices=["structure", "target_only"])
    parser.add_argument("--shape_mask_side_image", type=Path, default=None, help="Optional 2D silhouette mask for the side view. Bright pixels are treated as the object body.")
    parser.add_argument("--shape_mask_top_image", type=Path, default=None, help="Optional 2D silhouette mask for the top view. Bright pixels are treated as the object body.")
    parser.add_argument("--crop_scope", type=str, default="case_view", choices=["case_view", "panel"])
    parser.add_argument("--no_square_panels", action="store_true")
    parser.add_argument("--panel_frame", action="store_true")
    parser.add_argument("--crop_padding", type=int, default=2)
    parser.add_argument("--no_auto_crop", action="store_true")

    parser.add_argument("--radii", type=str, default="0", help="Comma-separated safety margin radii for cutting risk s, e.g. 0,1,2.")
    parser.add_argument("--score_mode", type=str, default="ucb", choices=["ucb", "decision"], help="Plot continuous UCB scores or thresholded binary decision risks in marginal profiles.")
    parser.add_argument("--ucb_lb", type=float, default=0.5, help="Decision threshold used when --score_mode decision.")
    parser.add_argument("--risk_ylim", type=str, default="0,1", help="Range for marginal risk profiles as 'low,high'.")
    parser.add_argument("--no_clip_risk_for_display", action="store_true", help="Do not clip UCB risk profiles to [0, 1] before plotting.")
    parser.add_argument("--risk_line_cmap", type=str, default="magma")
    parser.add_argument("--risk_threshold", type=float, default=0.5, help="Risk threshold guide line shown in marginal profiles.")
    parser.add_argument("--hide_threshold_line", action="store_true")
    parser.add_argument("--profile_linewidth", type=float, default=0.95)
    parser.add_argument("--profile_alpha", type=float, default=0.82)
    parser.add_argument("--fill_profile_area", dest="fill_profile_area", action="store_true", default=True, help="Fill the marginal cutting-risk profile area with a translucent color. Enabled by default.")
    parser.add_argument("--no_fill_profile_area", dest="fill_profile_area", action="store_false")
    parser.add_argument("--profile_fill_alpha", type=float, default=0.22)
    parser.add_argument("--profile_fill_mode", type=str, default="from_zero", choices=["above_threshold", "from_zero"])
    parser.add_argument("--profile_height_ratio", type=float, default=0.20)
    parser.add_argument("--profile_width_ratio", type=float, default=0.20)
    parser.add_argument("--profile_pad", type=float, default=0.015)
    parser.add_argument("--show_profile_axis_labels", action="store_true")
    parser.add_argument("--show_profile_ticks", action="store_true")
    parser.add_argument("--require_risk", action="store_true", help="Raise an error when a cost_map_log cannot be found. By default, missing risk profiles are skipped.")

    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title_fontsize", type=int, default=11)
    parser.add_argument("--label_fontsize", type=int, default=10)
    parser.add_argument("--subplot_wspace", type=float, default=None, help="Optional horizontal spacing between method columns passed to fig.subplots_adjust(wspace=...).")
    parser.add_argument("--subplot_hspace", type=float, default=None, help="Optional vertical spacing between view/case rows passed to fig.subplots_adjust(hspace=...).")
    parser.add_argument("--add_colorbar", action="store_true", help="Show a colorbar. Kept for backward compatibility.")
    parser.add_argument("--show_colorbar", action="store_true", help="Show a colorbar outside the panel grid.")
    parser.add_argument("--colorbar_layout_right", type=float, default=0.84)
    parser.add_argument("--colorbar_pad", type=float, default=0.025)
    parser.add_argument("--colorbar_width", type=float, default=0.018)
    parser.add_argument("--colorbar_shrink", type=float, default=0.9)
    args = parser.parse_args()

    method_specs = parse_manifest(args.manifest)
    method_labels = ordered_method_labels(method_specs)
    cases = discover_cases(method_specs, parse_case_filter(args.case_filter))
    view_specs = parse_view_specs(args.view_specs)
    presence_cmap = build_presence_colormap(args.presence_cmap)
    target = args.target if args.target is not None else infer_target_from_presence_color(args.target_color)
    radii = parse_radii(args.radii)
    risk_ylim = parse_ylim(args.risk_ylim)
    line_cmap = plt.get_cmap(args.risk_line_cmap)
    line_positions = np.linspace(0.18, 0.82, max(len(radii), 1))
    line_colors = [line_cmap(pos) for pos in line_positions]

    external_side_mask = load_external_shape_mask(args.shape_mask_side_image) if args.shape_mask_side_image is not None else None
    external_top_mask = load_external_shape_mask(args.shape_mask_top_image) if args.shape_mask_top_image is not None else None

    panel_data: dict[tuple[int, int, int], PanelData] = {}
    risk_data: dict[tuple[int, int], dict[int, AxisCost] | None] = {}
    for case_idx, case in enumerate(cases):
        for col_idx, label in enumerate(method_labels):
            spec = resolve_method_spec_for_case(method_specs, label=label, case=case)
            if not isinstance(spec, MethodRiskSpec):
                raise TypeError("Expected MethodRiskSpec from this script's parse_manifest().")
            episode = spec.episode if spec.episode is not None else args.episode
            requested_step = spec.step if spec.step is not None else args.step
            episode_dir = resolve_episode_dir(spec.root, case, episode)
            shape_mask_volume = read_shape_mask(episode_dir, side_length=args.side_length)

            if spec.source == "oracle":
                score_volume = read_oracle_score_map(episode_dir, target_color=args.target_color, side_length=args.side_length)
                is_gt = True
                resolved_step = requested_step if requested_step >= 0 else 0
            else:
                resolved_step = resolve_raw_prediction_step(episode_dir, requested_step)
                score_volume = read_raw_pred_score_map(
                    episode_dir,
                    step=resolved_step,
                    target_color=args.target_color,
                    side_length=args.side_length,
                )
                is_gt = False

            cost_map_log = resolve_cost_map_log(spec, episode_dir, resolved_step)
            risks: dict[int, AxisCost] | None = None
            if cost_map_log is not None:
                if cost_map_log.exists():
                    risks = load_cutting_risks(
                        cost_map_log=cost_map_log,
                        target=target,
                        radii=radii,
                        score_mode=args.score_mode,
                        ucb_lb=args.ucb_lb,
                    )
                else:
                    message = f"cost_map_log was not found for method={label!r}, case={case!r}: {cost_map_log}"
                    if args.require_risk:
                        raise FileNotFoundError(message)
                    warnings.warn(message)
            elif args.require_risk:
                raise FileNotFoundError(f"No cost_map_log was specified or inferable for method={label!r}, case={case!r}.")
            risk_data[(case_idx, col_idx)] = risks

            volume_shape = tuple(int(v) for v in score_volume.shape[:3])
            for view_idx, view in enumerate(view_specs):
                score = project_volume(score_volume, view.projection_axis, view.rot90)
                coords = visible_axis_coordinate_maps(
                    volume_shape=volume_shape,
                    projection_axis=view.projection_axis,
                    rot90=view.rot90,
                )
                external_mask = select_external_shape_mask_for_view(
                    view,
                    side_mask=external_side_mask,
                    top_mask=external_top_mask,
                )
                if external_mask is not None:
                    mask = resize_mask_to_shape(external_mask, score.shape)
                else:
                    mask = None if shape_mask_volume is None else project_volume(shape_mask_volume.astype(float), view.projection_axis, view.rot90) > 0
                panel_data[(case_idx, col_idx, view_idx)] = PanelData(score, mask, is_gt, coords)

    if not args.no_auto_crop:
        for case_idx, _case in enumerate(cases):
            for view_idx, _view in enumerate(view_specs):
                if args.crop_scope == "case_view":
                    bounds = crop_bounds_from_masks(
                        [panel_data[(case_idx, col_idx, view_idx)].mask for col_idx in range(len(method_labels))],
                        args.crop_padding,
                    )
                    for col_idx in range(len(method_labels)):
                        panel = panel_data[(case_idx, col_idx, view_idx)]
                        panel.score, panel.mask = apply_crop(panel.score, panel.mask, bounds)
                        panel.coords = apply_crop_to_coords(panel.coords, bounds)
                else:
                    for col_idx in range(len(method_labels)):
                        panel = panel_data[(case_idx, col_idx, view_idx)]
                        bounds = crop_bounds_from_masks([panel.mask], args.crop_padding)
                        panel.score, panel.mask = apply_crop(panel.score, panel.mask, bounds)
                        panel.coords = apply_crop_to_coords(panel.coords, bounds)

    if not args.no_square_panels:
        for panel in panel_data.values():
            pad_panel_to_square(panel)

    n_rows = len(cases) * len(view_specs)
    n_cols = len(method_labels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(2.35 * n_cols, 5.8), max(1.75 * n_rows, 3.2)),
        squeeze=False,
    )
    marginal_axes: list = []
    for col_idx, label in enumerate(method_labels):
        axes[0][col_idx].set_title(label, fontsize=args.title_fontsize, pad=4)
    for case_idx, case in enumerate(cases):
        for col_idx, _label in enumerate(method_labels):
            risks = risk_data[(case_idx, col_idx)]
            for view_idx, view in enumerate(view_specs):
                row_idx = case_idx * len(view_specs) + view_idx
                ax = axes[row_idx][col_idx]
                panel = panel_data[(case_idx, col_idx, view_idx)]
                render_score_panel(
                    ax,
                    panel.score,
                    panel.mask,
                    ground_truth=panel.is_ground_truth,
                    presence_cmap=presence_cmap,
                    background_mode=args.background_mode,
                    ground_truth_background_mode=args.ground_truth_background_mode,
                    ground_truth_style=args.ground_truth_style,
                    panel_frame=args.panel_frame,
                )
                marginal_axes.extend(
                    add_marginal_cutting_risk_profiles(
                        ax_main=ax,
                        panel=panel,
                        risks=risks,
                        radii=radii,
                        line_colors=line_colors,
                        risk_ylim=risk_ylim,
                        clip_risk_for_display=not args.no_clip_risk_for_display,
                        profile_height_ratio=args.profile_height_ratio,
                        profile_width_ratio=args.profile_width_ratio,
                        profile_pad=args.profile_pad,
                        profile_linewidth=args.profile_linewidth,
                        profile_alpha=args.profile_alpha,
                        fill_profile_area=args.fill_profile_area,
                        profile_fill_alpha=args.profile_fill_alpha,
                        profile_fill_mode=args.profile_fill_mode,
                        risk_threshold=args.risk_threshold,
                        show_threshold_line=not args.hide_threshold_line,
                        show_profile_axis_labels=args.show_profile_axis_labels,
                        show_profile_ticks=args.show_profile_ticks,
                        label_fontsize=args.label_fontsize,
                    )
                )
                if col_idx == 0:
                    ax.set_ylabel(f"{case}\n{view.label}" if view_idx == 0 else view.label, fontsize=args.label_fontsize, rotation=90, labelpad=8)

    if args.add_colorbar or args.show_colorbar:
        add_outside_colorbar(fig, list(axes.ravel()) + marginal_axes, presence_cmap, args)
    else:
        fig.tight_layout()
        apply_subplot_spacing(fig, args)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[OK] Saved presence-score figure with marginal cutting-risk profiles: {args.out_path}")
    print(f"[INFO] cutting-risk target={target}, radii={radii}, score_mode={args.score_mode}")


if __name__ == "__main__":
    main()


'''
# Example: complex Object D with translucent marginal cutting-risk s profiles
python scripts/analysis/plot_presence_score_maps_with_cutting_risk_marginals.py \
  --manifest analysis/revise/presence_frequency_maps/object_DEF/manifest_complex_DEF.csv \
  --out_path analysis/revise/presence_frequency_maps/object_DEF/complex_presence_frequency_maps_D_with_risk_marginals.pdf \
  --case_filter Object_D \
  --target_color complex_blue \
  --target blue \
  --side_length 49 \
  --episode 1 \
  --step 6 \
  --radii 0,1,2 \
  --score_mode ucb \
  --presence_cmap jet_bright \
  --background_mode white \
  --ground_truth_style structure \
  --ground_truth_background_mode white \
  --view_specs "Top view:x:-1,Side view:z:2" \
  --shape_mask_side_image sheetsander_silhouette/sheetsander_side_silhouette.png \
  --shape_mask_top_image sheetsander_silhouette/sheetsander_top_silhouette.png \
  --subplot_wspace 0.01 \
  --subplot_hspace 0.01 \
  --show_colorbar
'''
