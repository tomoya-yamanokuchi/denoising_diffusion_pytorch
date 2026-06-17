from __future__ import annotations

import argparse
import re
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
    ViewSpec,
    apply_crop,
    build_presence_colormap,
    crop_bounds_from_masks,
    load_external_shape_mask,
    parse_view_specs,
    project_volume,
    read_raw_pred_score_map,
    read_shape_mask,
    resize_mask_to_shape,
    select_external_shape_mask_for_view,
)
from plot_safety_margin_decision_maps import (
    load_pickle,
    parse_radii,
    remap_axis_scores_to_presence_coords,
    select_axis_cost_ensemble,
    to_risk,
)


@dataclass
class ViewPanel:
    view: ViewSpec
    score: np.ndarray
    mask: np.ndarray | None
    coords: dict[str, np.ndarray]


def infer_step_from_cost_map_log(path: Path) -> int | None:
    match = re.match(r"(-?\d+)_cost_map_logs\.pickle$", path.name)
    return None if match is None else int(match.group(1))


def visible_axis_coordinate_maps(
    *,
    volume_shape: tuple[int, int, int],
    projection_axis: str,
    rot90: int,
) -> dict[str, np.ndarray]:
    """Return displayed pixel-to-axis-index maps for a projected 3D volume.

    The returned arrays have the same shape and orientation as the displayed
    presence map produced by project_volume(...). They are used to align the
    marginal 1D cutting-risk profiles with the horizontal and vertical axes of
    the displayed heatmap.
    """
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


def axis_values(axis_scores: AxisCost, axis_name: str) -> np.ndarray:
    if axis_name == "x":
        return np.asarray(axis_scores.x_axis, dtype=float).reshape(-1)
    if axis_name == "y":
        return np.asarray(axis_scores.y_axis, dtype=float).reshape(-1)
    if axis_name == "z":
        return np.asarray(axis_scores.z_axis, dtype=float).reshape(-1)
    raise ValueError(f"Unknown axis_name={axis_name!r}.")


def infer_display_axis_profiles(coords: dict[str, np.ndarray]) -> tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]:
    """Find the axis/index sequence along heatmap columns and rows.

    Returns:
      (horizontal_axis_name, horizontal_indices),
      (vertical_axis_name, vertical_indices)
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
        if width > 1 and np.all(values == values[:1, :]) and np.unique(values[0, :]).size > 1:
            horizontal = (name, values[0, :].astype(int))
        if height > 1 and np.all(values == values[:, :1]) and np.unique(values[:, 0]).size > 1:
            vertical = (name, values[:, 0].astype(int))

    if horizontal is None or vertical is None:
        raise ValueError(
            "Could not infer horizontal/vertical display axes from the coordinate maps. "
            "This can happen if the view was cropped to a single pixel along one axis."
        )
    return horizontal, vertical


def make_presence_rgb(score: np.ndarray, mask: np.ndarray | None, *, presence_cmap, background_mode: str) -> np.ndarray:
    shape_mask = np.ones_like(score, dtype=bool) if mask is None else mask.astype(bool)
    colored = presence_cmap(np.clip(score, 0.0, 1.0))[..., :3]
    if background_mode == "low_score":
        rgb = np.broadcast_to(np.asarray(presence_cmap(0.0)[:3], dtype=float), (*score.shape, 3)).copy()
    elif background_mode == "light_gray":
        rgb = np.ones((*score.shape, 3), dtype=float)
        rgb[~shape_mask] = np.asarray([0.92, 0.92, 0.92])
    elif background_mode == "white":
        rgb = np.ones((*score.shape, 3), dtype=float)
    else:
        raise ValueError(f"Unknown background_mode={background_mode!r}")
    rgb[shape_mask] = colored[shape_mask]
    return rgb


def prepare_view_panels(
    *,
    score_volume: np.ndarray,
    view_specs: list[ViewSpec],
    shape_mask_volume: np.ndarray | None,
    shape_mask_side: np.ndarray | None,
    shape_mask_top: np.ndarray | None,
    auto_crop: bool,
    crop_padding: int,
    crop_score_threshold: float,
) -> list[ViewPanel]:
    panels: list[ViewPanel] = []
    volume_shape = tuple(int(v) for v in score_volume.shape[:3])
    if len(volume_shape) != 3:
        raise ValueError(f"score_volume must be 3D. Got shape={score_volume.shape}.")

    for view in view_specs:
        score = project_volume(score_volume, view.projection_axis, view.rot90)
        coords = visible_axis_coordinate_maps(
            volume_shape=volume_shape,
            projection_axis=view.projection_axis,
            rot90=view.rot90,
        )
        external_mask = select_external_shape_mask_for_view(
            view,
            side_mask=shape_mask_side,
            top_mask=shape_mask_top,
        )
        if external_mask is not None:
            mask = resize_mask_to_shape(external_mask, score.shape)
        elif shape_mask_volume is not None:
            mask = project_volume(shape_mask_volume.astype(float), view.projection_axis, view.rot90) > 0
        else:
            mask = None

        bounds = None
        if auto_crop:
            crop_mask = mask if mask is not None else (score > crop_score_threshold)
            bounds = crop_bounds_from_masks([crop_mask], crop_padding)
            score, mask = apply_crop(score, mask, bounds)
            coords = apply_crop_to_coords(coords, bounds)

        panels.append(ViewPanel(view=view, score=score, mask=mask, coords=coords))
    return panels


def compute_risk_by_radius(
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


def clipped_profile(values: np.ndarray, *, clip: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return np.clip(values, 0.0, 1.0) if clip else values


def style_profile_axis(ax, *, show_ticks: bool, show_axis_labels: bool) -> None:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color("0.25")
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if not show_axis_labels:
        ax.set_xlabel("")
        ax.set_ylabel("")


def fill_base_for_profile(*, risk_ylim: tuple[float, float], risk_threshold: float, mode: str) -> float:
    if mode == "from_zero":
        return risk_ylim[0]
    if mode == "above_threshold":
        return risk_threshold
    raise ValueError(f"Unknown profile_fill_mode={mode!r}. Use above_threshold or from_zero.")


def plot_joint_profiles(
    *,
    panels: list[ViewPanel],
    risks: dict[int, AxisCost],
    radii: list[int],
    out_path: Path,
    presence_cmap,
    background_mode: str,
    risk_ylim: tuple[float, float],
    clip_risk_for_display: bool,
    risk_line_cmap: str,
    show_legend: bool,
    show_presence_colorbar: bool,
    dpi: int,
    title_fontsize: int,
    label_fontsize: int,
    profile_height_ratio: float,
    profile_width_ratio: float,
    view_hspace: float,
    panel_wspace: float,
    profile_linewidth: float,
    profile_alpha: float,
    show_threshold_line: bool,
    risk_threshold: float,
    show_profile_axis_labels: bool,
    show_profile_ticks: bool,
    profile_title: str,
    legend_mode: str,
    legend_y: float,
    fill_profile_area: bool,
    profile_fill_alpha: float,
    profile_fill_mode: str,
) -> None:
    n_views = len(panels)
    if n_views == 0:
        raise ValueError("No view panels to plot.")
    if legend_mode not in {"figure", "axis", "none"}:
        raise ValueError(f"Unknown legend_mode={legend_mode!r}. Use figure, axis, or none.")
    fill_base = fill_base_for_profile(
        risk_ylim=risk_ylim,
        risk_threshold=risk_threshold,
        mode=profile_fill_mode,
    )

    fig_width = 5.2
    fig_height = max(2.35 * n_views, 3.0)
    fig, main_axes_array = plt.subplots(n_views, 1, figsize=(fig_width, fig_height), squeeze=False)
    main_axes = [main_axes_array[i, 0] for i in range(n_views)]
    if n_views > 1:
        fig.subplots_adjust(hspace=view_hspace)

    line_cmap = plt.get_cmap(risk_line_cmap)
    line_positions = np.linspace(0.18, 0.82, max(len(radii), 1))
    line_colors = [line_cmap(pos) for pos in line_positions]

    first_top_ax = None
    main_axes_for_colorbar = []
    for view_idx, (panel, ax_main) in enumerate(zip(panels, main_axes)):
        divider = make_axes_locatable(ax_main)
        ax_top = divider.append_axes(
            "top",
            size=f"{100.0 * profile_height_ratio:.1f}%",
            pad=panel_wspace,
            sharex=ax_main,
        )
        ax_right = divider.append_axes(
            "right",
            size=f"{100.0 * profile_width_ratio:.1f}%",
            pad=panel_wspace,
            sharey=ax_main,
        )
        if first_top_ax is None:
            first_top_ax = ax_top
        main_axes_for_colorbar.append(ax_main)

        height, width = panel.score.shape
        rgb = make_presence_rgb(panel.score, panel.mask, presence_cmap=presence_cmap, background_mode=background_mode)
        ax_main.imshow(rgb, interpolation="nearest", origin="upper")
        ax_main.set_xlim(-0.5, width - 0.5)
        ax_main.set_ylim(height - 0.5, -0.5)
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        ax_main.set_ylabel(panel.view.label, fontsize=label_fontsize, rotation=90, labelpad=8)
        for spine in ax_main.spines.values():
            spine.set_visible(False)

        (h_axis_name, h_indices), (v_axis_name, v_indices) = infer_display_axis_profiles(panel.coords)
        x_pixels = np.arange(width)
        y_pixels = np.arange(height)

        if show_threshold_line:
            ax_top.axhline(risk_threshold, color="0.55", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
            ax_right.axvline(risk_threshold, color="0.55", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)

        for color, radius in zip(line_colors, radii):
            risk = risks[radius]
            h_profile = clipped_profile(axis_values(risk, h_axis_name)[h_indices], clip=clip_risk_for_display)
            v_profile = clipped_profile(axis_values(risk, v_axis_name)[v_indices], clip=clip_risk_for_display)
            label = f"r={radius}"
            if fill_profile_area:
                h_where = h_profile >= fill_base if profile_fill_mode == "above_threshold" else np.ones_like(h_profile, dtype=bool)
                v_where = v_profile >= fill_base if profile_fill_mode == "above_threshold" else np.ones_like(v_profile, dtype=bool)
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
            ax_top.plot(x_pixels, h_profile, linewidth=profile_linewidth, alpha=profile_alpha, color=color, label=label, zorder=3)
            ax_right.plot(v_profile, y_pixels, linewidth=profile_linewidth, alpha=profile_alpha, color=color, label=label, zorder=3)

        ax_top.set_xlim(-0.5, width - 0.5)
        ax_top.set_ylim(*risk_ylim)
        ax_top.set_xticks([])
        if show_profile_ticks:
            ax_top.set_yticks([risk_ylim[0], risk_threshold, risk_ylim[1]])
        ax_top.tick_params(axis="y", labelsize=max(label_fontsize - 2, 6), length=2.0, width=0.6)
        if show_profile_axis_labels:
            ax_top.set_ylabel(f"{h_axis_name}-risk", fontsize=max(label_fontsize - 2, 6))
        if view_idx == 0 and profile_title:
            ax_top.set_title(profile_title, fontsize=title_fontsize, pad=4)
        style_profile_axis(ax_top, show_ticks=show_profile_ticks, show_axis_labels=show_profile_axis_labels)

        ax_right.set_xlim(*risk_ylim)
        ax_right.set_ylim(height - 0.5, -0.5)
        ax_right.set_yticks([])
        if show_profile_ticks:
            ax_right.set_xticks([risk_ylim[0], risk_threshold, risk_ylim[1]])
            ax_right.tick_params(axis="x", labelsize=max(label_fontsize - 2, 6), length=2.0, width=0.6)
        if show_profile_axis_labels:
            ax_right.set_xlabel(f"{v_axis_name}-risk", fontsize=max(label_fontsize - 2, 6))
        style_profile_axis(ax_right, show_ticks=show_profile_ticks, show_axis_labels=show_profile_axis_labels)

    if show_legend and first_top_ax is not None and legend_mode != "none":
        handles, labels = first_top_ax.get_legend_handles_labels()
        if legend_mode == "figure":
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, legend_y),
                fontsize=max(label_fontsize - 2, 6),
                frameon=False,
                ncol=len(radii),
                handlelength=1.8,
                columnspacing=1.0,
            )
        else:
            first_top_ax.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                fontsize=max(label_fontsize - 2, 6),
                frameon=False,
                ncol=len(radii),
                handlelength=1.8,
                columnspacing=1.0,
            )

    if show_presence_colorbar and main_axes_for_colorbar:
        sm = plt.cm.ScalarMappable(cmap=presence_cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=main_axes_for_colorbar, fraction=0.025, pad=0.12)
        cbar.set_label("Presence score", fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=max(label_fontsize - 1, 6))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def parse_ylim(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--risk_ylim must be given as 'low,high', e.g. 0,1")
    low, high = float(parts[0]), float(parts[1])
    if high <= low:
        raise ValueError(f"Invalid --risk_ylim={text!r}: high must be greater than low.")
    return low, high


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a voxel-wise presence heatmap with marginal axis-wise cutting-risk "
            "profiles computed from clip_ucb_raw pre-threshold scores."
        )
    )
    parser.add_argument("--cost_map_log", type=Path, required=True, help="Path to *_cost_map_logs.pickle. Used to infer episode directory and step.")
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--raw_pred_episode_dir", type=Path, default=None, help="Optional episode directory containing raw_pred_image. Defaults to cost_map_log.parent.")
    parser.add_argument("--step", type=int, default=None, help="Optional raw prediction step. Defaults to the prefix of *_cost_map_logs.pickle.")
    parser.add_argument("--target_color", type=str, default="simple_blue", choices=sorted(DEFAULT_COLOR_RANGES), help="Target color used to compute the voxel-wise presence heatmap from raw_pred_image samples.")
    parser.add_argument("--target", type=str, default="blue", choices=["blue", "red", "yellow"], help="Target color key in cost_ensembles used for axis-wise cutting-risk profiles.")
    parser.add_argument("--side_length", type=int, default=None)
    parser.add_argument("--radii", type=str, default="0,1,2", help="Comma-separated safety margin radii, e.g. 0,1,2.")
    parser.add_argument("--score_mode", type=str, default="ucb", choices=["ucb", "decision"], help="Plot continuous UCB scores or thresholded binary decision risks in the marginal profiles.")
    parser.add_argument("--ucb_lb", type=float, default=0.5, help="Decision threshold used when --score_mode decision.")
    parser.add_argument("--view_specs", type=str, default=None, help="View specs passed to the central presence heatmap, e.g. 'Top view:z:2,Side view:x:-1'.")
    parser.add_argument("--presence_cmap", type=str, default="jet_bright")
    parser.add_argument("--background_mode", type=str, default="low_score", choices=["white", "low_score", "light_gray"])
    parser.add_argument("--shape_mask_side_image", type=Path, default=None, help="Optional 2D silhouette mask for side-view display/cropping.")
    parser.add_argument("--shape_mask_top_image", type=Path, default=None, help="Optional 2D silhouette mask for top-view display/cropping.")
    parser.add_argument("--auto_crop", dest="auto_crop", action="store_true", default=True, help="Crop each view to the external/oracle silhouette mask or nonzero presence region before plotting. Enabled by default to match presence-map scripts.")
    parser.add_argument("--no_auto_crop", dest="auto_crop", action="store_false", help="Disable automatic cropping and show the full voxel grid.")
    parser.add_argument("--crop_padding", type=int, default=2)
    parser.add_argument("--crop_score_threshold", type=float, default=1e-6)
    parser.add_argument("--risk_ylim", type=str, default="0,1", help="Y/x range for marginal risk profiles as 'low,high'.")
    parser.add_argument("--no_clip_risk_for_display", action="store_true", help="Do not clip UCB risk profiles to [0, 1] before plotting.")
    parser.add_argument("--risk_line_cmap", type=str, default="magma")
    parser.add_argument("--show_legend", action="store_true")
    parser.add_argument("--legend_mode", type=str, default="figure", choices=["figure", "axis", "none"], help="Place the r legend outside the profiles, inside the top profile, or hide it.")
    parser.add_argument("--legend_y", type=float, default=1.02, help="Figure-level legend y position when --legend_mode figure.")
    parser.add_argument("--show_presence_colorbar", action="store_true")
    parser.add_argument("--hide_threshold_line", action="store_true", help="Hide the decision-threshold guide line in the marginal profiles.")
    parser.add_argument("--risk_threshold", type=float, default=0.5, help="Risk threshold guide line shown in marginal profiles.")
    parser.add_argument("--profile_linewidth", type=float, default=1.15)
    parser.add_argument("--profile_alpha", type=float, default=0.92)
    parser.add_argument("--fill_profile_area", action="store_true", help="Fill the marginal cutting-risk profile area with a translucent color.")
    parser.add_argument("--profile_fill_alpha", type=float, default=0.18)
    parser.add_argument("--profile_fill_mode", type=str, default="above_threshold", choices=["above_threshold", "from_zero"], help="Fill only risk above the threshold or fill from the lower y/x limit.")
    parser.add_argument("--show_profile_axis_labels", action="store_true", help="Show marginal axis labels such as x-risk and y-risk.")
    parser.add_argument("--show_profile_ticks", action="store_true", help="Show compact ticks on marginal profile axes.")
    parser.add_argument("--profile_title", type=str, default="Axis-wise cutting risk")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title_fontsize", type=int, default=9)
    parser.add_argument("--label_fontsize", type=int, default=9)
    parser.add_argument("--profile_height_ratio", type=float, default=0.22)
    parser.add_argument("--profile_width_ratio", type=float, default=0.22)
    parser.add_argument("--view_hspace", type=float, default=0.18)
    parser.add_argument("--panel_wspace", type=float, default=0.04)
    args = parser.parse_args()

    episode_dir = args.raw_pred_episode_dir if args.raw_pred_episode_dir is not None else args.cost_map_log.parent
    step = args.step if args.step is not None else infer_step_from_cost_map_log(args.cost_map_log)
    if step is None:
        raise ValueError(
            "Could not infer step from cost_map_log filename. "
            "Use a name like 0_cost_map_logs.pickle or pass --step explicitly."
        )

    radii = parse_radii(args.radii)
    view_specs = parse_view_specs(args.view_specs)
    presence_cmap = build_presence_colormap(args.presence_cmap)
    risk_ylim = parse_ylim(args.risk_ylim)

    score_volume = read_raw_pred_score_map(
        episode_dir,
        step=step,
        target_color=args.target_color,
        side_length=args.side_length,
    )
    shape_mask_volume = read_shape_mask(episode_dir, side_length=args.side_length)

    side_mask = load_external_shape_mask(args.shape_mask_side_image) if args.shape_mask_side_image is not None else None
    top_mask = load_external_shape_mask(args.shape_mask_top_image) if args.shape_mask_top_image is not None else None
    panels = prepare_view_panels(
        score_volume=score_volume,
        view_specs=view_specs,
        shape_mask_volume=shape_mask_volume,
        shape_mask_side=side_mask,
        shape_mask_top=top_mask,
        auto_crop=args.auto_crop,
        crop_padding=args.crop_padding,
        crop_score_threshold=args.crop_score_threshold,
    )

    logs = load_pickle(args.cost_map_log)
    if "cost_ensembles" not in logs:
        raise KeyError(f"cost_map_log does not contain 'cost_ensembles': {args.cost_map_log}")
    cost_ensemble = select_axis_cost_ensemble(logs["cost_ensembles"], args.target)  # type: ignore[arg-type]
    risks = compute_risk_by_radius(
        cost_ensemble=cost_ensemble,
        radii=radii,
        score_mode=args.score_mode,
        ucb_lb=args.ucb_lb,
    )

    plot_joint_profiles(
        panels=panels,
        risks=risks,
        radii=radii,
        out_path=args.out_path,
        presence_cmap=presence_cmap,
        background_mode=args.background_mode,
        risk_ylim=risk_ylim,
        clip_risk_for_display=not args.no_clip_risk_for_display,
        risk_line_cmap=args.risk_line_cmap,
        show_legend=args.show_legend,
        show_presence_colorbar=args.show_presence_colorbar,
        dpi=args.dpi,
        title_fontsize=args.title_fontsize,
        label_fontsize=args.label_fontsize,
        profile_height_ratio=args.profile_height_ratio,
        profile_width_ratio=args.profile_width_ratio,
        view_hspace=args.view_hspace,
        panel_wspace=args.panel_wspace,
        profile_linewidth=args.profile_linewidth,
        profile_alpha=args.profile_alpha,
        show_threshold_line=not args.hide_threshold_line,
        risk_threshold=args.risk_threshold,
        show_profile_axis_labels=args.show_profile_axis_labels,
        show_profile_ticks=args.show_profile_ticks,
        profile_title=args.profile_title,
        legend_mode=args.legend_mode,
        legend_y=args.legend_y,
        fill_profile_area=args.fill_profile_area,
        profile_fill_alpha=args.profile_fill_alpha,
        profile_fill_mode=args.profile_fill_mode,
    )

    print(f"[OK] Saved presence heatmap with marginal cutting-risk profiles: {args.out_path}")
    print(f"[INFO] episode_dir={episode_dir}")
    print(f"[INFO] raw_pred_step={step}")


if __name__ == "__main__":
    main()


'''
# Simple Object A example:
python scripts/analysis/plot_presence_with_cutting_risk_profiles.py \
  --cost_map_log /home/dev/workspace/dataset/nedo_dismantling_log/eval/unet_D64_T1000_S20_simple_2d_20260605_133339/simple_paper_A_T8_N6_eta0p5_D0_w0p2_M32_S20_E100000_proposed_A/epsilon_greedy_00/Object_A/episode_0/0_cost_map_logs.pickle \
  --out_path analysis/revise/cutting_risk_maps/object_A/presence_with_marginal_risk_profiles_A.pdf \
  --target_color simple_blue \
  --target blue \
  --side_length 16 \
  --radii 0,1,2 \
  --score_mode ucb \
  --view_specs "Side view:x:-1,Top view:z:2" \
  --fill_profile_area \
  --show_legend \
  --show_presence_colorbar
'''
