from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from denoising_diffusion_pytorch.cost.types import AxisCost, AxisCostEnsemble, AxisDecisionCost
from denoising_diffusion_pytorch.policy.decision.decision_rules import (
    clip_ucb_raw,
    compute_clip_ucb_scores,
)


TargetColor = Literal["blue", "red", "yellow"]
ViewName = Literal["side", "top"]


def parse_radii(text: str) -> list[int]:
    radii: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        radius = int(item)
        if radius < 0:
            raise ValueError(f"Safety margin radius must be non-negative, got {radius}.")
        radii.append(radius)
    if not radii:
        raise ValueError("No valid radii were parsed. Example: --radii 0,1,2")
    return radii


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def select_axis_cost_ensemble(cost_ensembles, target: TargetColor) -> AxisCostEnsemble:
    if not hasattr(cost_ensembles, target):
        raise ValueError(
            f"cost_ensembles has no target color {target!r}. "
            "Expected one of: blue, red, yellow."
        )
    return getattr(cost_ensembles, target)


def to_risk(scores: AxisCost, *, mode: str, decision: AxisDecisionCost | None = None) -> AxisCost:
    if mode == "ucb":
        return scores
    if mode == "decision":
        if decision is None:
            raise ValueError("decision must be provided when mode='decision'.")
        return AxisCost(
            x_axis=(decision.x_axis > 0).astype(float),
            y_axis=(decision.y_axis > 0).astype(float),
            z_axis=(decision.z_axis > 0).astype(float),
        )
    raise ValueError(f"Unknown score_mode={mode!r}. Use 'ucb' or 'decision'.")


def build_view_map(axis_scores: AxisCost, view: ViewName) -> np.ndarray:
    """
    Convert axis-wise scores to a 2D risk map for visualization.

    The policy uses 1D decision costs over x/y/z candidate slices. This function
    displays those exact axis-wise risks as a 2D map by assigning each pixel the
    maximum risk of the two axes visible in the requested view.

      top  view: max(x_axis[x], y_axis[y])
      side view: max(x_axis[x], z_axis[z])

    The map is therefore not a voxel occupancy/presence projection; it is a 2D
    visualization of the actual axis-wise decision cost used by the policy.
    """
    x = np.asarray(axis_scores.x_axis, dtype=float).reshape(-1)
    y = np.asarray(axis_scores.y_axis, dtype=float).reshape(-1)
    z = np.asarray(axis_scores.z_axis, dtype=float).reshape(-1)

    if view == "top":
        return np.maximum(y[:, None], x[None, :])
    if view == "side":
        # Vertical axis is z, horizontal axis is x.
        return np.maximum(z[::-1, None], x[None, :])
    raise ValueError(f"Unknown view={view!r}. Use 'side' or 'top'.")


def maybe_normalize_for_display(score_map: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "decision":
        return np.clip(score_map, 0.0, 1.0)
    return np.clip(score_map, 0.0, 1.0)


def plot_maps(
    *,
    maps: dict[tuple[int, str], np.ndarray],
    radii: list[int],
    views: list[str],
    out_path: Path,
    cmap_name: str,
    score_mode: str,
    show_colorbar: bool,
    dpi: int,
    title_fontsize: int,
    label_fontsize: int,
    colorbar_layout_right: float,
    colorbar_pad: float,
    colorbar_width: float,
    colorbar_shrink: float,
) -> None:
    n_rows = len(views)
    n_cols = len(radii)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(2.0 * n_cols, 5.5), max(1.55 * n_rows, 3.0)),
        squeeze=False,
    )
    cmap = plt.get_cmap(cmap_name)

    for col_idx, radius in enumerate(radii):
        axes[0][col_idx].set_title(f"r={radius}", fontsize=title_fontsize, pad=4)

    for row_idx, view in enumerate(views):
        for col_idx, radius in enumerate(radii):
            ax = axes[row_idx][col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_box_aspect(1)
            image = maybe_normalize_for_display(maps[(radius, view)], mode=score_mode)
            ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
            if col_idx == 0:
                label = "Side view" if view == "side" else "Top view"
                ax.set_ylabel(label, fontsize=label_fontsize, rotation=90, labelpad=8)
            for spine in ax.spines.values():
                spine.set_visible(False)

    if show_colorbar:
        fig.tight_layout(rect=[0.0, 0.0, colorbar_layout_right, 1.0])
        positions = [ax.get_position() for ax in axes.ravel()]
        right_edge = max(pos.x1 for pos in positions)
        bottom_edge = min(pos.y0 for pos in positions)
        top_edge = max(pos.y1 for pos in positions)
        total_height = top_edge - bottom_edge
        cbar_height = total_height * colorbar_shrink
        cbar_bottom = bottom_edge + (total_height - cbar_height) / 2.0
        cbar_left = min(right_edge + colorbar_pad, 0.98 - colorbar_width)
        cax = fig.add_axes([cbar_left, cbar_bottom, colorbar_width, cbar_height])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar_label = "Decision risk" if score_mode == "decision" else "UCB risk score"
        cbar.set_label(cbar_label, fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=label_fontsize - 1)
    else:
        fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_axis_arrays(
    *,
    out_dir: Path | None,
    radii: list[int],
    score_by_radius: dict[int, AxisCost],
    decision_by_radius: dict[int, AxisDecisionCost],
) -> None:
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for radius in radii:
        scores = score_by_radius[radius]
        decision = decision_by_radius[radius]
        np.savez(
            out_dir / f"decision_axis_cost_r{radius}.npz",
            ucb_x=scores.x_axis,
            ucb_y=scores.y_axis,
            ucb_z=scores.z_axis,
            decision_x=decision.x_axis,
            decision_y=decision.y_axis,
            decision_z=decision.z_axis,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the exact safety-margin-aware axis-wise decision costs "
            "used by clip_ucb_raw from an existing *_cost_map_logs.pickle file."
        )
    )
    parser.add_argument("--cost_map_log", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--target", type=str, default="blue", choices=["blue", "red", "yellow"])
    parser.add_argument("--ucb_lb", type=float, required=True, help="Decision threshold used by clip_ucb_raw, e.g. 0.5.")
    parser.add_argument("--radii", type=str, default="0,1,2", help="Comma-separated safety margin radii, e.g. 0,1,2.")
    parser.add_argument("--score_mode", type=str, default="ucb", choices=["ucb", "decision"], help="Plot continuous UCB scores or thresholded binary decision risks.")
    parser.add_argument("--views", type=str, default="side,top", help="Comma-separated views: side,top.")
    parser.add_argument("--cmap", type=str, default="jet")
    parser.add_argument("--show_colorbar", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title_fontsize", type=int, default=11)
    parser.add_argument("--label_fontsize", type=int, default=10)
    parser.add_argument("--colorbar_layout_right", type=float, default=0.86)
    parser.add_argument("--colorbar_pad", type=float, default=0.025)
    parser.add_argument("--colorbar_width", type=float, default=0.018)
    parser.add_argument("--colorbar_shrink", type=float, default=0.45)
    parser.add_argument("--save_axis_arrays_dir", type=Path, default=None)
    args = parser.parse_args()

    radii = parse_radii(args.radii)
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    for view in views:
        if view not in {"side", "top"}:
            raise ValueError(f"Unknown view={view!r}. Use side or top.")

    logs = load_pickle(args.cost_map_log)
    if "cost_ensembles" not in logs:
        raise KeyError(f"cost_map_log does not contain 'cost_ensembles': {args.cost_map_log}")
    cost_ensemble = select_axis_cost_ensemble(logs["cost_ensembles"], args.target)  # type: ignore[arg-type]

    score_by_radius: dict[int, AxisCost] = {}
    decision_by_radius: dict[int, AxisDecisionCost] = {}
    maps: dict[tuple[int, str], np.ndarray] = {}

    for radius in radii:
        scores = compute_clip_ucb_scores(
            cost_ensemble=cost_ensemble,
            safety_margin_voxels=radius,
            ucb_beta=1.0,
        )
        decision = clip_ucb_raw(
            cost_ensemble=cost_ensemble,
            ucb_lb=args.ucb_lb,
            safety_margin_voxels=radius,
        )
        risk = to_risk(scores, mode=args.score_mode, decision=decision)
        score_by_radius[radius] = scores
        decision_by_radius[radius] = decision
        for view in views:
            maps[(radius, view)] = build_view_map(risk, view)  # type: ignore[arg-type]

    save_axis_arrays(
        out_dir=args.save_axis_arrays_dir,
        radii=radii,
        score_by_radius=score_by_radius,
        decision_by_radius=decision_by_radius,
    )

    plot_maps(
        maps=maps,
        radii=radii,
        views=views,
        out_path=args.out_path,
        cmap_name=args.cmap,
        score_mode=args.score_mode,
        show_colorbar=args.show_colorbar,
        dpi=args.dpi,
        title_fontsize=args.title_fontsize,
        label_fontsize=args.label_fontsize,
        colorbar_layout_right=args.colorbar_layout_right,
        colorbar_pad=args.colorbar_pad,
        colorbar_width=args.colorbar_width,
        colorbar_shrink=args.colorbar_shrink,
    )
    print(f"[OK] Saved safety-margin decision map figure: {args.out_path}")


if __name__ == "__main__":
    main()


'''
python scripts/analysis/plot_safety_margin_decision_maps.py \
  --cost_map_log /home/dev/workspace/dataset/nedo_dismantling_log/eval/unet_D64_T1000_S20_simple_2d_20260605_133339/simple_paper_A_T8_N6_eta0p5_D0_w0p2_M32_S20_E100000_proposed_A/epsilon_greedy_00/Object_A/episode_0/0_cost_map_logs.pickle \
  --target blue \
  --ucb_lb 0.5 \
  --radii 0,1,2 \
  --score_mode ucb \
  --out_path analysis/revise/presence_maps_safety_margin/safety_margin_decision_ucb_A.pdf \
  --show_colorbar
'''


'''
python scripts/analysis/plot_safety_margin_decision_maps.py \
  --cost_map_log /home/dev/workspace/dataset/nedo_dismantling_log/eval/unet_D64_T1000_S20_simple_2d_20260605_133339/simple_paper_A_T8_N6_eta0p5_D0_w0p2_M32_S20_E100000_proposed_A/epsilon_greedy_00/Object_A/episode_0/0_cost_map_logs.pickle \
  --target blue \
  --ucb_lb 0.5 \
  --radii 0,1,2 \
  --score_mode decision \
  --out_path analysis/revise/presence_maps_safety_margin/safety_margin_decision_binary_A.pdf \
  --show_colorbar
'''

