from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_presence_score_maps import (
    DEFAULT_COLOR_RANGES,
    PanelData,
    add_outside_colorbar,
    apply_crop,
    build_presence_colormap,
    crop_bounds_from_masks,
    pad_to_square,
    parse_view_specs,
    project_volume,
    read_oracle_score_map,
    read_raw_pred_score_map,
    render_score_panel,
    resolve_episode_dir,
)


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


def dilate_score_volume(score_volume: np.ndarray, radius: int) -> np.ndarray:
    """
    Apply a voxel-wise maximum filter with Chebyshev radius `radius`.

    This is intended for visualization of safety-margin-aware presence maps:
    r=0 is the raw target-part presence score, and r>0 expands nearby risky
    voxels by taking the maximum score inside a cubic neighborhood.
    """
    radius = int(radius)
    score_volume = np.asarray(score_volume, dtype=np.float32)

    if radius <= 0:
        return score_volume.copy()

    padded = np.pad(
        score_volume,
        pad_width=radius,
        mode="constant",
        constant_values=0.0,
    )
    out = np.zeros_like(score_volume, dtype=np.float32)
    nx, ny, nz = score_volume.shape

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                sx = slice(radius + dx, radius + dx + nx)
                sy = slice(radius + dy, radius + dy + ny)
                sz = slice(radius + dz, radius + dz + nz)
                out = np.maximum(out, padded[sx, sy, sz])

    return out


def save_score_volumes(score_volumes: dict[int, np.ndarray], out_dir: Path | None) -> None:
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for radius, volume in score_volumes.items():
        np.save(out_dir / f"presence_score_r{radius}.npy", volume)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize safety-margin-aware presence score maps by loading one "
            "raw prediction ensemble and applying voxel-wise dilation radii."
        )
    )
    parser.add_argument("--rollout_root", type=Path, required=True, help="Root such as .../epsilon_greedy_00 containing case/episode directories.")
    parser.add_argument("--case", type=str, required=True, help="Case name, e.g. Object_A.")
    parser.add_argument("--episode", type=int, required=True, help="Episode index used for the raw prediction ensemble.")
    parser.add_argument("--step", type=int, default=-1, help="Prediction step. Negative means the last available step.")
    parser.add_argument("--oracle_root", type=Path, default=None, help="Optional root for the ground-truth oracle image. Defaults to --rollout_root.")
    parser.add_argument("--oracle_episode", type=int, default=None, help="Optional oracle episode index. Defaults to --episode.")
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--radii", type=str, default="0,1,2", help="Comma-separated safety margin radii, e.g. 0,1,2.")
    parser.add_argument("--target_color", type=str, default="blue", choices=sorted(DEFAULT_COLOR_RANGES))
    parser.add_argument("--side_length", type=int, default=None)
    parser.add_argument("--view_specs", type=str, default=None)
    parser.add_argument("--presence_cmap", type=str, default="jet_bright")
    parser.add_argument("--background_mode", type=str, default="low_score", choices=["white", "low_score", "light_gray"])
    parser.add_argument("--ground_truth_background_mode", type=str, default="white", choices=["white", "low_score", "light_gray"])
    parser.add_argument("--ground_truth_style", type=str, default="target_only", choices=["structure", "target_only"])
    parser.add_argument("--no_ground_truth", action="store_true")
    parser.add_argument("--no_square_panels", action="store_true")
    parser.add_argument("--panel_frame", action="store_true")
    parser.add_argument("--crop_padding", type=int, default=2)
    parser.add_argument("--crop_score_threshold", type=float, default=1e-6)
    parser.add_argument("--no_auto_crop", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title_fontsize", type=int, default=11)
    parser.add_argument("--label_fontsize", type=int, default=10)
    parser.add_argument("--add_colorbar", action="store_true", help="Show a colorbar. Kept for compatibility with related scripts.")
    parser.add_argument("--show_colorbar", action="store_true", help="Show a colorbar outside the panel grid.")
    parser.add_argument("--colorbar_layout_right", type=float, default=0.86)
    parser.add_argument("--colorbar_pad", type=float, default=0.025)
    parser.add_argument("--colorbar_width", type=float, default=0.018)
    parser.add_argument("--colorbar_shrink", type=float, default=0.45)
    parser.add_argument("--save_score_volumes_dir", type=Path, default=None, help="Optional directory to save dilated 3D score volumes as .npy files.")
    args = parser.parse_args()

    radii = parse_radii(args.radii)
    view_specs = parse_view_specs(args.view_specs)
    presence_cmap = build_presence_colormap(args.presence_cmap)

    episode_dir = resolve_episode_dir(args.rollout_root, args.case, args.episode)
    base_score_volume = read_raw_pred_score_map(
        episode_dir,
        step=None if args.step < 0 else args.step,
        target_color=args.target_color,
        side_length=args.side_length,
    )

    score_volumes = {
        radius: dilate_score_volume(base_score_volume, radius)
        for radius in radii
    }
    save_score_volumes(score_volumes, args.save_score_volumes_dir)

    oracle_score_volume = None
    if not args.no_ground_truth:
        oracle_root = args.oracle_root if args.oracle_root is not None else args.rollout_root
        oracle_episode = args.oracle_episode if args.oracle_episode is not None else args.episode
        oracle_episode_dir = resolve_episode_dir(oracle_root, args.case, oracle_episode)
        oracle_score_volume = read_oracle_score_map(
            oracle_episode_dir,
            target_color=args.target_color,
            side_length=args.side_length,
        )

    panel_data: dict[tuple[int, int], PanelData] = {}
    crop_masks: dict[tuple[int, int], np.ndarray] = {}

    col_labels = [f"r={radius}" for radius in radii]
    for col_idx, radius in enumerate(radii):
        volume = score_volumes[radius]
        for view_idx, view in enumerate(view_specs):
            score = project_volume(volume, view.projection_axis, view.rot90)
            panel_data[(col_idx, view_idx)] = PanelData(score=score, mask=None, is_ground_truth=False)
            crop_masks[(col_idx, view_idx)] = score > args.crop_score_threshold

    if oracle_score_volume is not None:
        gt_col_idx = len(col_labels)
        col_labels.append("Ground Truth")
        for view_idx, view in enumerate(view_specs):
            score = project_volume(oracle_score_volume, view.projection_axis, view.rot90)
            panel_data[(gt_col_idx, view_idx)] = PanelData(score=score, mask=None, is_ground_truth=True)
            crop_masks[(gt_col_idx, view_idx)] = score > 0.5

    n_cols = len(col_labels)
    n_rows = len(view_specs)

    if not args.no_auto_crop:
        for view_idx, _view in enumerate(view_specs):
            bounds = crop_bounds_from_masks(
                [crop_masks[(col_idx, view_idx)] for col_idx in range(n_cols)],
                args.crop_padding,
            )
            for col_idx in range(n_cols):
                panel = panel_data[(col_idx, view_idx)]
                panel.score, panel.mask = apply_crop(panel.score, panel.mask, bounds)

    if not args.no_square_panels:
        for panel in panel_data.values():
            panel.score, panel.mask = pad_to_square(panel.score, panel.mask)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(2.0 * n_cols, 5.5), max(1.55 * n_rows, 3.0)),
        squeeze=False,
    )

    for col_idx, label in enumerate(col_labels):
        axes[0][col_idx].set_title(label, fontsize=args.title_fontsize, pad=4)

    for col_idx in range(n_cols):
        for view_idx, view in enumerate(view_specs):
            ax = axes[view_idx][col_idx]
            panel = panel_data[(col_idx, view_idx)]
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
            if col_idx == 0:
                ax.set_ylabel(view.label, fontsize=args.label_fontsize, rotation=90, labelpad=8)

    if args.add_colorbar or args.show_colorbar:
        add_outside_colorbar(fig, axes, presence_cmap, args)
    else:
        fig.tight_layout()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[OK] Saved safety-margin presence-score figure: {args.out_path}")


if __name__ == "__main__":
    main()



'''
python scripts/analysis/plot_safety_margin_presence_maps.py \
  --rollout_root /home/dev/workspace/dataset/nedo_dismantling_log/eval/unet_D64_T1000_S20_simple_2d_20260605_133339/simple_paper_A_T8_N6_eta0p5_D0_w0p2_M32_S20_E100000_proposed_A/epsilon_greedy_00 \
  --case Object_A \
  --episode 0 \
  --step 0 \
  --target_color simple_blue \
  --side_length 16 \
  --radii 0,1,2 \
  --oracle_root /home/dev/workspace/dataset/nedo_dismantling_log/eval/unet_D64_T1000_S20_simple_2d_20260605_133339/simple_paper_T8_N6_eta0p5_D0_w0p2_M32_S20_E100000_proposed_ABC_GT/oracle_obs \
  --out_path analysis/revise/presence_maps_safety_margin/object_A/safety_margin_presence_A.pdf \
  --presence_cmap jet_bright \
  --background_mode low_score \
  --ground_truth_style target_only \
  --show_colorbar
'''
