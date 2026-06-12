from __future__ import annotations

import argparse
import re
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
)
from plot_safety_margin_presence_maps import dilate_score_volume, parse_radii


def infer_step_from_cost_map_log(path: Path) -> int | None:
    match = re.match(r"(-?\d+)_cost_map_logs\.pickle$", path.name)
    return None if match is None else int(match.group(1))


def save_score_volumes(score_volumes: dict[int, np.ndarray], out_dir: Path | None) -> None:
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for radius, volume in score_volumes.items():
        np.save(out_dir / f"presence_score_r{radius}.npy", volume)


def save_projected_maps(
    *,
    score_volumes: dict[int, np.ndarray],
    view_specs,
    out_dir: Path | None,
) -> None:
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for radius, volume in score_volumes.items():
        for view in view_specs:
            projected = project_volume(volume, view.projection_axis, view.rot90)
            safe_label = view.label.lower().replace(" ", "_")
            np.save(out_dir / f"presence_score_r{radius}_{safe_label}.npy", projected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot target-part 3D shape presence score maps anchored to a specific "
            "evaluation cost-map log. The script uses the matching raw_pred_image "
            "ensemble from the same episode/step that produced the cost_map_log."
        )
    )
    parser.add_argument("--cost_map_log", type=Path, required=True, help="Path to *_cost_map_logs.pickle. Used to infer episode directory and step.")
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--raw_pred_episode_dir", type=Path, default=None, help="Optional episode directory containing raw_pred_image. Defaults to cost_map_log.parent.")
    parser.add_argument("--step", type=int, default=None, help="Optional raw prediction step. Defaults to the prefix of *_cost_map_logs.pickle.")
    parser.add_argument("--oracle_episode_dir", type=Path, default=None, help="Optional episode directory containing oracle_obs_cast_z*.png. Defaults to raw_pred_episode_dir.")
    parser.add_argument("--target_color", type=str, default="blue", choices=sorted(DEFAULT_COLOR_RANGES))
    parser.add_argument("--side_length", type=int, default=None)
    parser.add_argument("--radii", type=str, default="0", help="Comma-separated visualization dilation radii for the 3D presence volume. Use 0 for the exact raw-prediction presence map.")
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
    parser.add_argument("--save_score_volumes_dir", type=Path, default=None, help="Optional directory to save 3D presence score volumes as .npy files.")
    parser.add_argument("--save_projected_maps_dir", type=Path, default=None, help="Optional directory to save projected 2D maps as .npy files.")
    args = parser.parse_args()

    episode_dir = args.raw_pred_episode_dir if args.raw_pred_episode_dir is not None else args.cost_map_log.parent
    step = args.step
    if step is None:
        step = infer_step_from_cost_map_log(args.cost_map_log)
    if step is None:
        raise ValueError(
            "Could not infer step from cost_map_log filename. "
            "Use a name like 0_cost_map_logs.pickle or pass --step explicitly."
        )

    radii = parse_radii(args.radii)
    view_specs = parse_view_specs(args.view_specs)
    presence_cmap = build_presence_colormap(args.presence_cmap)

    base_score_volume = read_raw_pred_score_map(
        episode_dir,
        step=step,
        target_color=args.target_color,
        side_length=args.side_length,
    )
    score_volumes = {
        radius: dilate_score_volume(base_score_volume, radius)
        for radius in radii
    }
    save_score_volumes(score_volumes, args.save_score_volumes_dir)
    save_projected_maps(
        score_volumes=score_volumes,
        view_specs=view_specs,
        out_dir=args.save_projected_maps_dir,
    )

    oracle_score_volume = None
    if not args.no_ground_truth:
        oracle_episode_dir = args.oracle_episode_dir if args.oracle_episode_dir is not None else episode_dir
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
    print(f"[OK] Saved cost-log anchored presence-score figure: {args.out_path}")
    print(f"[INFO] episode_dir={episode_dir}")
    print(f"[INFO] raw_pred_step={step}")


if __name__ == "__main__":
    main()


COST_LOG = "/home/dev/workspace/dataset/nedo_dismantling_log/eval/unet_D64_T1000_S20_simple_2d_20260605_133339/simple_paper_A_T8_N6_eta0p5_D0_w0p2_M32_S20_E100000_proposed_A/epsilon_greedy_00/Object_A/episode_0/0_cost_map_logs.pickle"
OUT_DIR  = "analysis/debug_axis_alignment/Object_A_ep0_step0"


# simpel: A
'''
python scripts/analysis/plot_cost_log_presence_score_maps.py \
  --cost_map_log ${COST_LOG} \
  --out_path ${OUT_DIR}/presence_frequency.pdf \
  --target_color simple_blue \
  --side_length 16 \
  --radii 0 \
  --save_score_volumes_dir ${OUT_DIR}/presence_volume \
  --save_projected_maps_dir ${OUT_DIR}/presence_projected \
  --presence_cmap jet_bright \
  --background_mode low_score \
  --ground_truth_style target_only \
  --show_colorbar
'''


'''
python scripts/analysis/plot_cost_log_presence_score_maps.py \
  --cost_map_log /home/dev/workspace/dataset/nedo_dismantling_log/eval/unet_D64_T1000_S20_simple_2d_20260605_133339/simple_paper_A_T8_N6_eta0p5_D0_w0p2_M32_S20_E100000_proposed_A/epsilon_greedy_00/Object_A/episode_0/0_cost_map_logs.pickle \
  --out_path analysis/revise/presence_maps_safety_margin/object_A/cost_log_presence_safety_margin_A.pdf \
  --target_color simple_blue \
  --side_length 16 \
  --radii 0,1,2 \
  --presence_cmap jet_bright \
  --background_mode low_score \
  --ground_truth_style target_only \
  --show_colorbar
'''


'''
python scripts/analysis/plot_cost_log_presence_score_maps.py \
  --cost_map_log /home/dev/workspace/dataset/nedo_dismantling_log/eval/unet_D64_T1000_S20_simple_2d_20260605_133339/simple_paper_A_T8_N6_eta0p5_D0_w0p2_M32_S20_E100000_proposed_A/epsilon_greedy_00/Object_A/episode_0/0_cost_map_logs.pickle \
  --out_path analysis/revise/presence_maps_safety_margin/object_A/cost_log_presence_safety_margin_A.pdf \
  --target_color simple_blue \
  --side_length 16 \
  --radii 0,1,2 \
  --presence_cmap jet_bright \
  --background_mode low_score \
  --ground_truth_style target_only \
  --show_colorbar \
  --save_score_volumes_dir analysis/revise/presence_maps_safety_margin/object_A/volumes_A \
  --save_projected_maps_dir analysis/revise/presence_maps_safety_margin/object_A/projected_A
'''
