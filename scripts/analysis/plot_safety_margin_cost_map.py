# scripts/analysis/plot_safety_margin_cost_map.py
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


AxisName = Literal["x", "y", "z"]
TargetColor = Literal["blue", "red", "yellow"]


AXES: list[AxisName] = ["x", "y", "z"]


AXIS_LABELS = {
    "x": "X-axis slices",
    "y": "Y-axis slices",
    "z": "Z-axis slices",
}


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def resolve_cost_map_log_path(
    *,
    cost_map_log: Path | None,
    rollout_root: Path | None,
    case: str | None,
    episode: int | None,
    step: int | None,
) -> Path:
    if cost_map_log is not None:
        if not cost_map_log.exists():
            raise FileNotFoundError(cost_map_log)
        return cost_map_log

    if rollout_root is None or case is None or episode is None or step is None:
        raise ValueError(
            "Specify either --cost_map_log, or all of "
            "--rollout_root, --case, --episode, and --step."
        )

    path = rollout_root / case / f"episode_{episode}" / f"{step}_cost_map_logs.pickle"

    if not path.exists():
        raise FileNotFoundError(
            f"Cost-map log was not found:\n{path}\n\n"
            "Expected path pattern:\n"
            "<rollout_root>/<case>/episode_<episode>/<step>_cost_map_logs.pickle"
        )

    return path


def get_axis_ensemble(
    logs: dict,
    *,
    target_color: TargetColor,
    axis: AxisName,
) -> np.ndarray:
    cost_ensembles = logs["cost_ensembles"]
    color_ensemble = getattr(cost_ensembles, target_color)
    values = getattr(color_ensemble, f"{axis}_axis")
    return np.asarray(values)


def compute_ucb_risk(
    axis_ensemble: np.ndarray,
    *,
    ucb_beta: float = 1.0,
) -> np.ndarray:
    """
    Reproduce the UCB-style presence score used in decision_rules.clip_ucb_raw.

    axis_ensemble shape is expected to be:
        (num_samples, num_slices)
    """
    axis_ensemble = np.asarray(axis_ensemble)

    if axis_ensemble.ndim != 2:
        raise ValueError(
            "Expected axis_ensemble to have shape (num_samples, num_slices), "
            f"but got shape {axis_ensemble.shape}."
        )

    presence_bool = np.where(axis_ensemble > 0, 1.0, 0.0)

    return presence_bool.mean(axis=0) + ucb_beta * presence_bool.std(axis=0)


def max_filter_1d(values: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return np.asarray(values, dtype=float)

    values = np.asarray(values, dtype=float).reshape(-1)
    n = len(values)
    filtered = np.empty_like(values)

    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        filtered[i] = values[lo:hi].max()

    return filtered


def axis_offset(axis: AxisName, side_length: int) -> int:
    if axis == "z":
        return 0
    if axis == "x":
        return side_length
    if axis == "y":
        return 2 * side_length
    raise ValueError(f"Unsupported axis: {axis}")


def global_indices_to_local_indices(
    global_indices: list[int] | None,
    *,
    axis: AxisName,
    side_length: int,
) -> list[int]:
    if not global_indices:
        return []

    offset = axis_offset(axis, side_length)
    local_indices = []

    for global_index in global_indices:
        if offset <= int(global_index) < offset + side_length:
            local_indices.append(int(global_index) - offset)

    return local_indices


def get_selected_local_indices(
    logs: dict,
    *,
    axis: AxisName,
    side_length: int,
) -> list[int]:
    global_indices = logs.get("slice_range")
    return global_indices_to_local_indices(
        global_indices,
        axis=axis,
        side_length=side_length,
    )


def get_candidate_local_indices(
    logs: dict,
    *,
    axis: AxisName,
    side_length: int,
) -> list[int]:
    slice_candidate = logs.get("slice_candidate", {})
    key = f"candidate_{axis}"
    global_indices = slice_candidate.get(key)
    return global_indices_to_local_indices(
        global_indices,
        axis=axis,
        side_length=side_length,
    )


def add_selected_region(
    ax,
    selected_indices: list[int],
    *,
    label: str = "selected cut",
) -> None:
    if len(selected_indices) == 0:
        return

    lo = min(selected_indices) - 0.5
    hi = max(selected_indices) + 0.5

    ax.axvspan(
        lo,
        hi,
        alpha=0.18,
        color="gray",
        label=label,
    )


def add_candidate_markers(
    ax,
    candidate_indices: list[int],
    *,
    y: float,
) -> None:
    if len(candidate_indices) == 0:
        return

    ax.scatter(
        candidate_indices,
        np.full(len(candidate_indices), y),
        marker="|",
        s=80,
        color="black",
        alpha=0.7,
        label="feasible candidates",
    )


def plot_axis_line(
    ax,
    *,
    standard_risk: np.ndarray,
    margin_risk: np.ndarray,
    eta: float,
    safety_margin_voxels: int,
    axis: AxisName,
    selected_indices: list[int],
    candidate_indices: list[int],
    show_selected: bool,
    show_candidates: bool,
    risk_ymax: float | None,
) -> None:
    x = np.arange(len(standard_risk))

    ax.plot(
        x,
        standard_risk,
        marker="o",
        label="Standard risk map",
    )
    ax.plot(
        x,
        margin_risk,
        marker="s",
        label=f"Safety-margin-aware map ($r={safety_margin_voxels}$)",
    )

    ax.axhline(
        eta,
        linestyle="--",
        linewidth=1.2,
        color="black",
        label=f"threshold $\\eta={eta}$",
    )

    if show_selected:
        add_selected_region(ax, selected_indices)

    if show_candidates:
        y_marker = -0.04 if risk_ymax is None else -0.04 * risk_ymax
        add_candidate_markers(ax, candidate_indices, y=y_marker)

    ax.set_title(AXIS_LABELS[axis])
    ax.set_xlabel("Slice index")
    ax.set_ylabel("Presence score")
    ax.grid(True, alpha=0.3)

    if risk_ymax is not None:
        ax.set_ylim(-0.08 * risk_ymax, risk_ymax)
    else:
        ymax = max(float(standard_risk.max()), float(margin_risk.max()), eta)
        ax.set_ylim(-0.05 * ymax, ymax * 1.10)


def plot_axis_strip(
    ax,
    *,
    standard_risk: np.ndarray,
    margin_risk: np.ndarray,
    eta: float,
    safety_margin_voxels: int,
    axis: AxisName,
    selected_indices: list[int],
    show_selected: bool,
    vmax: float,
) -> None:
    values = np.stack(
        [
            np.clip(standard_risk, 0.0, vmax),
            np.clip(margin_risk, 0.0, vmax),
        ],
        axis=0,
    )

    im = ax.imshow(
        values,
        aspect="auto",
        vmin=0.0,
        vmax=vmax,
        cmap="jet",
        interpolation="nearest",
    )

    # Threshold contour-like visual guide.
    for row_idx, risk in enumerate([standard_risk, margin_risk]):
        risky = np.where(risk > eta)[0]
        if len(risky) > 0:
            ax.scatter(
                risky,
                np.full(len(risky), row_idx),
                marker="|",
                s=80,
                color="white",
                alpha=0.85,
            )

    if show_selected and len(selected_indices) > 0:
        for idx in selected_indices:
            ax.axvline(idx, color="black", linewidth=1.2, alpha=0.8)

    ax.set_title(AXIS_LABELS[axis])
    ax.set_xlabel("Slice index")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(
        [
            "Standard",
            f"Margin $r={safety_margin_voxels}$",
        ]
    )

    return im


def plot_cost_map(
    *,
    logs: dict,
    target_color: TargetColor,
    axes_to_plot: list[AxisName],
    eta: float,
    safety_margin_voxels: int,
    ucb_beta: float,
    plot_style: str,
    show_selected: bool,
    show_candidates: bool,
    risk_ymax: float | None,
    strip_vmax: float,
    out_path: Path,
) -> None:
    n_axes = len(axes_to_plot)

    if plot_style == "line":
        fig, axes = plt.subplots(
            1,
            n_axes,
            figsize=(4.2 * n_axes, 3.3),
            squeeze=False,
        )
        axes = axes[0]

        for ax, axis in zip(axes, axes_to_plot):
            axis_ensemble = get_axis_ensemble(
                logs,
                target_color=target_color,
                axis=axis,
            )
            standard_risk = compute_ucb_risk(axis_ensemble, ucb_beta=ucb_beta)
            margin_risk = max_filter_1d(standard_risk, safety_margin_voxels)

            side_length = len(standard_risk)
            selected_indices = get_selected_local_indices(
                logs,
                axis=axis,
                side_length=side_length,
            )
            candidate_indices = get_candidate_local_indices(
                logs,
                axis=axis,
                side_length=side_length,
            )

            plot_axis_line(
                ax,
                standard_risk=standard_risk,
                margin_risk=margin_risk,
                eta=eta,
                safety_margin_voxels=safety_margin_voxels,
                axis=axis,
                selected_indices=selected_indices,
                candidate_indices=candidate_indices,
                show_selected=show_selected,
                show_candidates=show_candidates,
                risk_ymax=risk_ymax,
            )

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(labels)),
            frameon=False,
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.86))

    elif plot_style == "strip":
        fig, axes = plt.subplots(
            1,
            n_axes,
            figsize=(4.2 * n_axes, 2.4),
            squeeze=False,
        )
        axes = axes[0]

        last_im = None

        for ax, axis in zip(axes, axes_to_plot):
            axis_ensemble = get_axis_ensemble(
                logs,
                target_color=target_color,
                axis=axis,
            )
            standard_risk = compute_ucb_risk(axis_ensemble, ucb_beta=ucb_beta)
            margin_risk = max_filter_1d(standard_risk, safety_margin_voxels)

            side_length = len(standard_risk)
            selected_indices = get_selected_local_indices(
                logs,
                axis=axis,
                side_length=side_length,
            )

            last_im = plot_axis_strip(
                ax,
                standard_risk=standard_risk,
                margin_risk=margin_risk,
                eta=eta,
                safety_margin_voxels=safety_margin_voxels,
                axis=axis,
                selected_indices=selected_indices,
                show_selected=show_selected,
                vmax=strip_vmax,
            )

        if last_im is not None:
            cbar = fig.colorbar(
                last_im,
                ax=axes,
                fraction=0.03,
                pad=0.02,
            )
            cbar.set_label("Presence score")

        fig.tight_layout()

    else:
        raise ValueError(f"Unsupported plot_style: {plot_style}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def parse_axes(axis_text: str) -> list[AxisName]:
    if axis_text == "all":
        return ["x", "y", "z"]

    axes = [x.strip() for x in axis_text.split(",") if x.strip()]
    invalid = [x for x in axes if x not in AXES]
    if invalid:
        raise ValueError(f"Invalid axes: {invalid}. Available axes: {AXES} or 'all'.")

    return axes  # type: ignore[return-value]


def build_default_out_path(
    *,
    out_dir: Path,
    case: str | None,
    episode: int | None,
    step: int | None,
    target_color: str,
    axis: str,
    plot_style: str,
    fmt: str,
) -> Path:
    case_label = case or "case"
    episode_label = "unknown" if episode is None else str(episode)
    step_label = "unknown" if step is None else str(step)

    filename = (
        f"safety_margin_cost_map_"
        f"{case_label}_episode_{episode_label}_step_{step_label}_"
        f"{target_color}_{axis}_{plot_style}.{fmt}"
    )

    return out_dir / filename


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cost_map_log",
        type=Path,
        default=None,
        help="Direct path to <step>_cost_map_logs.pickle.",
    )
    parser.add_argument(
        "--rollout_root",
        type=Path,
        default=None,
        help="Rollout root used with --case, --episode, and --step.",
    )
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)

    parser.add_argument(
        "--target_color",
        type=str,
        default="blue",
        choices=["blue", "red", "yellow"],
        help="Target part color used in the segmentation cost ensemble.",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="all",
        help="Axis to visualize: x, y, z, x,y,z, or all.",
    )

    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--safety_margin_voxels", type=int, required=True)
    parser.add_argument("--ucb_beta", type=float, default=1.0)

    parser.add_argument(
        "--plot_style",
        type=str,
        default="line",
        choices=["line", "strip"],
        help="line is best for analysis; strip is closer to a heat-map visualization.",
    )

    parser.add_argument(
        "--show_selected",
        action="store_true",
        help="Overlay the selected slice range saved in the rollout log.",
    )
    parser.add_argument(
        "--show_candidates",
        action="store_true",
        help="Show feasible candidate slice indices in line plots.",
    )

    parser.add_argument(
        "--risk_ymax",
        type=float,
        default=1.2,
        help="Y-axis upper limit for line plots. Use -1 for automatic scaling.",
    )
    parser.add_argument(
        "--strip_vmax",
        type=float,
        default=1.0,
        help="Colorbar upper limit for strip plots. Values above this are clipped.",
    )

    parser.add_argument("--out_path", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("analysis/revise/safety_margin/cost_map_figures"))
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"])

    args = parser.parse_args()

    cost_map_log_path = resolve_cost_map_log_path(
        cost_map_log=args.cost_map_log,
        rollout_root=args.rollout_root,
        case=args.case,
        episode=args.episode,
        step=args.step,
    )

    logs = load_pickle(cost_map_log_path)

    axes_to_plot = parse_axes(args.axis)

    risk_ymax = None if args.risk_ymax < 0 else args.risk_ymax

    if args.out_path is None:
        out_path = build_default_out_path(
            out_dir=args.out_dir,
            case=args.case,
            episode=args.episode,
            step=args.step,
            target_color=args.target_color,
            axis=args.axis.replace(",", ""),
            plot_style=args.plot_style,
            fmt=args.format,
        )
    else:
        out_path = args.out_path

    plot_cost_map(
        logs=logs,
        target_color=args.target_color,  # type: ignore[arg-type]
        axes_to_plot=axes_to_plot,
        eta=args.eta,
        safety_margin_voxels=args.safety_margin_voxels,
        ucb_beta=args.ucb_beta,
        plot_style=args.plot_style,
        show_selected=args.show_selected,
        show_candidates=args.show_candidates,
        risk_ymax=risk_ymax,
        strip_vmax=args.strip_vmax,
        out_path=out_path,
    )

    print(f"[OK] Loaded cost-map log: {cost_map_log_path}")
    print(f"[OK] Saved figure       : {out_path}")


if __name__ == "__main__":
    main()

'''
python scripts/analysis/plot_safety_margin_cost_map.py \
  --rollout_root /home/dev/workspace/dataset/nedo_dismantling_log/eval/unet_D64_T1000_S20_simple_2d_20260605_133339/simple_paper_A_T8_N6_eta0p5_D2_w0p2_M32_S20_E100000_proposed_A_action_noise_analysis/epsilon_greedy_00 \
  --case Object_A \
  --episode 0 \
  --step 3 \
  --eta 0.5 \
  --safety_margin_voxels 2 \
  --plot_style line \
  --axis all \
  --out_dir analysis/revise/safety_margin/cost_map_figures
'''
