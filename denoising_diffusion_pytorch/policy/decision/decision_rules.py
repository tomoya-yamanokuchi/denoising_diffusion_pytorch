# denoising_diffusion_pytorch/policy/decision_rules.py

from __future__ import annotations

import numpy as np

from ...cost.types import AxisCost, AxisCostEnsemble, AxisDecisionCost


def compute_clip_ucb_scores(
    cost_ensemble: AxisCostEnsemble,
    safety_margin_voxels: int = 0,
    ucb_beta: float = 1.0,
) -> AxisCost:
    """
    Compute the continuous axis-wise UCB scores used by clip_ucb_raw.

    The returned scores are computed with the same pre-threshold logic as the
    policy decision rule:

      binary slice cost -> mean + ucb_beta * std -> 1D max filter.

    Keeping this logic in a separate function allows analysis scripts to
    visualize the exact pre-threshold safety-margin-aware scores used by the
    policy without duplicating the decision-rule implementation.
    """
    cost_z_bool = np.where(cost_ensemble.z_axis > 0, 1, 0)
    cost_x_bool = np.where(cost_ensemble.x_axis > 0, 1, 0)
    cost_y_bool = np.where(cost_ensemble.y_axis > 0, 1, 0)

    cost_z_ucb = cost_z_bool.mean(0) + ucb_beta * cost_z_bool.std(0)
    cost_x_ucb = cost_x_bool.mean(0) + ucb_beta * cost_x_bool.std(0)
    cost_y_ucb = cost_y_bool.mean(0) + ucb_beta * cost_y_bool.std(0)

    cost_z_ucb = _max_filter_1d(cost_z_ucb, safety_margin_voxels)
    cost_x_ucb = _max_filter_1d(cost_x_ucb, safety_margin_voxels)
    cost_y_ucb = _max_filter_1d(cost_y_ucb, safety_margin_voxels)

    return AxisCost(
        x_axis=cost_x_ucb,
        y_axis=cost_y_ucb,
        z_axis=cost_z_ucb,
    )


def clip_ucb_raw(
    cost_ensemble: AxisCostEnsemble,
    ucb_lb: float,
    safety_margin_voxels: int = 0,
) -> AxisDecisionCost:
    scores = compute_clip_ucb_scores(
        cost_ensemble=cost_ensemble,
        safety_margin_voxels=safety_margin_voxels,
        ucb_beta=1.0,
    )

    return AxisDecisionCost(
        x_axis=np.where(scores.x_axis <= ucb_lb, 0, 10),
        y_axis=np.where(scores.y_axis <= ucb_lb, 0, 10),
        z_axis=np.where(scores.z_axis <= ucb_lb, 0, 10),
    )


def _max_filter_1d(values: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return values

    values = np.asarray(values, dtype=float).reshape(-1)
    n = len(values)
    filtered = np.empty_like(values)

    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        filtered[i] = values[lo:hi].max()

    return filtered
