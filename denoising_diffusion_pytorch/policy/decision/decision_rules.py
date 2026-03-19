# denoising_diffusion_pytorch/policy/decision_rules.py

from __future__ import annotations

import numpy as np

from ...cost.types import AxisCostEnsemble, AxisDecisionCost


def clip_ucb_raw(cost_ensemble: AxisCostEnsemble, ucb_lb: float) -> AxisDecisionCost:
    cost_z_bool = np.where(cost_ensemble.z_axis > 0, 1, 0)
    cost_x_bool = np.where(cost_ensemble.x_axis > 0, 1, 0)
    cost_y_bool = np.where(cost_ensemble.y_axis > 0, 1, 0)

    ucb_beta = 1.0

    cost_z_ucb = cost_z_bool.mean(0) + ucb_beta * cost_z_bool.std(0)
    cost_x_ucb = cost_x_bool.mean(0) + ucb_beta * cost_x_bool.std(0)
    cost_y_ucb = cost_y_bool.mean(0) + ucb_beta * cost_y_bool.std(0)

    return AxisDecisionCost(
        x_axis=np.where(cost_x_ucb <= ucb_lb, 0, 10),
        y_axis=np.where(cost_y_ucb <= ucb_lb, 0, 10),
        z_axis=np.where(cost_z_ucb <= ucb_lb, 0, 10),
    )
