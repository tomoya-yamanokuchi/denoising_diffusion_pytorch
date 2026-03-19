# denoising_diffusion_pytorch/policy/decision_aggregator.py

from __future__ import annotations

from denoising_diffusion_pytorch.cost.types import (
    SegmentationCostEnsemble,
    SegmentationDecisionCost,
)
from denoising_diffusion_pytorch.policy.decision.decision_rules import clip_ucb_raw
from denoising_diffusion_pytorch.action_plan.types import DecisionConfig


class DecisionAggregator:
    def __init__(self, decision_config: DecisionConfig):
        self.decision_config = decision_config

    def aggregate(self, cost_ensembles: SegmentationCostEnsemble) -> SegmentationDecisionCost:
        rule = self._resolve_rule()

        return SegmentationDecisionCost(
            blue   = rule(cost_ensembles.blue),
            red    = rule(cost_ensembles.red),
            yellow = rule(cost_ensembles.yellow),
        )

    def _resolve_rule(self):
        mode = self.decision_config.mode

        if mode == "clip_ucb_raw":
            ucb_lb = self.decision_config.param.ucb_lb
            return lambda axis_ensemble: clip_ucb_raw(axis_ensemble, ucb_lb=ucb_lb)

        raise ValueError(f"Unknown decision mode: {mode}")
