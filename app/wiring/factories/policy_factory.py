from ..types.policy_assets import PolicyAssets
from denoising_diffusion_pytorch.policy.cutting_surface_planner_v9 import cutting_surface_planner
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler
from denoising_diffusion_pytorch.policy.planning.candidate_building.active_range_detector import ActiveRangeDetector
from denoising_diffusion_pytorch.policy.planning.candidate_building.local_candidate_range_factory import LocalCandidateRangeFactory
from denoising_diffusion_pytorch.policy.planning.candidate_building.observed_action_pruner import ObservedActionPruner
from denoising_diffusion_pytorch.policy.planning.candidate_building.axis_candidate_selection_policy import AxisCandidateSelectionPolicy
from denoising_diffusion_pytorch.policy.planning.candidate_building.axis_candidate_range_builder import AxisCandidateRangeBuilder
from denoising_diffusion_pytorch.policy.planning.action_selection.selection_policy import SelectionPolicy
from denoising_diffusion_pytorch.policy.planning.action_selection.action_candidates_selector import ActionCandidatesSelector
from denoising_diffusion_pytorch.policy.planning.candidate_building.action_candidate_building_coordinator import ActionCandidateBuildingCoordinator


class PolicyFactory:
    def __init__(self, assets: PolicyAssets):
        self._assets = assets


    def create(self, obs_model: voxel_cut_handler):
        # ---
        side_length = obs_model.voxel_hander.grid_side_len

        active_range_detector   = ActiveRangeDetector(cost_threshold=0.0)
        local_candidate_factory = LocalCandidateRangeFactory()
        pruner                  = ObservedActionPruner()

        axis_candidate_selection_policy = AxisCandidateSelectionPolicy()

        axis_candidate_range_builder = AxisCandidateRangeBuilder(
            active_range_detector   = active_range_detector,
            local_candidate_factory = local_candidate_factory,
            pruner                  = pruner,
            selection_policy        = axis_candidate_selection_policy,
            expected_side_length    = side_length,
        )

        action_candidates_selector = ActionCandidatesSelector(
            candidate_coordinator = ActionCandidateBuildingCoordinator(
                candidate_builder=axis_candidate_range_builder,
                expected_side_length=side_length,
            ),
            selection_policy = SelectionPolicy(),
        )

        return cutting_surface_planner(
            obs_model                  = obs_model,
            # ---
            inferencer                 = self._assets.trained_assets.inferencer,
            trainer                    = self._assets.trained_assets.trainer,
            policy_config              = self._assets.policy_config,
            # ---
            action_candidates_selector = action_candidates_selector,

        )

