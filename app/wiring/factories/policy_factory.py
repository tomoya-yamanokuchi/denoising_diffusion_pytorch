from ..types.policy_assets import PolicyAssets
from denoising_diffusion_pytorch.policy.cutting_surface_planner_v9 import cutting_surface_planner
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler
from denoising_diffusion_pytorch.policy.planning.action_axis_index_mapper import ActionAxisIndexMapper
from denoising_diffusion_pytorch.policy.planning.active_range_detector import ActiveRangeDetector
from denoising_diffusion_pytorch.policy.planning.local_candidate_range_factory import LocalCandidateRangeFactory
from denoising_diffusion_pytorch.policy.planning.observed_action_pruner import ObservedActionPruner
from denoising_diffusion_pytorch.policy.planning.axis_candidate_selection_policy import AxisCandidateSelectionPolicy
from denoising_diffusion_pytorch.policy.planning.axis_candidate_range_builder import AxisCandidateRangeBuilder
from denoising_diffusion_pytorch.policy.planning.slice_range_selection_policy import SliceRangeSelectionPolicy
from denoising_diffusion_pytorch.policy.planning.axis_slice_range_selector import AxisSliceRangeSelector



class PolicyFactory:
    def __init__(self, assets: PolicyAssets):
        self._assets = assets


    def create(self, obs_model: voxel_cut_handler):
        # ---
        side_length = obs_model.voxel_hander.grid_side_len

        mapper                  = ActionAxisIndexMapper(side_length=side_length)
        active_range_detector   = ActiveRangeDetector(cost_threshold=0.0)
        local_candidate_factory = LocalCandidateRangeFactory()
        pruner                  = ObservedActionPruner()

        axis_candidate_selection_policy = AxisCandidateSelectionPolicy(
            empty_candidate_fallback="single_zero"
        )

        axis_candidate_range_builder = AxisCandidateRangeBuilder(
            mapper                  = mapper,
            active_range_detector   = active_range_detector,
            local_candidate_factory = local_candidate_factory,
            pruner                  = pruner,
            selection_policy        = axis_candidate_selection_policy,
        )

        slice_range_selection_policy = SliceRangeSelectionPolicy()
        # split_update_factory         = SplitObservationUpdateFactory(mapper=mapper)

        axis_slice_range_selector = AxisSliceRangeSelector(
            candidate_builder=axis_candidate_range_builder,
            selection_policy=slice_range_selection_policy,
            # split_update_factory=split_update_factory,
            expected_side_length=side_length,
        )

        return cutting_surface_planner(
            obs_model        = obs_model,
            # ---
            inferencer       = self._assets.trained_assets.inferencer,
            trainer          = self._assets.trained_assets.trainer,
            policy_config    = self._assets.policy_config,
            axis_slice_range_selector = axis_slice_range_selector,
        )

