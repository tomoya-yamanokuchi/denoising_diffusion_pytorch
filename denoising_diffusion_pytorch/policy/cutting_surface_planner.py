import numpy as np

from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy, pil_image_load_to_numpy
from denoising_diffusion_pytorch.utils.os_utils import create_folder,pickle_utils
from denoising_diffusion_pytorch.policy.types import PlanningPolicyInput
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler
from denoising_diffusion_pytorch.cost.color_mask_cost_estimator import ColorMaskCostEstimator
from denoising_diffusion_pytorch.cost.segmentation_cost_collector import SegmentationCostCollector
from denoising_diffusion_pytorch.policy.decision.decision_aggregator import DecisionAggregator
from denoising_diffusion_pytorch.policy.inference.slice_image_inferencer import SliceImageInferencer

from .ensemble_image_builder import EnsembleImageBuilder
from .types import PolicyConfig, AxisCostSet
from .planning.action_selection.action_candidates_selector import ActionCandidatesSelector
from .planning.visibility.visibility_constraint_set import VisibilityConstraintSet
from .planning.action_definition.action_candidates import ActionCandidates



class cutting_surface_planner():
    def __init__(self,
            slice_image_inferencer : SliceImageInferencer,
            obs_model                 : voxel_cut_handler,
            policy_config             : PolicyConfig,
            action_candidates_selector: ActionCandidatesSelector,
        ):
        self.ensemble_obs_model        = obs_model
        self.slice_image_inferencer    = slice_image_inferencer
        self.policy_config             = policy_config
        self.sample_image_num          = policy_config.inference.sample_image_num
        # ---
        self.color_mask_cost_estimator = ColorMaskCostEstimator(
            obs_model    = obs_model,
            segmentation = policy_config.segmentation,
        )
        # ---
        self.decision_aggregator = DecisionAggregator(
            decision_config=policy_config.decision
        )
        self.action_candidates_selector = action_candidates_selector
        self.voxel_grid_side_length     = policy_config.voxel_grid_side_length
        # ---
        self.ensemble_image_builder    = EnsembleImageBuilder(obs_model)
        self.visibility_constraints    = VisibilityConstraintSet(self.voxel_grid_side_length)
        self.oracle_image_z            = None

    def reset(self):
        self.visibility_constraints = VisibilityConstraintSet(self.voxel_grid_side_length)
        self.oracle_image_z         = None


    def get_optimal_act(self,
            observation_history: dict,
            planning_input     : PlanningPolicyInput,
            iters              : int,
            save_path          : str,
        ):
        last_step_images = self.slice_image_inferencer.predict(planning_input)

        raw_pred_image_save_path = save_path+f"/raw_pred_image/step_{iters}"
        create_folder(raw_pred_image_save_path)
        ## save each generated images
        for k in range(last_step_images.shape[0]):
            pil_image_save_from_numpy(last_step_images[k]/255.0,raw_pred_image_save_path+f"/ensemble_z_{k}.png")
            # pass


        last_step_images_tmp = []
        for k in range(last_step_images.shape[0]):
            # import ipdb; ipdb.set_trace()
            load_last_step_images = pil_image_load_to_numpy(raw_pred_image_save_path+f"/ensemble_z_{k}.png")
            last_step_images_tmp.append(load_last_step_images*255.0)
        last_step_images = np.asarray(last_step_images_tmp)

        ## -------------------- calculate cutting costs --------------------
        collector = SegmentationCostCollector()
        for p in range(self.sample_image_num):
            seg_cost = self.color_mask_cost_estimator.estimate_all(
                image = last_step_images[p] / 255.0,
            )
            collector.add(seg_cost)
        cost_ensembles = collector.build()

        ## ------------ calculate aggregated cost from ensemble  ------------
        costs_decision = self.decision_aggregator.aggregate(cost_ensembles)
        cost_x_b = costs_decision.blue.x_axis
        cost_y_b = costs_decision.blue.y_axis
        cost_z_b = costs_decision.blue.z_axis

        ## ------------------------ create log data  ------------------------
        ensemble_images = self.ensemble_image_builder.build_from_generated_samples(last_step_images)

        cost_map_logs = {
            "cost_ensembles": cost_ensembles,
            "costs_decision": costs_decision,
        }

        #####################################################################
        ## get slice range for pats remove
        #####################################################################
        selection = self.action_candidates_selector.select(
            axis_costs = AxisCostSet(
                x = cost_x_b,
                y = cost_y_b,
                z = cost_z_b,
            ),
            observation_history = observation_history,
        )
        selected_candidates = selection.optimal_selected_slice_range
        self.update_visibility_constraints(selected_candidates)

        # ---- log ----
        cost_map_logs["slice_candidate"] = {
            "candidate_x": None if selection.slice_range_candidates_across_axes.x is None else selection.slice_range_candidates_across_axes.x.to_list(),
            "candidate_y": None if selection.slice_range_candidates_across_axes.y is None else selection.slice_range_candidates_across_axes.y.to_list(),
            "candidate_z": None if selection.slice_range_candidates_across_axes.z is None else selection.slice_range_candidates_across_axes.z.to_list(),
        }
        cost_map_logs["slice_range"] = (
            None if selected_candidates is None else selected_candidates.to_list()
        )
        pickle_utils().save(dataset=cost_map_logs, save_path=save_path+f"/{iters}_cost_map_logs.pickle")


        infos = {"ensemble_image": ensemble_images}

        return selected_candidates, infos


    def set_oracle_obs(self,oracle_obs_image):
        self.oracle_image_z = oracle_obs_image


    def update_visibility_constraints(self, candidates: ActionCandidates):
        self.visibility_constraints.add_from_action_candidates(candidates)
