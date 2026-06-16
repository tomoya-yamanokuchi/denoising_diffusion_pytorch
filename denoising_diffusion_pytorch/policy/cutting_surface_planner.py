# denoising_diffusion_pytorch/policy/cutting_surface_planner.py
import numpy as np

from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy, pil_image_load_to_numpy
from denoising_diffusion_pytorch.utils.os_utils import create_folder,pickle_utils
from denoising_diffusion_pytorch.policy.types import PlanningPolicyInput
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler
from denoising_diffusion_pytorch.cost.color_mask_cost_estimator import ColorMaskCostEstimator
from denoising_diffusion_pytorch.cost.segmentation_cost_collector import SegmentationCostCollector
from denoising_diffusion_pytorch.cost.types import AxisCost
from denoising_diffusion_pytorch.policy.decision.decision_aggregator import DecisionAggregator
from denoising_diffusion_pytorch.policy.inference.slice_image_inferencer import SliceImageInferencer
from denoising_diffusion_pytorch.policy.planning.random_planning_service import (
    RandomPlanningService,
)

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
        # ---
        self.random_planning_service = RandomPlanningService(
            action_candidates_selector=action_candidates_selector,
        )

    def reset(self):
        self.visibility_constraints = VisibilityConstraintSet(self.voxel_grid_side_length)
        self.oracle_image_z         = None


    def get_optimal_act(self,
            observation_history: dict,
            planning_input     : PlanningPolicyInput,
            iters              : int,
            save_path          : str,
        ):
        if self.policy_config.control.mode == "random":
            selected_candidates, infos = self.random_planning_service.plan(
                observation_history=observation_history,
                iters=iters,
                save_path=save_path,
            )
            self.update_visibility_constraints(selected_candidates)
            return selected_candidates, infos

        if self.policy_config.control.mode == "oracle_obs":
            last_step_images = self._predict_oracle_images()
            # import ipdb; ipdb.set_trace()
        else:
            last_step_images = self.slice_image_inferencer.predict(planning_input)
            # import ipdb; ipdb.set_trace()

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

        # ---- decision-reliability metric logs ----
        # This only evaluates and saves diagnostic metrics. It does not feed back
        # into costs_decision, action candidate selection, or visibility updates.
        cost_map_logs["brier_score"] = self._compute_target_surface_brier_score_logs(
            cost_ensembles=cost_ensembles,
        )

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


    def _predict_oracle_images(self) -> np.ndarray:
        if self.oracle_image_z is None:
            raise RuntimeError(
                "oracle_image_z is None. "
                "set_oracle_obs() must be called before oracle_obs planning."
            )

        oracle = np.asarray(self.oracle_image_z)

        # oracle_obs_model の画像は通常 0–1 float。
        # ただし将来 0–255 が渡っても壊れないようにガードする。
        if oracle.max() <= 1.0:
            oracle = oracle * 255.0

        oracle = np.clip(oracle, 0, 255).astype(np.uint8)

        return np.repeat(
            oracle[None, ...],
            repeats=self.sample_image_num,
            axis=0,
        )


    def _compute_target_surface_brier_score_logs(self, cost_ensembles) -> dict:
        """
        Evaluate the target-surface Brier score for planning diagnostics.

        For each axis-wise cutting surface a, the probability p(a) is the
        fraction of generated samples whose target-color slice cost is non-zero.
        The ground-truth label y(a) is computed from the current oracle target
        image and is one when the target part intersects that cutting surface.

        This metric is intentionally separated from the UCB decision score used
        by the planner: p(a)=mean(binary target presence), while planning still
        uses costs_decision computed by DecisionAggregator.
        """
        if self.oracle_image_z is None:
            return {
                "metric": "target_surface_brier_score",
                "target": "blue",
                "available": False,
                "reason": "oracle_image_z is None",
            }

        pred_presence = self._mean_binary_presence_from_ensemble(cost_ensembles.blue)
        gt_presence = self._oracle_target_binary_presence(target="blue")

        axis_values = {
            "x": self._brier_vector(pred_presence.x_axis, gt_presence.x_axis),
            "y": self._brier_vector(pred_presence.y_axis, gt_presence.y_axis),
            "z": self._brier_vector(pred_presence.z_axis, gt_presence.z_axis),
        }

        concatenated = np.concatenate([
            axis_values["x"],
            axis_values["y"],
            axis_values["z"],
        ])

        return {
            "metric": "target_surface_brier_score",
            "target": "blue",
            "available": True,
            "overall": float(np.mean(concatenated)),
            "per_axis": {
                axis: float(np.mean(values)) for axis, values in axis_values.items()
            },
            "num_surfaces": {
                axis: int(values.size) for axis, values in axis_values.items()
            },
            "presence_probability": {
                "x": pred_presence.x_axis,
                "y": pred_presence.y_axis,
                "z": pred_presence.z_axis,
            },
            "ground_truth_presence": {
                "x": gt_presence.x_axis,
                "y": gt_presence.y_axis,
                "z": gt_presence.z_axis,
            },
            "per_surface_brier": axis_values,
        }


    def _mean_binary_presence_from_ensemble(self, axis_cost_ensemble) -> AxisCost:
        return AxisCost(
            x_axis=(np.asarray(axis_cost_ensemble.x_axis) > 0).astype(float).mean(axis=0),
            y_axis=(np.asarray(axis_cost_ensemble.y_axis) > 0).astype(float).mean(axis=0),
            z_axis=(np.asarray(axis_cost_ensemble.z_axis) > 0).astype(float).mean(axis=0),
        )


    def _oracle_target_binary_presence(self, target: str = "blue") -> AxisCost:
        oracle_image = np.asarray(self.oracle_image_z, dtype=float)
        if oracle_image.size == 0:
            raise ValueError("oracle_image_z is empty; cannot compute Brier score.")
        if np.nanmax(oracle_image) > 1.0:
            oracle_image = oracle_image / 255.0

        oracle_cost = self.color_mask_cost_estimator.estimate_all(image=oracle_image)
        target_cost = getattr(oracle_cost, target)
        return AxisCost(
            x_axis=(np.asarray(target_cost.x_axis) > 0).astype(float).reshape(-1),
            y_axis=(np.asarray(target_cost.y_axis) > 0).astype(float).reshape(-1),
            z_axis=(np.asarray(target_cost.z_axis) > 0).astype(float).reshape(-1),
        )


    def _brier_vector(self, probability: np.ndarray, target: np.ndarray) -> np.ndarray:
        probability = np.asarray(probability, dtype=float).reshape(-1)
        target = np.asarray(target, dtype=float).reshape(-1)
        if probability.shape != target.shape:
            raise ValueError(
                "Brier score shape mismatch: "
                f"probability.shape={probability.shape}, target.shape={target.shape}"
            )
        return (probability - target) ** 2


    def update_visibility_constraints(self, candidates: ActionCandidates):
        self.visibility_constraints.add_from_action_candidates(candidates)
