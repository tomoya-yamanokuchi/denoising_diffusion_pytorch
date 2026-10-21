# policy_factory.py


import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from ..types.policy_assets import PolicyAssets
from denoising_diffusion_pytorch.policy.cutting_surface_planner import cutting_surface_planner
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler
from denoising_diffusion_pytorch.policy.planning.candidate_building.active_range_detector import ActiveRangeDetector
from denoising_diffusion_pytorch.policy.planning.candidate_building.local_candidate_range_factory import LocalCandidateRangeFactory
from denoising_diffusion_pytorch.policy.planning.candidate_building.observed_action_pruner import ObservedActionPruner
from denoising_diffusion_pytorch.policy.planning.candidate_building.axis_candidate_selection_policy import AxisCandidateSelectionPolicy
from denoising_diffusion_pytorch.policy.planning.candidate_building.axis_candidate_range_builder import AxisCandidateRangeBuilder
from denoising_diffusion_pytorch.policy.planning.action_selection.selection_policy import SelectionPolicy
from denoising_diffusion_pytorch.policy.planning.action_selection.action_candidates_selector import ActionCandidatesSelector
from denoising_diffusion_pytorch.policy.planning.candidate_building.action_candidate_building_coordinator import ActionCandidateBuildingCoordinator
from denoising_diffusion_pytorch.policy.inference.slice_image_inferencer_factory import SliceImageInferencerFactory
from denoising_diffusion_pytorch.policy.planning.action_selection.random_contiguous_selection_policy import RandomContiguousSelectionPolicy
from denoising_diffusion_pytorch.policy.planning.candidate_building.full_action_space_candidate_coordinator import FullActionSpaceCandidateCoordinator



class PolicyFactory:
    def __init__(self, assets: PolicyAssets):
        self._assets = assets


    def _load_i2sb_x1(self, case_spec):
        cfg_train = self._assets.trained_assets.cfg_train
        products = cfg_train.dataset.products

        case_name = str(case_spec.name).lower()

        matches = [
            product
            for product in products
            if case_name.startswith(str(product.name).lower() + "_")
            or case_name == str(product.name).lower()
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Cannot uniquely identify I2SB product for "
                f"case={case_spec.name}. "
                f"Available products: {[p.name for p in products]}"
            )

        product = matches[0]
        image_size = int(cfg_train.dataset.image_size)

        transform = T.Compose([
            T.Resize(
                image_size,
                interpolation=InterpolationMode.NEAREST,
            ),
            T.ToTensor(),
        ])

        with Image.open(str(product.exterior_only_path)) as image:
            x1 = transform(image.convert("RGB"))

        x1 = (x1 * 2.0 - 1.0).float()

        print(
            f"[PolicyFactory] I2SB X1 loaded: "
            f"product={product.name}, "
            f"path={product.exterior_only_path}, "
            f"shape={tuple(x1.shape)}"
        )

        return x1


    def create(self, obs_model: voxel_cut_handler, case_spec=None):
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

        action_candidates_selector = self._build_action_candidates_selector(
            axis_candidate_range_builder = axis_candidate_range_builder,
            side_length                  = side_length,
        )

        slice_image_inferencer = SliceImageInferencerFactory.build(
            inferencer    = self._assets.trained_assets.inferencer,
            policy_config = self._assets.policy_config,
        )


        policy = cutting_surface_planner(
            obs_model=obs_model,
            slice_image_inferencer=slice_image_inferencer,
            policy_config=self._assets.policy_config,
            action_candidates_selector=action_candidates_selector,
        )

        policy.normalized_x1 = None

        if self._assets.policy_config.inference.model == "i2sb":
            if case_spec is None:
                raise ValueError("I2SB evaluation requires case_spec.")

            policy.normalized_x1 = self._load_i2sb_x1(case_spec)

        return policy



    def _build_action_candidates_selector(
        self,
        *,
        axis_candidate_range_builder,
        side_length: int,
    ) -> ActionCandidatesSelector:
        control_cfg = self._assets.policy_config.control

        if control_cfg.mode == "random":
            candidate_coordinator = FullActionSpaceCandidateCoordinator(
                expected_side_length=side_length,
            )
            selection_policy = RandomContiguousSelectionPolicy(
                seed=control_cfg.random_seed,
                max_range_length=control_cfg.random_max_range_length,
            )
        else:
            candidate_coordinator = ActionCandidateBuildingCoordinator(
                candidate_builder=axis_candidate_range_builder,
                expected_side_length=side_length,
            )
            selection_policy = SelectionPolicy()

        return ActionCandidatesSelector(
            candidate_coordinator=candidate_coordinator,
            selection_policy=selection_policy,
            side_length=side_length,
        )
