import random
import numpy as np
import torch

from denoising_diffusion_pytorch.utils.normalization import LimitsNormalizer
from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy, pil_image_load_to_numpy
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils,load_yaml

# from  denoising_diffusion_pytorch.policy.cvaeac_tmp_valid import validate,load_vaeac_model
from denoising_diffusion_pytorch.utils.vaeac_utils.vaeac_utils import vaeac_validate
from denoising_diffusion_pytorch.policy.diffusion_1d_policy_utils import get_2d_image_to_1d
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler, dismantling_env


from denoising_diffusion_pytorch.cost.color_mask_cost_estimator import ColorMaskCostEstimator
from denoising_diffusion_pytorch.cost.segmentation_cost_collector import SegmentationCostCollector
from denoising_diffusion_pytorch.policy.decision.decision_aggregator import DecisionAggregator
from .ensemble_image_builder import EnsembleImageBuilder
from .types import PolicyConfig, AxisCostSet
from .planning.action_selection.action_candidates_selector import ActionCandidatesSelector
from .planning.visibility.visibility_constraint_set import VisibilityConstraintSet
from .planning.action_definition.action_candidates import ActionCandidates


class cutting_surface_planner():

    def __init__(self,
            inferencer, # dissufion / vaeac
            trainer,
            obs_model                 : voxel_cut_handler,
            policy_config             : PolicyConfig,
            action_candidates_selector: ActionCandidatesSelector,
        ):
        self.ensemble_obs_model        = obs_model
        self.inferencer                = inferencer
        self.trainer                   = trainer
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


    def infer_image_by_conditional_diffusion(self,normalized_cond):
        if self.policy_config.control.mode == "no_cond":
            cond = None
            normalized_cond[:] = -1.0
            mask = normalized_cond.repeat(self.sample_image_num,1,1,1)
        else:
            mask_tmp = (normalized_cond != -1.0).any(dim=0) # どこを観測している/していないかを示すラベル
            # dim=0 に対して .any() を使うことで、少なくとも1チャンネルが -1.0 以外ならTrue。
            # 結果的に、3チャンネルすべてが -1.0 のピクセルだけ False になります。
            cond = {
                0: {
                    "idx": torch.where(mask_tmp),
                    "val": normalized_cond
                    }
                    }
            # cond = {0:{ "idx":torch.where(normalized_cond>-1.0),
                        # "val":normalized_cond}}
            mask =  normalized_cond.repeat(self.sample_image_num,1,1,1)


        omega = self.policy_config.inference.guidance_scale
        ## infer image by diffusion model
        sample_image        = self.inferencer.ema_model.sample(batch_size=self.sample_image_num, return_all_timesteps=True, cond = cond, mask= mask, omega = omega).detach().cpu()

        '''
        sample_image: (batch, diffusion step, width, height, channel) かCWHとか
         - batchはsampleサイズを意味する（論文でいうところのM=32）
         - return_all_timesteps：DDIMの途中のdenoisingの結果も返す
        '''

        # batch_images        = (torch.permute(sample_image,(0,1,3,4,2))*255.0).numpy().astype(np.uint8)
        batch_images        = (torch.permute(sample_image,(0,1,3,4,2))*255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)
        last_step_images    = batch_images[:,-1,:,:,:]

        return last_step_images # (batch, 1, width, height, channel)



    def infer_image_by_vaeac(self,normalized_cond):

        # self.inferencer.training= False
        self.inferencer.eval()

        if self.policy_config.control.mode == "no_cond":
            # mask_       = torch.where(normalized_cond==-1.0, torch.tensor(1),torch.tensor(1))[:1,:,:]
            mask_       = torch.where((normalized_cond == -1.0).all(dim=0), torch.tensor(1),torch.tensor(1))
            observation =  normalized_cond.repeat(self.sample_image_num,1,1,1)
            mask        = mask_.repeat(self.sample_image_num,1,1,1)
        else:
            # mask_       = torch.where(normalized_cond==-1.0, torch.tensor(1),torch.tensor(0))[:1,:,:]
            mask_       = torch.where((normalized_cond == -1.0).all(dim=0), torch.tensor(1),torch.tensor(0))
            observation =  normalized_cond.repeat(self.sample_image_num,1,1,1)
            mask        = mask_.repeat(self.sample_image_num,1,1,1)

        data = {"image":observation,
                "mask":mask,
                "observed":observation}

        # sample_iage_ = validate(model=self.vaeac_model,data  = data).detach().cpu()
        sample_image_ = vaeac_validate(model=self.inferencer,data  = data).detach().cpu()
        sample_image = sample_image_.unsqueeze(1)

        # batch_images        = (((torch.permute(sample_image,(0,1,3,4,2))+1.0)/2.0)*255.0).numpy().astype(np.uint8)
        batch_images        = (((torch.permute(sample_image,(0,1,3,4,2))+1.0)/2.0)*255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)


        last_step_images    = batch_images[:,-1,:,:,:]
        return last_step_images


    def infer_image_by_diffusion_1D(self, slice_image):
        grid_3dim       = self.ensemble_obs_model.voxel_hander.grid_side_len
        cond_image_1d_  = get_2d_image_to_1d(image=to_torch(slice_image), grid_3_dim=grid_3dim, is_shuffle=False)
        cond_image_1d   = to_np(cond_image_1d_)


        normalizer_values      = LimitsNormalizer(cond_image_1d[3:,:])
        normalizer_indices     = LimitsNormalizer(cond_image_1d[:3,:])

        voxel_values  = to_torch(normalizer_values.normalize(cond_image_1d[3:,:]))
        voxel_indices = to_torch(normalizer_indices.normalize(cond_image_1d[:3,:]))

        cond = {0:{ "idx":torch.where(voxel_values.mean(0)>-1.0),
                    "val":voxel_values,
                    "pos":voxel_indices,
                    "data":torch.cat((voxel_indices,voxel_values), dim=0)}}

        # import ipdb;ipdb.set_trace()

        # sampled_seq = self.inferencer.model.sample(batch_size = self.sample_image_num, return_all_timesteps=True, cond = cond)
        sampled_seq = self.inferencer.ema_model.sample(batch_size = self.sample_image_num, return_all_timesteps=True, cond = cond)

        # import ipdb;ipdb.set_trace()

        # get last step images [batch_size, [R,G,B,X,Y,Z], 1dim,]
        dd = sampled_seq[:,-1,:,:]
        sampled_image = self.trainer.get_1d_to_2d_images(dd).detach().cpu()

        # import ipdb;ipdb.set_trace()

        # last_step_images = (torch.permute(sampled_image,(0,2,3,1))*255.0).numpy().astype(np.uint8)
        last_step_images = (torch.permute(sampled_image,(0,2,3,1))*255.0).clamp(0, 255).numpy().astype(np.uint8)


        return last_step_images


    def get_optimal_act(self,
            observation_history              : dict,
            env2                             : dismantling_env,
            last_executed_global_action_index: int,
            iters                            : int,
            save_path                        : str,
        ):
        step_results = env2.step(
            action_idx  = last_executed_global_action_index,
            partial_obs = self.visibility_constraints.to_legacy_partial_obs()
        )
        obs        = step_results.observation
        slice_img  = obs.axis_images.z # 学習とテストで固定させておく

        # import ipdb; ipdb.set_trace()

        cond_image_save_path = save_path+"/conditions/"
        create_folder(cond_image_save_path)
        cond_ds_image_save_path = save_path+"/conditions_ds/"
        create_folder(cond_ds_image_save_path)

        # pil_image_save_from_numpy(obs.axis_images.z"],f"{cond_image_save_path}/seq_obs_cast_{iters}_axis_z_{0}.png")
        # pil_image_save_from_numpy(obs["sequential_obs"]["y"],f"{cond_image_save_path}/seq_obs_cast_{iters}_axis_y_{0}.png")
        # pil_image_save_from_numpy(obs["sequential_obs"]["x"],f"{cond_image_save_path}/seq_obs_cast_{iters}_axis_x_{0}.png")


        normalizer      = LimitsNormalizer(slice_img)
        normalized_cond = normalizer.normalize(slice_img).transpose(2,0,1)
        normalized_cond = to_torch(normalized_cond)

        ## conditional image generation
        if self.policy_config.inference.model == "vaeac":
            last_step_images = self.infer_image_by_vaeac(normalized_cond=normalized_cond)
        elif self.policy_config.inference.model=="conditional_diffusion":
            last_step_images = self.infer_image_by_conditional_diffusion(normalized_cond=normalized_cond)
        elif self.policy_config.inference.model=="diffusion_1D":
            last_step_images = self.infer_image_by_diffusion_1D(slice_img)
        else:
            import ipdb;ipdb.set_trace()

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
        selected_candidates = selection.slice_range
        self.update_visibility_constraints(selected_candidates)

        # ---- log ----
        cost_map_logs["slice_candidate"] = {
            "candidate_x": None if selection.slice_candidates.x is None else selection.slice_candidates.x.to_list(),
            "candidate_y": None if selection.slice_candidates.y is None else selection.slice_candidates.y.to_list(),
            "candidate_z": None if selection.slice_candidates.z is None else selection.slice_candidates.z.to_list(),
        }
        cost_map_logs["slice_range"] = (
            None if selected_candidates is None else selected_candidates.to_list()
        )
        pickle_utils().save(dataset=cost_map_logs, save_path=save_path+f"/{iters}_cost_map_logs.pickle")


        # =================================================
        infos                 = {"ensemble_image": ensemble_images}
        slice_range           = None if selected_candidates is None else selected_candidates.to_list()
        sort_action_candidate = None

        return slice_range, sort_action_candidate, infos


    def set_oracle_obs(self,oracle_obs_image):
        self.oracle_image_z = oracle_obs_image


    def update_visibility_constraints(self, candidates: ActionCandidates):
        self.visibility_constraints.add_from_action_candidates(candidates)
