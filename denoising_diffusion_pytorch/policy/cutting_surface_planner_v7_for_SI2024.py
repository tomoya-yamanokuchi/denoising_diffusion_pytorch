


import random
import numpy as np
import torch
from PIL import Image
from scipy import stats

from denoising_diffusion_pytorch.utils.normalization import LimitsNormalizer
from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np
from denoising_diffusion_pytorch.utils.pil_utils import numpy_to_pil ,cv2_hsv_mask ,pil_to_cv2,color_range_mask
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils,load_yaml

from  denoising_diffusion_pytorch.policy.cvaeac_tmp_valid import validate,load_vaeac_model
from  denoising_diffusion_pytorch.policy.diffusion_1d_policy_utils import get_2d_image_to_1d

class cutting_surface_planner():

    def __init__(self, diffusion, trainer,sample_image_num, obs_model ,config):

        self.ensemble_obs_model     = obs_model
        self.diffusion              = diffusion
        self.trainer                = trainer
        self.sample_image_num       = sample_image_num
        self.policy_config          = config
        self.split_obs_config       = {}
        self.vaeac_model            = load_vaeac_model()


    def get_color_mask_image(self,images,mask_config):

        cost_map = {}
        for idx, val in enumerate(images):
            cost_map[val] = color_range_mask(image=images[val],mask_config=mask_config)

        return cost_map


    def get_slice_range(self,cost,axis,observation_history):

        if axis == "z":
            offset = 0
        elif axis=="x":
            offset = cost.shape[0]
        elif axis =="y":
            offset =cost.shape[0]+cost.shape[0]


        top, btm = self.find_nonzero_indices(cost)

        top_slice_range = np.arange(offset, offset+top-1).tolist()
        btm_slice_range = np.arange(offset+btm+1, offset+cost.shape[0]).tolist()


        observation_history_keys = list(observation_history.keys())
        slice_range_top = [x for x in top_slice_range if x not in observation_history_keys]
        slice_range_btm = [x for x in btm_slice_range if x not in observation_history_keys]
        if len(slice_range_top)==0 and len(slice_range_btm)==0:
            slice_range =[0]
        elif len(slice_range_top) >= len(slice_range_btm):
            slice_range = slice_range_top
        elif len(slice_range_top) <= len(slice_range_btm):
            slice_range_obj = reversed(slice_range_btm)
            slice_range     = list(slice_range_obj)
        else:
            import ipdb;ipdb.set_trace()

        return slice_range


    def get_split_idx(self,cost,axis,observation_history):

            if axis == "z":
                offset = 0
            elif axis=="x":
                offset = self.ensemble_obs_model.voxel_hander.box_array.grid_3dim_size[0]
            elif axis =="y":
                offset = self.ensemble_obs_model.voxel_hander.box_array.grid_3dim_size[0]+self.ensemble_obs_model.voxel_hander.box_array.grid_3dim_size[0]

            indices_of_ones = cost+offset

            # import ipdb;ipdb.set_trace()

            if len(indices_of_ones)==0:
                slice_range =-1
            else:

                observation_history_keys = list(observation_history.keys())
                sprit_range_ = [x for x in indices_of_ones if x not in observation_history_keys]
                if len(sprit_range_)==0:
                    slice_range =-1
                else:
                    slice_range  = random.choice(sprit_range_)
                    # print(f"split_enable | candidate : {sprit_range_}, set :{slice_range}")
                    # import ipdb;ipdb.set_trace()

            # import ipdb;ipdb.set_trace()

            return slice_range


    def find_nonzero_indices(self, arr):

        ## find indices array > cost threshold
        cost_threshold = 0
        start_index = np.argmax((arr>cost_threshold)!=0)
        end_index = len(arr) - np.argmax((arr[::-1]>cost_threshold) != 0) - 1

        return start_index, end_index


    def find_false_true_false_indices(self,lst):
        result = []
        start = -1

        for i in range(len(lst)):
            if lst[i] == False:
                if start != -1 and i - start > 1:
                    result.extend(range(start + 1, i))
                start = i
        return np.asarray(result)


    def compare_lists_for_zero(self, list1, list2):
        if len(list1) != len(list2):
            raise ValueError("リストの長さが異なります。")
        result = []
        for item1, item2 in zip(list1, list2):
            result.append(item1 == 0 and item2 == 0)

        return result


    def random_non_negative_one_element(self,input_list):
        # -1ではない要素をフィルタリング
        non_negative_elements = [element for element in input_list if element != -1]

        if not non_negative_elements:
            # raise ValueError("リストには -1 以外の要素が含まれていません。")
            return -1

        # ランダムに1つ抽出
        return random.choice(non_negative_elements)


    def find_nonzero_indices_both_1(self, lst, start_index):
        forward_index = -1
        backward_index = -1

        # Search forward from start_index + 1
        for i in range(start_index + 1, len(lst)):
            if lst[i] != 0:
                forward_index = i
                break

        # Search backward from start_index - 1
        for i in range(start_index - 1, -1, -1):
            if lst[i] != 0:
                backward_index = i
                break
        if backward_index == -1:
            backward_index=0
        elif forward_index==-1:
            forward_index=len(lst)

        return forward_index, backward_index

    def find_nonzero_indices_both_2(self, lst, start_index):
        forward_index = 0
        backward_index = 0

        # Search forward from start_index + 1
        for i in range(start_index + 1, len(lst)):
            forward_index+=lst[i]
            # if lst[i] != 0:
                # forward_index = i
                # break

        # Search backward from start_index - 1
        for i in range(start_index - 1, -1, -1):
            backward_index+=lst[i]
            # if lst[i] != 0:
                # backward_index = i
                # break

        # if backward_index == -1:
        #     backward_index=0
        # elif forward_index==-1:
        #     forward_index=len(lst)

        return forward_index, backward_index


    def get_color_mask_cost(self,image,mask_config):

        ## diffusion is trained with z axis sliced image
        self.ensemble_obs_model.cast_2d_image_to_box_color(img=image,config={"axis":"z"})
        ## get each axis ensemble images
        cast_image_z = self.ensemble_obs_model.get_2d_image(axis="z")
        cast_image_x = self.ensemble_obs_model.get_2d_image(axis="x")
        cast_image_y = self.ensemble_obs_model.get_2d_image(axis="y")

        cast_images = {     "image_x" : cast_image_x,
                            "image_y" : cast_image_y,
                            "image_z" : cast_image_z,}

        cost_map = self.get_color_mask_image(images=cast_images,mask_config=mask_config)

        ## transform mini batch image shape do not need transform axis, then permute if fixed "z"
        cost_x = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map["image_x"],permute="z").sum(3).sum(1).sum(1)
        cost_y = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map["image_y"],permute="z").sum(3).sum(1).sum(1)
        cost_z = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map["image_z"],permute="z").sum(3).sum(1).sum(1)


        cost_maps = {"x_axis":cost_x,
                     "y_axis":cost_y,
                     "z_axis":cost_z,}

        return cost_maps

    def replace_outliers_with_mean(self,data):
        # 新しい配列をコピーして作成
        cleaned_data = data.copy()
        # 各列に対して処理を行う
        for col in range(data.shape[1]):
            col_data = data[:, col]
            Q1 = np.percentile(col_data, 25)
            Q3 = np.percentile(col_data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            col_mean = np.mean(col_data[(col_data >= lower_bound) & (col_data <= upper_bound)])
            # 外れ値を平均で置換
            cleaned_data[:, col] = np.where((col_data < lower_bound) | (col_data > upper_bound), col_mean, col_data)
        return cleaned_data


    def get_outlier_removed_cost(self,data,mode,t=0):

        if mode == "remove_outliers_cal_cost_mean":
            cost_z = self.replace_outliers_with_mean(data["z_axis"]).mean(0)
            cost_x = self.replace_outliers_with_mean(data["x_axis"]).mean(0)
            cost_y = self.replace_outliers_with_mean(data["y_axis"]).mean(0)
        elif mode == "remove_outliers_cal_cost_mode":
            cost_z = stats.mode(self.replace_outliers_with_mean(data["z_axis"]), axis=0).mode[0]
            cost_x = stats.mode(self.replace_outliers_with_mean(data["x_axis"]), axis=0).mode[0]
            cost_y = stats.mode(self.replace_outliers_with_mean(data["y_axis"]), axis=0).mode[0]
        elif mode == "cal_cost_mode":
            cost_z = stats.mode(data["z_axis"], axis=0).mode[0]
            cost_x = stats.mode(data["x_axis"], axis=0).mode[0]
            cost_y = stats.mode(data["y_axis"], axis=0).mode[0]
        elif mode == "cal_cost_mean":
            cost_z = data["z_axis"].mean(0)
            cost_x = data["x_axis"].mean(0)
            cost_y = data["y_axis"].mean(0)
        elif mode == "cal_cost_mean_ucb":

            ## convert  cost to bool({0,1]})
            cost_z_bool = np.where(data["z_axis"]>0,1,0)
            cost_x_bool = np.where(data["x_axis"]>0,1,0)
            cost_y_bool = np.where(data["y_axis"]>0,1,0)

            ## set hyper param
            ucb_beta = 1.0
            # cost_lb_discount_factor = 0.99
            cost_lb_discount_factor  = self.policy_config["decision_mode"]['param']["cost_lb_discount_factor"]
            # import ipdb;ipdb.set_trace()
            # cost_lb  = 1.0-np.power(0.99,t)
            cost_lb  = 1.0-np.power(cost_lb_discount_factor,t)

            ## calculate ucb
            cost_z_ucb = cost_z_bool.mean(0)+ucb_beta*cost_z_bool.var(0)
            cost_x_ucb = cost_x_bool.mean(0)+ucb_beta*cost_x_bool.var(0)
            cost_y_ucb = cost_y_bool.mean(0)+ucb_beta*cost_y_bool.var(0)

            ## remove cost<cost_lb
            # cost_z = np.where(cost_z_ucb<=cost_lb,0,10)
            # cost_x = np.where(cost_x_ucb<=cost_lb,0,10)
            # cost_y = np.where(cost_y_ucb<=cost_lb,0,10)

            # cost_z =cost_z_ucb
            # cost_x =cost_x_ucb
            # cost_y =cost_y_ucb

            cost_z = np.where(cost_z_ucb<=cost_lb,0,10)
            cost_x = np.where(cost_x_ucb<=cost_lb,0,10)
            cost_y = np.where(cost_y_ucb<=cost_lb,0,10)



        else:
            import ipdb;ipdb.set_trace()

        return {"cost_z":cost_z,
                "cost_x":cost_x,
                "cost_y":cost_y,}

    def dump_cost(self,cost_ensembles,cost):
        if cost_ensembles is None:
            cost_ensembles = cost
        else:
            for idx, val in enumerate(cost):
                cost_ensembles[val] = np.vstack((cost_ensembles[val], cost[val]))

        return cost_ensembles


    def infer_image_by_diffusion(self,normalized_cond):


        if self.policy_config["ctrl_mode"] == "no_cond":
            # import ipdb;ipdb.set_trace()
            cond = None
        else:
            cond = {0:{ "idx":torch.where(normalized_cond>-1.0),
                        "val":normalized_cond}}

        ## infer image by diffusion model
        sample_image        = self.diffusion.model.sample(batch_size=self.sample_image_num, return_all_timesteps=True, cond = cond ).detach().cpu()
        batch_images        = (torch.permute(sample_image,(0,1,3,4,2))*255.0).numpy().astype(np.uint8)
        last_step_images    = batch_images[:,-1,:,:,:]

        return last_step_images

    def infer_image_by_vaeac(self,normalized_cond):

        if self.policy_config["ctrl_mode"] == "no_cond":
            mask_       = torch.where(normalized_cond==-1.0, torch.tensor(1),torch.tensor(1))[:1,:,:]
            observation =  normalized_cond.repeat(self.sample_image_num,1,1,1)
            mask        = mask_.repeat(self.sample_image_num,1,1,1)
        else:
            mask_       = torch.where(normalized_cond==-1.0, torch.tensor(1),torch.tensor(0))[:1,:,:]
            observation =  normalized_cond.repeat(self.sample_image_num,1,1,1)
            mask        = mask_.repeat(self.sample_image_num,1,1,1)

        data = {"image":observation,
                "mask":mask,
                "observed":observation}

        sample_image_ = validate(model=self.vaeac_model,data  = data).detach().cpu()
        sample_image = sample_image_.unsqueeze(1)

        batch_images        = (((torch.permute(sample_image,(0,1,3,4,2))+1.0)/2.0)*255.0).numpy().astype(np.uint8)

        last_step_images    = batch_images[:,-1,:,:,:]
        return last_step_images
    
    
    def infer_image_by_diffusion_1D(self, slice_image):
        

        grid_3dim       = self.ensemble_obs_model.voxel_hander.grid_side_len
        cond_image_1d_  = get_2d_image_to_1d(image=to_torch(slice_image), grid_3_dim=grid_3dim, is_shuffle=False)
        cond_image_1d   = to_np(cond_image_1d_)


        normalizer_values      = LimitsNormalizer(cond_image_1d[3:,:])
        normalizer_indices     = LimitsNormalizer(cond_image_1d[:3,:])

        voxel_indices = to_torch(normalizer_indices.normalize(cond_image_1d[:3,:]))
        voxel_values  = to_torch(normalizer_values.normalize(cond_image_1d[3:,:]))

        cond = {0:{ "idx":torch.where(voxel_values.mean(0)>-1.0),
                    "val":voxel_values,
                    "pos":voxel_indices,
                    "data":torch.cat((voxel_indices,voxel_values), dim=0)}}

        # import ipdb;ipdb.set_trace()

        sampled_seq = self.diffusion.model.sample(batch_size = self.sample_image_num, return_all_timesteps=True, cond = cond)
        # import ipdb;ipdb.set_trace()

        # get last step images [batch_size, [R,G,B,X,Y,Z], 1dim,]
        dd = sampled_seq[:,-1,:,:]
        sampled_image = self.trainer.get_1d_to_2d_images(dd).detach().cpu()

        # import ipdb;ipdb.set_trace()

        last_step_images = (torch.permute(sampled_image,(0,2,3,1))*255.0).numpy().astype(np.uint8)

        return last_step_images


    def get_optimal_act(self,slice_img_,observation_history, env2,tmp_action,iters,save_path):
        """_summary_

        Args:
            slice_img_ (np.asarray): full observed image
            observation_history (dict): observed sliced images index dict
            env2 (_type_): cutting plane observation model to get partial observation of conditional image
            tmp_action (_type_): current cutting action index
            iters (_type_): current task step index
            save_path (_type_): _description_

        Returns:
            _type_: _description_
        """

        """
        ####################################################################
        # get partial observation image for conditional image generation ###
        #####################################################################
        self.split_obs_config : Information about the slice range that will not be observed due to the split by the cutting.
                                Defaults to {}.
                                e.g.,{'[0, 2]': {'axis': 'z', 'range': [0, 2], 'offset': 0}}
        """
        obs,reward,done,info = env2.step(action_idx=tmp_action, partial_obs = self.split_obs_config)


        slice_img  = obs["sequential_obs"]["z"]
        print(f"inner action:{tmp_action}")
        save_path_ = save_path+"/conditions/"
        create_folder(save_path_)


        pil_image_save_from_numpy(obs["sequential_obs"]["z"],f"{save_path_}/seq_obs_cast_{iters}_axis_z_{0}.png")
        pil_image_save_from_numpy(obs["sequential_obs"]["y"],f"{save_path_}/seq_obs_cast_{iters}_axis_y_{0}.png")
        pil_image_save_from_numpy(obs["sequential_obs"]["x"],f"{save_path_}/seq_obs_cast_{iters}_axis_x_{0}.png")



        if self.policy_config["ctrl_mode"] != "oracle_obs" and self.policy_config["ctrl_mode"] != "random" :

            normalizer      = LimitsNormalizer(slice_img)
            normalized_cond = normalizer.normalize(slice_img).transpose(2,0,1)
            normalized_cond = to_torch(normalized_cond)


            ## conditional image generation
            if self.policy_config["infer_model"] == "vaeac":
                last_step_images = self.infer_image_by_vaeac(normalized_cond=normalized_cond)
            elif self.policy_config["infer_model"]=="diffusion":
                last_step_images = self.infer_image_by_diffusion(normalized_cond=normalized_cond)
            elif self.policy_config["infer_model"]=="diffusion_1D":
                last_step_images = self.infer_image_by_diffusion_1D(slice_img)
            else:
                import ipdb;ipdb.set_trace()


            ## save each generated images
            for k in range(last_step_images.shape[0]):
                raw_pred_image_save_path = save_path+f"/raw_pred_image/step_{iters}"
                create_folder(raw_pred_image_save_path)
                pil_image_save_from_numpy(last_step_images[k]/255.0,raw_pred_image_save_path+f"/ensemble_z_{k}.png")


            ## calculate cutting costs

            cost_b_ensembles = None
            cost_r_ensembles = None
            cost_y_ensembles = None

            for p in range(last_step_images.shape[0]):

                cost_b = self.get_color_mask_cost(last_step_images[p]/255.0,mask_config=self.policy_config["image_mask_config_b"])
                cost_r = self.get_color_mask_cost(last_step_images[p]/255.0,mask_config=self.policy_config["image_mask_config_r"])
                cost_y = self.get_color_mask_cost(last_step_images[p]/255.0,mask_config=self.policy_config["image_mask_config_y"])

                cost_b_ensembles = self.dump_cost(cost_ensembles=cost_b_ensembles,cost=cost_b)
                cost_r_ensembles = self.dump_cost(cost_ensembles=cost_r_ensembles,cost=cost_r)
                cost_y_ensembles = self.dump_cost(cost_ensembles=cost_y_ensembles,cost=cost_y)


            edited_cost_b = self.get_outlier_removed_cost(cost_b_ensembles,mode=self.policy_config["decision_mode"]["mode"],t=iters)
            cost_x_b = edited_cost_b["cost_x"]
            cost_y_b = edited_cost_b["cost_y"]
            cost_z_b = edited_cost_b["cost_z"]

            edited_cost_r = self.get_outlier_removed_cost(cost_r_ensembles,mode=self.policy_config["decision_mode"]["mode"],t=iters)
            cost_x_r = edited_cost_r["cost_x"]
            cost_y_r = edited_cost_r["cost_y"]
            cost_z_r = edited_cost_r["cost_z"]

            edited_cost_y = self.get_outlier_removed_cost(cost_y_ensembles,mode=self.policy_config["decision_mode"]["mode"],t=iters)
            cost_x_y = edited_cost_y["cost_x"]
            cost_y_y = edited_cost_y["cost_y"]
            cost_z_y = edited_cost_y["cost_z"]

            # import ipdb;ipdb.set_trace()

            ## get ensemble image
            ensemble_image   = last_step_images.mean(0)/255.0

        elif self.policy_config["ctrl_mode"] == "oracle_obs" or self.policy_config["ctrl_mode"] == "random":
            ensemble_image = self.oracle_image_z



        ## diffusion is trained with z axis sliced image
        self.ensemble_obs_model.cast_2d_image_to_box_color(img=ensemble_image,config={"axis":"z"})
        ## get each axis ensemble images
        ensemble_image_z = self.ensemble_obs_model.get_2d_image(axis="z")
        ensemble_image_x = self.ensemble_obs_model.get_2d_image(axis="x")
        ensemble_image_y = self.ensemble_obs_model.get_2d_image(axis="y")


        ensemble_images = {"image_x":ensemble_image_x,
                            "image_y":ensemble_image_y,
                            "image_z":ensemble_image_z,}



        #############################################################
        ## get each segmented images
        ##############################################################

        cost_map_yellow = self.get_color_mask_image(images=ensemble_images,mask_config=self.policy_config["image_mask_config_y"])
        cost_map_blue   = self.get_color_mask_image(images=ensemble_images,mask_config=self.policy_config["image_mask_config_b"])
        cost_map_red    = self.get_color_mask_image(images=ensemble_images,mask_config=self.policy_config["image_mask_config_r"])


        # pil_image_save_from_numpy(ensemble_image_z,"./ensemble_z.png")
        # pil_image_save_from_numpy(ensemble_image_x,"./ensemble_x.png")
        # pil_image_save_from_numpy(ensemble_image_y,"./ensemble_y.png")


        # import ipdb;ipdb.set_trace()

        #############################################################
        ## get cost of each segmented images for oracle obs and random
        ##############################################################
        ## transform mini batch image shape do not need transform axis, then permute if fixed "z"
        # cost_x_y = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_yellow["image_x"],permute="z").sum(3).sum(1).sum(1)
        # cost_y_y = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_yellow["image_y"],permute="z").sum(3).sum(1).sum(1)
        # cost_z_y = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_yellow["image_z"],permute="z").sum(3).sum(1).sum(1)


        ## transform mini batch image shape do not need transform axis, then permute if fixed "z"
        # cost_x_b = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_blue["image_x"],permute="z").sum(3).sum(1).sum(1)
        # cost_y_b = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_blue["image_y"],permute="z").sum(3).sum(1).sum(1)
        # cost_z_b = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_blue["image_z"],permute="z").sum(3).sum(1).sum(1)


        ## transform mini batch image shape do not need transform axis, then permute if fixed "z"
        # cost_x_r = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_red["image_x"],permute="z").sum(3).sum(1).sum(1)
        # cost_y_r = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_red["image_y"],permute="z").sum(3).sum(1).sum(1)
        # cost_z_r = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_red["image_z"],permute="z").sum(3).sum(1).sum(1)




        ###############################################
        ## get split position for parts splitting
        ##################################################
        # split_candidate_x = self.find_false_true_false_indices(self.compare_lists_for_zero(cost_x_b,cost_x_r))
        # split_candidate_y = self.find_false_true_false_indices(self.compare_lists_for_zero(cost_y_b,cost_y_r))
        # split_candidate_z = self.find_false_true_false_indices(self.compare_lists_for_zero(cost_z_b,cost_z_r))

        # split_cost_th_candidate  ={ "x":np.unique(cost_x_b+cost_x_r)[1],
        #                             "y":np.unique(cost_y_b+cost_y_r)[1],
        #                             "z":np.unique(cost_z_b+cost_z_r)[1],}

        # split_cost_th = split_cost_th_candidate[min(split_cost_th_candidate, key=split_cost_th_candidate.get)]

        split_cost_th = 0.0
        split_candidate_x = self.find_false_true_false_indices(((cost_x_b+cost_x_r).clip(split_cost_th,None)-split_cost_th)==0.0)
        split_candidate_y = self.find_false_true_false_indices(((cost_y_b+cost_y_r).clip(split_cost_th,None)-split_cost_th)==0.0)
        split_candidate_z = self.find_false_true_false_indices(((cost_z_b+cost_z_r).clip(split_cost_th,None)-split_cost_th)==0.0)

        split_index_y     = self.get_split_idx(split_candidate_y,axis="y",observation_history=observation_history)
        split_index_x     = self.get_split_idx(split_candidate_x,axis="x",observation_history=observation_history)
        split_index_z     = self.get_split_idx(split_candidate_z,axis="z",observation_history=observation_history)

        split_candidate = [split_index_x,split_index_y,split_index_z]
        split_index     = self.random_non_negative_one_element(split_candidate)


        if split_index != -1 :

            if split_index <=15:
                split_index_ = split_index
                offset = 0
                axis = "z"
                cost_ = cost_z_b
            elif 15<split_index<=31:
                split_index_ = split_index-16
                cost_ = cost_x_b
                offset = 16
                axis = "x"
            else:
                split_index_ = split_index-32
                cost_ = cost_y_b
                offset = 32
                axis = "y"


            forward_index ,back_ward_index = self.find_nonzero_indices_both_1(lst=cost_,start_index = split_index_)



            if np.abs(split_index-forward_index)<(split_index-back_ward_index):
                split_range = (np.asarray([back_ward_index, split_index_])).tolist()
            elif np.abs(split_index-forward_index)>(split_index-back_ward_index):
                split_range = (np.asarray([split_index,forward_index])).tolist()
            else:
                # import ipdb;ipdb.set_trace()

                forward_index ,back_ward_index = self.find_nonzero_indices_both_2(lst=cost_,start_index = split_index_)

                if  forward_index>back_ward_index:
                    split_range = [0,split_index_]
                elif forward_index<back_ward_index:
                    split_range = [split_index_,cost_.shape[0]]

                # import ipdb;ipdb.set_trace()



            self.split_obs_config[str(split_range)] ={"axis":axis,
                                                      "range":split_range,
                                                      "offset":offset}


            # for idx,val in enumerate(self.split_obs_config):
            #     for i in range(self.split_obs_config[val]["range"][0],self.split_obs_config[val]["range"][1]):
            #         offset = self.split_obs_config[val]["offset"]
            #         loc = offset+i
            #         observation_history[loc]={"axis":self.split_obs_config[val]["axis"],
            #                                        "loc":i}

            print(f"split_range:{self.split_obs_config}")




        #####################################################################
        # get slice range for pats remove
        #####################################################################


        if iters == 0 and split_index==-1:
            print("fail to first split")
            # slice_range_z =self.get_slice_range(cost=cost_z_b+cost_z_r,axis="z",observation_history=observation_history)
            # slice_range_x =self.get_slice_range(cost=cost_x_b+cost_x_r,axis="x",observation_history=observation_history)
            # slice_range_y =self.get_slice_range(cost=cost_y_b+cost_y_r,axis="y",observation_history=observation_history)
            slice_range_z =self.get_slice_range(cost=cost_z_b,axis="z",observation_history=observation_history)
            slice_range_x =self.get_slice_range(cost=cost_x_b,axis="x",observation_history=observation_history)
            slice_range_y =self.get_slice_range(cost=cost_y_b,axis="y",observation_history=observation_history)
            slice_range_candidates = [slice_range_z,slice_range_x,slice_range_y]
            # 最も長いリスト
            slice_range = max(slice_range_candidates, key=len)

            # import ipdb;ipdb.set_trace()

        # elif split_index!=-1:
        elif split_index ==-1000:
            print(f"iter:{iters},split_idx:{split_index}")
            slice_range = [split_index]

        else:
            slice_range_z =self.get_slice_range(cost=cost_z_b,axis="z",observation_history=observation_history)
            slice_range_x =self.get_slice_range(cost=cost_x_b,axis="x",observation_history=observation_history)
            slice_range_y =self.get_slice_range(cost=cost_y_b,axis="y",observation_history=observation_history)
            slice_range_candidates = [slice_range_z,slice_range_x,slice_range_y]
            # 最も長いリスト
            slice_range = max(slice_range_candidates, key=len)


            # import ipdb;ipdb.set_trace()


        cost_x = cost_x_b
        cost_y = cost_y_b
        cost_z = cost_z_b

        # cost_x = cost_x_b+cost_x_r
        # cost_y = cost_y_b+cost_y_r
        # cost_z = cost_z_b+cost_z_r


        # slice_range_z =self.get_slice_range(cost=cost_z,axis="z",observation_history=observation_history)
        # slice_range_x =self.get_slice_range(cost=cost_x,axis="x",observation_history=observation_history)
        # slice_range_y =self.get_slice_range(cost=cost_y,axis="y",observation_history=observation_history)

        # slice_range_z =self.get_slice_range(cost=cost_z_b+cost_z_r,axis="z",observation_history=observation_history)
        # slice_range_x =self.get_slice_range(cost=cost_x_b+cost_x_r,axis="x",observation_history=observation_history)
        # slice_range_y =self.get_slice_range(cost=cost_y_b+cost_y_r,axis="y",observation_history=observation_history)

        # slice_range_z =self.get_slice_range(cost=cost_z_b,axis="z",observation_history=observation_history)
        # slice_range_x =self.get_slice_range(cost=cost_x_b,axis="x",observation_history=observation_history)
        # slice_range_y =self.get_slice_range(cost=cost_y_b,axis="y",observation_history=observation_history)

        # slice_range_candidates = [slice_range_z,slice_range_x,slice_range_y]
        # # 最も長いリスト
        # slice_range = max(slice_range_candidates, key=len)



        #################################################
        ## random policy setting
        ################################################
        if self.policy_config["ctrl_mode"] == "random" :
            ### random action
            # random_number = random.randint(1, 5)
            # random_slice_range_candidate = random.sample(range(0, 47 + 1), random_number)
            # observation_history_keys = list(observation_history.keys())
            # slice_range = [x for x in random_slice_range_candidate if x not in observation_history_keys]

            random_number = random.randint(1, 3)
            random_slice_range_candidate = [i for i in range(0, 47)]
            observation_history_keys = list(observation_history.keys())
            slice_range_candidates = [x for x in random_slice_range_candidate if x not in observation_history_keys]
            slice_range = random.sample(slice_range_candidates, random_number)

            # print(random_number)
            # print(random_slice_range_candidate)
            print(slice_range)

            # import ipdb;ipdb.set_trace()
            # action_idx  =  random.choice(list(sort_action_candidate.keys()))
            # slice_range = [action_idx]






        #######################################################################################################
        #### not necessary part
        #######################################################################################################

        cost_data = {"z":cost_z,
                     "x":cost_x,
                     "y":cost_y}

        cost_map = {}
        h =0
        for idx, val in enumerate(cost_data):
            for j in range(len(cost_data[val])):
                cost_map.update({h:{"axis":val,"loc":j,"cost":cost_data[val][j]}})
                h+=1

        ##  remove already observed actions
        un_observed_action_keys = set(cost_map.keys())-set(observation_history.keys())
        action_candidate = {key: cost_map[key] for key in un_observed_action_keys}


        # 第2階層の"Age"キーで昇順でソートする。
        sort_action_candidate = dict(sorted(action_candidate.items(),
                key=lambda x:x[1]['cost'],
                reverse=False))
        ######################################################################################################3

        infos ={"ensemble_image":{"z":ensemble_image_z,
                                  "x":ensemble_image_x,
                                  "y":ensemble_image_y,}}





        print(f"split_candidate :{split_candidate}, spit_idx : {split_index}, slice_range :{slice_range}")





        return slice_range, sort_action_candidate, infos



    def set_oracle_obs(self,oracle_obs_image):
        self.oracle_image_z = oracle_obs_image