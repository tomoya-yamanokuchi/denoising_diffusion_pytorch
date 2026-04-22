


import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from scipy import stats
import cv2


from denoising_diffusion_pytorch.utils.normalization import LimitsNormalizer
from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np
from denoising_diffusion_pytorch.utils.pil_utils import numpy_to_pil ,cv2_hsv_mask ,pil_to_cv2,color_range_mask,color_mask
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy, pil_image_load_to_numpy
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils,load_yaml

# from  denoising_diffusion_pytorch.policy.cvaeac_tmp_valid import validate,load_vaeac_model
from  denoising_diffusion_pytorch.utils.vaeac_utils.vaeac_utils import vaeac_validate
from  denoising_diffusion_pytorch.policy.diffusion_1d_policy_utils import get_2d_image_to_1d


class cutting_surface_planner():

    def __init__(self, diffusion, trainer,sample_image_num, obs_model ,config):

        self.ensemble_obs_model     = obs_model
        self.diffusion              = diffusion
        self.trainer                = trainer
        self.sample_image_num       = sample_image_num
        self.policy_config          = config
        self.split_obs_config       = {}






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

        cost_map = self.get_color_mask_image(images=cast_images,mask_config=mask_config) # 軸の区切りの概念なしのdiffusionが出力する生画像

        ## transform mini batch image shape do not need transform axis, then permute if fixed "z"
        cost_x = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map["image_x"],permute="z").sum(3).sum(1).sum(1) # 画像を軸のスライス単位に区切って、コスト計算する
        cost_y = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map["image_y"],permute="z").sum(3).sum(1).sum(1) # 画像を軸のスライス単位に区切って、コスト計算する
        cost_z = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map["image_z"],permute="z").sum(3).sum(1).sum(1) # 画像を軸のスライス単位に区切って、コスト計算する


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
            cost_z_bool = np.where(data["z_axis"]>0,1,0) #１ピクセルでもしきい値以上での目標部品が存在するものとして計算
            cost_x_bool = np.where(data["x_axis"]>0,1,0) #１ピクセルでもしきい値以上での目標部品が存在するものとして計算
            cost_y_bool = np.where(data["y_axis"]>0,1,0) #１ピクセルでもしきい値以上での目標部品が存在するものとして計算

            ## set hyper param
            ucb_beta = 1.0
            # cost_lb_discount_factor = 0.99
            cost_lb_discount_factor  = self.policy_config["decision_mode"]['param']["cost_lb_discount_factor"]
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

        elif mode == "clip_ucb":

            ## convert  cost to bool({0,1]})
            cost_z_bool = np.where(data["z_axis"]>0,1,0)
            cost_x_bool = np.where(data["x_axis"]>0,1,0)
            cost_y_bool = np.where(data["y_axis"]>0,1,0)

            ucb_beta = 1.0
            cost_lb  = self.policy_config["decision_mode"]['param']["ucb_lb"]

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


        elif mode == "clip_ucb_raw":

            ## convert  cost to bool({0,1]})
            cost_z_bool = np.where(data["z_axis"]>0,1,0)
            cost_x_bool = np.where(data["x_axis"]>0,1,0)
            cost_y_bool = np.where(data["y_axis"]>0,1,0)

            ucb_beta = 1.0
            cost_lb  = self.policy_config["decision_mode"]['param']["ucb_lb"]

            ## calculate ucb
            cost_z_ucb = cost_z_bool.mean(0)+ucb_beta*cost_z_bool.std(0)
            cost_x_ucb = cost_x_bool.mean(0)+ucb_beta*cost_x_bool.std(0)
            cost_y_ucb = cost_y_bool.mean(0)+ucb_beta*cost_y_bool.std(0)

            ## remove cost<cost_lb
            # cost_z = np.where(cost_z_ucb<=cost_lb,0,10)
            # cost_x = np.where(cost_x_ucb<=cost_lb,0,10)
            # cost_y = np.where(cost_y_ucb<=cost_lb,0,10)

            # cost_z =cost_z_ucb
            # cost_x =cost_x_ucb
            # cost_y =cost_y_ucb

            cost_z = np.where(cost_z_ucb<=cost_lb,0,10) #10に意味はない、あるかないかを示すための定数
            cost_x = np.where(cost_x_ucb<=cost_lb,0,10) #10に意味はない、あるかないかを示すための定数
            cost_y = np.where(cost_y_ucb<=cost_lb,0,10) #10に意味はない、あるかないかを示すための定数



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
            # cond = {0:{ "idx":torch.where(normalized_cond>-1.0),
            #             "val":normalized_cond}}
            mask = (normalized_cond != -1.0).any(dim=0)
            cond = {
                0: {
                    "idx": torch.where(mask),
                    "val": normalized_cond
                    }
                   }

        ## infer image by diffusion model
        # sample_image        = self.diffusion.model.sample(batch_size=self.sample_image_num, return_all_timesteps=True, cond = cond ).detach().cpu()
        sample_image        = self.diffusion.ema_model.sample(batch_size=self.sample_image_num, return_all_timesteps=True, cond = cond ).detach().cpu()


        # sample_image_1        = self.diffusion.model.sample(batch_size=self.sample_image_num, return_all_timesteps=True, cond = cond ).detach().cpu()
        # sample_image_2        = self.diffusion.model.sample(batch_size=self.sample_image_num, return_all_timesteps=True, cond = cond ).detach().cpu()
        # sample_image_3        = self.diffusion.model.sample(batch_size=self.sample_image_num, return_all_timesteps=True, cond = cond ).detach().cpu()
        # sample_image_4        = self.diffusion.model.sample(batch_size=self.sample_image_num, return_all_timesteps=True, cond = cond ).detach().cpu()
        # sample_image = torch.cat((sample_image_1,sample_image_2,sample_image_4),dim=0)

        batch_images        = (torch.permute(sample_image,(0,1,3,4,2))*255.0).numpy().astype(np.uint8)
        last_step_images    = batch_images[:,-1,:,:,:]

        # import ipdb;ipdb.set_trace()

        # # Resize & ToTensor
        # env_img_dim = self.ensemble_obs_model.init_imgs_y.shape[0]
        # transform_func = transforms.Compose(
        #                                     [
        #                                         transforms.Resize((env_img_dim, env_img_dim), interpolation=InterpolationMode.NEAREST),
        #                                         # transforms.Resize((env_img_dim, env_img_dim)),
        #                                         transforms.ToTensor(),])
        # # 各画像に適用
        # transformed_imgs = [transform_func(Image.fromarray(img)) for img in last_step_images]
        # # バッチTensorにまとめる（B, C, H, W）
        # batch_tensor = torch.stack(transformed_imgs)  # shape: (32, 3, 512, 512)
        # last_step_images = (batch_tensor.permute(0,2,3,1)*255.0).numpy().astype(np.uint8)

        return last_step_images



    def infer_image_by_conditional_diffusion(self,normalized_cond):


        if self.policy_config["ctrl_mode"] == "no_cond":
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

        # import ipdb; ipdb.set_trace()
        omega = self.policy_config["cfg_omega"]
        ## infer image by diffusion model
        # sample_image        = self.diffusion.model.sample(batch_size=self.sample_image_num, return_all_timesteps=True, cond = cond, mask= mask).detach().cpu()
        # sample_image        = self.diffusion.ema_model.sample(batch_size=self.sample_image_num, return_all_timesteps=True, cond = cond, mask= mask, omega = 0.2).detach().cpu()
        sample_image        = self.diffusion.ema_model.sample(batch_size=self.sample_image_num, return_all_timesteps=True, cond = cond, mask= mask, omega = omega).detach().cpu()
        # sample_image        = self.diffusion.model.sample(batch_size=self.sample_image_num, return_all_timesteps=True, cond = cond, mask= mask).detach().cpu()

        '''
        sample_image: (batch, diffusion step, width, height, channel) かCWHとか
         - batchはsampleサイズを意味する（論文でいうところのM=32）
         - return_all_timesteps：DDIMの途中のdenoisingの結果も返す
        '''

        # batch_images        = (torch.permute(sample_image,(0,1,3,4,2))*255.0).numpy().astype(np.uint8)
        batch_images        = (torch.permute(sample_image,(0,1,3,4,2))*255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)
        last_step_images    = batch_images[:,-1,:,:,:]


        # Resize & ToTensor
        # env_img_dim = self.ensemble_obs_model.init_imgs_y.shape[0]
        # transform_func = transforms.Compose(
        #                                     [
        #                                         transforms.Resize((env_img_dim, env_img_dim), interpolation=InterpolationMode.NEAREST),
        #                                         # transforms.Resize((env_img_dim, env_img_dim)),
        #                                         transforms.ToTensor(),])
        # # 各画像に適用
        # transformed_imgs = [transform_func(Image.fromarray(img)) for img in last_step_images]
        # # バッチTensorにまとめる（B, C, H, W）
        # batch_tensor = torch.stack(transformed_imgs)  # shape: (32, 3, 512, 512)
        # last_step_images = (batch_tensor.permute(0,2,3,1)*255.0).numpy().astype(np.uint8)

        return last_step_images # (batch, 1, width, height, channel)




    def infer_image_by_vaeac(self,normalized_cond):

        # self.diffusion.training= False
        self.diffusion.eval()

        if self.policy_config["ctrl_mode"] == "no_cond":
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
        sample_image_ = vaeac_validate(model=self.diffusion,data  = data).detach().cpu()
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

        # sampled_seq = self.diffusion.model.sample(batch_size = self.sample_image_num, return_all_timesteps=True, cond = cond)
        sampled_seq = self.diffusion.ema_model.sample(batch_size = self.sample_image_num, return_all_timesteps=True, cond = cond)

        # import ipdb;ipdb.set_trace()

        # get last step images [batch_size, [R,G,B,X,Y,Z], 1dim,]
        dd = sampled_seq[:,-1,:,:]
        sampled_image = self.trainer.get_1d_to_2d_images(dd).detach().cpu()

        # import ipdb;ipdb.set_trace()

        # last_step_images = (torch.permute(sampled_image,(0,2,3,1))*255.0).numpy().astype(np.uint8)
        last_step_images = (torch.permute(sampled_image,(0,2,3,1))*255.0).clamp(0, 255).numpy().astype(np.uint8)


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

        # import ipdb;ipdb.set_trace()

        ###################################################################################
        ## prior_based_ep_00がなかったときの実装
        ####################################################################################
        # obs,reward,done,info = env2.step(action_idx=tmp_action, partial_obs = self.split_obs_config)


        ################################################################################
        ## actionがprior_based_ep_00のときは，policy partial observation用のenv2を進めずに，
        ## 観測画像が黒＝すべての領域が未観測としてそれぞれの方策を使う
        ################################################################################
        if tmp_action == "prior_based_ep_00":
            obs = env2.get_obs()
        else:
            obs,reward,done,info = env2.step(action_idx=tmp_action, partial_obs = self.split_obs_config)


        slice_img  = obs["sequential_obs"]["z"] #学習とテストで固定させておく
        print(f"inner action:{tmp_action}")

        cond_image_save_path = save_path+"/conditions/"
        create_folder(cond_image_save_path)
        cond_ds_image_save_path = save_path+"/conditions_ds/"
        create_folder(cond_ds_image_save_path)


        pil_image_save_from_numpy(obs["sequential_obs"]["z"],f"{cond_image_save_path}/seq_obs_cast_{iters}_axis_z_{0}.png")
        # pil_image_save_from_numpy(obs["sequential_obs"]["y"],f"{cond_image_save_path}/seq_obs_cast_{iters}_axis_y_{0}.png")
        # pil_image_save_from_numpy(obs["sequential_obs"]["x"],f"{cond_image_save_path}/seq_obs_cast_{iters}_axis_x_{0}.png")



        if self.policy_config["ctrl_mode"] != "oracle_obs" and self.policy_config["ctrl_mode"] != "random" :

            # import ipdb;ipdb.set_trace()
            if  tmp_action == "prior_based_ep_00":
                normalized_cond = to_torch((np.ones_like(slice_img)*-1.0).transpose(2,0,1))
            else:
                normalizer      = LimitsNormalizer(slice_img)
                normalized_cond = normalizer.normalize(slice_img).transpose(2,0,1)
                normalized_cond = to_torch(normalized_cond)

            ## conditional image generation
            if self.policy_config["infer_model"] == "vaeac":
                last_step_images = self.infer_image_by_vaeac(normalized_cond=normalized_cond)
            elif self.policy_config["infer_model"]=="diffusion":
                last_step_images = self.infer_image_by_diffusion(normalized_cond=normalized_cond)
            elif self.policy_config["infer_model"]=="conditional_diffusion":
                last_step_images = self.infer_image_by_conditional_diffusion(normalized_cond=normalized_cond)
            elif self.policy_config["infer_model"]=="diffusion_1D":
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

            ## create raw_cost for logs
            raw_cost = {"cost_b": cost_b_ensembles,
                        "cost_r": cost_r_ensembles,
                        "cost_y": cost_y_ensembles}

            # import ipdb; ipdb.set_trace()

            # 実際にはブルーだけを使っている
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
            self.ensemble_obs_model.cast_2d_image_to_box_color(img=ensemble_image,config={"axis":"z"})
            ## get each axis ensemble images
            ensemble_image_z = self.ensemble_obs_model.get_2d_image(axis="z")
            ensemble_image_x = self.ensemble_obs_model.get_2d_image(axis="x")
            ensemble_image_y = self.ensemble_obs_model.get_2d_image(axis="y")

            ## create edited_cost for logs
            edited_cost = {"cost_b":edited_cost_b,
                           "cost_r":edited_cost_r,
                           "cost_y":edited_cost_y,}

            cost_map_logs ={"raw_cost":raw_cost,
                            "editied_cost":edited_cost}



        elif self.policy_config["ctrl_mode"] == "oracle_obs" or self.policy_config["ctrl_mode"] == "random":

            ensemble_image = self.oracle_image_z
            self.ensemble_obs_model.cast_2d_image_to_box_color(img=ensemble_image,config={"axis":"z"})
            ## get each axis ensemble images
            ensemble_image_z = self.ensemble_obs_model.get_2d_image(axis="z")
            ensemble_image_x = self.ensemble_obs_model.get_2d_image(axis="x")
            ensemble_image_y = self.ensemble_obs_model.get_2d_image(axis="y")
            ensemble_images  = {"image_x":ensemble_image_x,
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

            ############################################################
            # get cost of each segmented images for oracle obs and random
            #############################################################
            # transform mini batch image shape do not need transform axis, then permute if fixed "z"
            cost_x_y = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_yellow["image_x"],permute="z").sum(3).sum(1).sum(1)
            cost_y_y = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_yellow["image_y"],permute="z").sum(3).sum(1).sum(1)
            cost_z_y = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_yellow["image_z"],permute="z").sum(3).sum(1).sum(1)


            # transform mini batch image shape do not need transform axis, then permute if fixed "z"
            cost_x_b = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_blue["image_x"],permute="z").sum(3).sum(1).sum(1)
            cost_y_b = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_blue["image_y"],permute="z").sum(3).sum(1).sum(1)
            cost_z_b = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_blue["image_z"],permute="z").sum(3).sum(1).sum(1)


            # transform mini batch image shape do not need transform axis, then permute if fixed "z"
            cost_x_r = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_red["image_x"],permute="z").sum(3).sum(1).sum(1)
            cost_y_r = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_red["image_y"],permute="z").sum(3).sum(1).sum(1)
            cost_z_r = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_red["image_z"],permute="z").sum(3).sum(1).sum(1)

            ## create raw_cost for logs
            raw_cost = {"cost_b": cost_map_blue,
                        "cost_r": cost_map_red,
                        "cost_y": cost_map_yellow}

            cost_map_logs ={"raw_cost":raw_cost,
                            "editied_cost":raw_cost}


        '''
        #############################################################
        ## get split position for parts splitting !!! under development
        #############################################################
        # split_cost_th = split_cost_th_candidate[min(split_cost_th_candidate, key=split_cost_th_candidate.get)]
        split_cost_th     = 0.0
        # split_candidate_x = self.find_false_true_false_indices(((cost_x_b+cost_x_r).clip(split_cost_th,None)-split_cost_th)==0.0)
        # split_candidate_y = self.find_false_true_false_indices(((cost_y_b+cost_y_r).clip(split_cost_th,None)-split_cost_th)==0.0)
        # split_candidate_z = self.find_false_true_false_indices(((cost_z_b+cost_z_r).clip(split_cost_th,None)-split_cost_th)==0.0)

        split_candidate_x = self.find_false_true_false_indices(((cost_x_b).clip(split_cost_th,None)-split_cost_th)==0.0)
        split_candidate_y = self.find_false_true_false_indices(((cost_y_b).clip(split_cost_th,None)-split_cost_th)==0.0)
        split_candidate_z = self.find_false_true_false_indices(((cost_z_b).clip(split_cost_th,None)-split_cost_th)==0.0)


        split_index_y     = self.get_split_idx(split_candidate_y,axis="y",observation_history=observation_history)
        split_index_x     = self.get_split_idx(split_candidate_x,axis="x",observation_history=observation_history)
        split_index_z     = self.get_split_idx(split_candidate_z,axis="z",observation_history=observation_history)

        split_candidate = [split_index_x,split_index_y,split_index_z]
        split_index     = self.random_non_negative_one_element(split_candidate)

        if split_index != -1 :
            grid_size =  int(self.ensemble_obs_model.voxel_hander.box_array.grid_3dim_size[0])
            split_index_z_max =  int(grid_size-1)
            split_index_x_max =  int(grid_size*2-1)
            split_index_y_max =  int(grid_size*2)

            if split_index <=split_index_z_max:
                split_index_ = split_index
                offset = 0
                axis = "z"
                cost_ = cost_z_b
            elif split_index_z_max<split_index<=split_index_x_max:
                split_index_ = split_index-grid_size
                cost_ = cost_x_b
                offset = grid_size
                axis = "x"
            else:
                split_index_ = split_index-split_index_y_max
                cost_ = cost_y_b
                offset = split_index_y_max
                axis = "y"

            forward_index ,back_ward_index = self.find_nonzero_indices_both_1(lst=cost_,start_index = split_index_)


            if np.abs(split_index-forward_index)<(split_index-back_ward_index):
                split_range = (np.asarray([back_ward_index, split_index_])).tolist()
            elif np.abs(split_index-forward_index)>(split_index-back_ward_index):
                split_range = (np.asarray([split_index,forward_index])).tolist()
            else:
                # import ipdb;ipdb.set_trace()
                forward_index,back_ward_index = self.find_nonzero_indices_both_2(lst=cost_,start_index = split_index_)

                if  forward_index>back_ward_index:
                    split_range = [0,split_index_]
                elif forward_index<back_ward_index:
                    split_range = [split_index_,cost_.shape[0]]
                else:
                    trigger = np.random.randint(2)
                    if trigger == 1:
                        split_range = [0,split_index_]
                    else:
                        split_range = [split_index_,cost_.shape[0]]
                    # split_range = [split_index_,cost]
                    # import ipdb;ipdb.set_trace()

            # if split_range is None:
                # import ipdb;ipdb.set_trace()
                print(f"===================: {split_range}")
                self.split_obs_config[str(split_range)] = { "axis":axis,
                                                            "range":split_range,
                                                            "offset":offset}

                print(f"split_range:{self.split_obs_config}")
        '''

        #####################################################################
        ## get slice range for pats remove
        #####################################################################
        split_index= 2
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

            if min(slice_range)<env2.grid_config['side_length']:
                axis = "z"
                offset= 0
            elif min(slice_range)<env2.grid_config['side_length']:
                axis = "x"
                offset = env2.grid_config['side_length']
            else:
                axis = "y"
                offset = env2.grid_config['side_length']+env2.grid_config['side_length']

            if len(slice_range)!=1:

                import ipdb; ipdb.set_trace()
                if slice_range[0]>slice_range[1]:
                    temp_slice_range = (np.asarray(slice_range[::-1])-offset)[1:]
                else:
                    temp_slice_range = (np.asarray(slice_range)-offset)[:-1]

                split_range = [temp_slice_range[0],temp_slice_range[-1]]
                self.split_obs_config[str(split_range)] = { "axis":axis,
                                                            "range":split_range,
                                                            "offset":offset}
            else:
                a = 0

        # elif split_index!=-1:
        elif split_index ==-1000:
            # import ipdb;ipdb.set_trace()
            # print(f"iter:{iters},split_idx:{split_index}")
            # slice_range = [split_index]
            a = 0
        else:
            slice_range_z =self.get_slice_range(cost=cost_z_b,axis="z",observation_history=observation_history)
            slice_range_x =self.get_slice_range(cost=cost_x_b,axis="x",observation_history=observation_history)
            slice_range_y =self.get_slice_range(cost=cost_y_b,axis="y",observation_history=observation_history)
            slice_range_candidates = [slice_range_z,slice_range_x,slice_range_y]
            # 最も長いリスト
            slice_range = max(slice_range_candidates, key=len)

            if min(slice_range)<env2.grid_config['side_length']:
                axis = "z"
                offset= 0
            # elif min(slice_range)<env2.grid_config['side_length']:
            elif min(slice_range)<env2.grid_config['side_length']+env2.grid_config['side_length']:
                axis = "x"
                offset = env2.grid_config['side_length']
            else:
                axis = "y"
                offset = env2.grid_config['side_length']+env2.grid_config['side_length']

            # import ipdb; ipdb.set_trace()

            if len(slice_range)!=1:

                if slice_range[0]>slice_range[1]:
                    temp_slice_range = (np.asarray(slice_range[::-1])-offset)[1:]
                else:
                    temp_slice_range = (np.asarray(slice_range)-offset)[:-1]

                split_range = [temp_slice_range[0],temp_slice_range[-1]]
                self.split_obs_config[str(split_range)] = { "axis":axis,
                                                            "range":split_range,
                                                            "offset":offset}
            else:
                a = 0

            import ipdb;ipdb.set_trace()
            cost_map_logs["slice_candidate"]={"candidate_x":slice_range_x,
                                              "candidate_y":slice_range_y,
                                              "candidate_z":slice_range_z}
            cost_map_logs["slice_range"]=slice_range
            pickle_utils().save(dataset=cost_map_logs, save_path=save_path+f"/{iters}_cost_map_logs.pickle")


        print("------------------------------------")
        print(f"split_range:{self.split_obs_config}")
        print("------------------------------------")
        cost_x = cost_x_b
        cost_y = cost_y_b
        cost_z = cost_z_b




        #################################################
        ## random policy setting
        ################################################
        if self.policy_config["ctrl_mode"] == "random" :
            action_index_max                = int(self.ensemble_obs_model.voxel_hander.box_array.grid_3dim_size[0]*3-1)
            random_number                   = random.randint(1, 3)
            random_slice_range_candidate    = [i for i in range(0, action_index_max)]
            observation_history_keys        = list(observation_history.keys())
            slice_range_candidates          = [x for x in random_slice_range_candidate if x not in observation_history_keys]
            slice_range                     = random.sample(slice_range_candidates, random_number)
            print(slice_range)


        #######################################################################################################
        #### not necessary part, deprecated
        #######################################################################################################
        cost_data ={"z":cost_z,
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

        infos ={"ensemble_image": { "z":ensemble_image_z,
                                    "x":ensemble_image_x,
                                    "y":ensemble_image_y,}}

        # print(f"split_candidate :{split_candidate}, spit_idx : {split_index}, slice_range :{slice_range}")
        print(f"spit_idx : {split_index}, slice_range :{slice_range}")


        return slice_range, sort_action_candidate, infos



    def set_oracle_obs(self,oracle_obs_image):
        self.oracle_image_z = oracle_obs_image


    def update_split_obs_config(self, slice_range ,grid_config):
        # import ipdb; ipdb.set_trace()
        if min(slice_range)<grid_config['side_length']:
            axis = "z"
            offset= 0
        elif min(slice_range)<grid_config['side_length']+grid_config['side_length']:
            axis = "x"
            offset = grid_config['side_length']
        else:
            axis = "y"
            offset = grid_config['side_length']+grid_config['side_length']


        if len(slice_range)!=1:

            if slice_range[0]>slice_range[1]:
                temp_slice_range = (np.asarray(slice_range[::-1])-offset)[1:]
            else:
                temp_slice_range = (np.asarray(slice_range)-offset)[:-1]

            split_range = [temp_slice_range[0],temp_slice_range[-1]]
            self.split_obs_config[str(split_range)] = { "axis":axis,
                                                        "range":split_range,
                                                        "offset":offset} # action history
