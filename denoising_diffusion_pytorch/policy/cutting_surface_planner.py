


import random
import numpy as np
import torch


from denoising_diffusion_pytorch.utils.normalization import LimitsNormalizer
from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy



class cutting_surface_planner():


    def __init__(self, diffusion, sample_image_num, obs_model ,config):

        # self.ensemble_obs_model     = voxel_cut_handler(grid_config=grid_config, mesh_components=mesh_components,zero_initialize=True)
        self.ensemble_obs_model     = obs_model
        self.diffusion              = diffusion
        self.sample_image_num       = sample_image_num
        self.policy_config          = config




    def color_range_mask(self, image, mask_config):
        """
        2次元画像から指定した色の範囲内であれば1、そうでなければ0を判定する関数
        Args:
            image (numpy.ndarray): 2次元画像（16x16のRGB画像）
            lower_bound (tuple): 色の下限範囲 (R, G, B)
            upper_bound (tuple): 色の上限範囲 (R, G, B)
        Returns:
            numpy.ndarray: 範囲内であれば1、そうでなければ0を持つ2次元マスク
        """
        lower_bound = mask_config["target_mask_lb"]
        upper_bound = mask_config["target_mask_ub"]

        # RGB画像の各チャンネルを個別に比較
        mask_r = (image[:,:,0] >= lower_bound[0]) & (image[:,:,0] <= upper_bound[0])
        mask_g = (image[:,:,1] >= lower_bound[1]) & (image[:,:,1] <= upper_bound[1])
        mask_b = (image[:,:,2] >= lower_bound[2]) & (image[:,:,2] <= upper_bound[2])

        # # 全てのチャンネルで条件を満たしているピクセルを1、それ以外のピクセルを0とする
        # mask = mask_r & mask_g & mask_b
        # return mask.astype(int)


        # # 全てのチャンネルで条件を満たしているピクセルを黒、それ以外のピクセルを白とする
        # mask = np.zeros_like(mask_r, dtype=np.uint8)  # 黒のマスクを初期化
        # mask[mask_r & mask_g & mask_b] = 1.0 # 全てのチャンネルで条件を満たすピクセルに白を設定


        # mask = np.zeros_like(image,dtype=np.uint8)
        # mask[:,:,0]=mask_r*0.0
        # mask[:,:,1]=mask_g*0.0
        # mask[:,:,2]=mask_b*0.0

        # return mask

        # 全てのチャンネルで条件を満たしているピクセルを白、それ以外のピクセルを黒とする
        mask = np.zeros_like(image, dtype=np.uint8)  # 黒のマスクを初期化
        mask[mask_r & mask_g & mask_b] = [1., 1., 1.]  # 全てのチャンネルで条件を満たすピクセルに白を設定
        return mask




    # def get_optimal_act(self,ensemble_image,obs_config):

    def get_optimal_act(self,slice_img,observation_history):


        if self.policy_config["ctrl_mode"] != "oracle_obs" and \
            self.policy_config["ctrl_mode"] != "random" :

            normalizer      = LimitsNormalizer(slice_img)
            normalized_cond = normalizer.normalize(slice_img).transpose(2,0,1)
            normalized_cond = to_torch(normalized_cond)


            if self.policy_config["ctrl_mode"] == "no_cond":
                cond = None

            else:
                cond = {0:{ "idx":torch.where(normalized_cond>-1.0),
                            "val":normalized_cond}}

            sample_image        = self.diffusion.model.sample(batch_size=self.sample_image_num, return_all_timesteps=True, cond = cond ).detach().cpu()
            batch_images        = (torch.permute(sample_image,(0,1,3,4,2))*255.0).numpy().astype(np.uint8)
            last_step_images    = batch_images[:,-1,:,:,:]
            ensemble_image      = last_step_images.mean(0)/255.0

        elif self.policy_config["ctrl_mode"] == "oracle_obs" or self.policy_config["ctrl_mode"] == "random":
            ensemble_image = self.oracle_image_z




        ## diffusion is trained with z axis sliced image
        self.ensemble_obs_model.cast_2d_image_to_box_color(img=ensemble_image,config={"axis":"z"})



        ensemble_image_z = self.ensemble_obs_model.get_2d_image(axis="z")
        ensemble_image_x = self.ensemble_obs_model.get_2d_image(axis="x")
        ensemble_image_y = self.ensemble_obs_model.get_2d_image(axis="y")



        target_mask = np.asarray([0.2,0.8,0.8])
        # image_mask_config = {"target_mask":target_mask,
        #                     "target_mask_lb":target_mask-0.5,
        #                     "target_mask_ub":target_mask+0.5,}
        image_mask_config = {"target_mask":target_mask,
                            "target_mask_lb":target_mask-np.asarray([0.1,0.1,0.1]),
                            "target_mask_ub":target_mask+np.asarray([0.7,0.2,0.2])}


        # pil_image_save_from_numpy(ensemble_image_z,"./ensemble_z.png")
        # pil_image_save_from_numpy(ensemble_image_x,"./ensemble_x.png")
        # pil_image_save_from_numpy(ensemble_image_y,"./ensemble_y.png")


        cost_map_z = self.color_range_mask(ensemble_image_z,image_mask_config)
        # pil_image_save_from_numpy(cost_map_z,"./cost_map_z.png")


        cost_map_x = self.color_range_mask(ensemble_image_x,image_mask_config)
        # pil_image_save_from_numpy(cost_map_x,"./cost_map_x.png")


        cost_map_y = self.color_range_mask(ensemble_image_y,image_mask_config)
        # pil_image_save_from_numpy(cost_map_y,"./cost_map_y.png")






        ## transform mini batch image shape do not need transform axis, then permute if fixed "z"
        ensemble_image_x_mini_batch = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_x,permute="z")
        ensemble_image_y_mini_batch = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_y,permute="z")
        ensemble_image_z_mini_batch = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_z,permute="z")


        cost_x = ensemble_image_x_mini_batch.mean(3).sum(1).sum(1)
        cost_y = ensemble_image_y_mini_batch.mean(3).sum(1).sum(1)
        cost_z = ensemble_image_z_mini_batch.mean(3).sum(1).sum(1)


        # cost_data = {"x":cost_x,
        #              "y":cost_y,
        #              "z":cost_z}


        cost_data = {"z":cost_z,
                     "x":cost_x,
                     "y":cost_y}

        # cost_data = {"z":cost_z,
        #              "y":cost_x,
        #              "x":cost_y}

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


        # ランダムに数を生成
        rand_num = np.random.random()

        infos ={"ensemble_image":{"z":ensemble_image_z,
                                  "x":ensemble_image_x,
                                  "y":ensemble_image_y,}}


        # action_idx = list(sort_action_candidate.keys())[0]
        # return action_idx,sort_action_candidate,infos


        if self.policy_config["ctrl_mode"] == "random":
            #################################################################
            # randomly set next action
            #################################################################
            action_idx =  random.choice(list(sort_action_candidate.keys()))
            return [action_idx] , sort_action_candidate ,infos

        elif    self.policy_config["ctrl_mode"] == "epsilon_greedy_00" or \
                self.policy_config["ctrl_mode"] == "no_cond" or \
                self.policy_config["ctrl_mode"] == "oracle_obs":
            ##################################################################
            ## greedy select next action with minimal cost
            ##################################################################
            action_idx = list(sort_action_candidate.keys())[0]
            return [action_idx],sort_action_candidate,infos

        elif self.policy_config["ctrl_mode"] == "epsilon_greedy_001":
            epsilon_param = 0.01
        elif self.policy_config["ctrl_mode"] == "epsilon_greedy_01":
            epsilon_param = 0.1
        elif self.policy_config["ctrl_mode"] == "epsilon_greedy_05":
            epsilon_param = 0.5
        else:
            NotImplementedError()

        ##################################################################
        ## epsilon-greedy select next action with minimal cost
        ##################################################################
        if rand_num < epsilon_param:
            print("random")
            # ランダムに行動を選択する
            action_idx =  random.choice(list(sort_action_candidate.keys()))
            return [action_idx] ,sort_action_candidate,infos

        else:
            # 最適な行動を選択する
            action_idx = list(sort_action_candidate.keys())[0]

            return [action_idx],sort_action_candidate,infos



    def set_oracle_obs(self,oracle_obs_image):
        self.oracle_image_z = oracle_obs_image