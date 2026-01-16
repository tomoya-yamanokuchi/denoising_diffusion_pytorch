


import random
import numpy as np
import torch
from PIL import Image

from denoising_diffusion_pytorch.utils.normalization import LimitsNormalizer
from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np
from denoising_diffusion_pytorch.utils.pil_utils import numpy_to_pil ,cv2_hsv_mask ,pil_to_cv2
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

    def get_color_mask_image(self,images,mask_config):

        cost_map = {}
        for idx, val in enumerate(images):
            cost_map[val] = self.color_range_mask(image=images[val],mask_config=mask_config)

        return cost_map


    def get_slice_range(self,cost,axis,observation_history):

        if axis == "z":
            offset = 0
        elif axis=="x":
            offset = cost.shape[0]
        elif axis =="y":
            offset =cost.shape[0]+cost.shape[0]


        top, btm = self.find_nonzero_indices(cost)

        # top_slice_range = np.arange(0,top-1).tolist()
        # btm_slice_range = np.arange(btm,cost.shape[0]-1).tolist()

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


    def find_nonzero_indices(self, arr):

        # start_index = np.argmax(arr != 0)
        # end_index = len(arr) - np.argmax(arr[::-1] != 0) - 1

        ## find indices array > cost threshold
        cost_threshold = 0
        start_index = np.argmax((arr>cost_threshold)!=0)
        end_index = len(arr) - np.argmax((arr[::-1]>cost_threshold) != 0) - 1

        return start_index, end_index


    # def get_optimal_act(self,ensemble_image,obs_config):

    def get_optimal_act(self,slice_img_,observation_history, env2,tmp_action,iters):

        obs,reward,done,info = env2.step(action_idx=tmp_action)
        slice_img = obs["sequential_obs"]["z"]
        print(f"inner action:{tmp_action}")
        # pil_image_save_from_numpy(obs["sequential_obs"]["z"],f"./seq_obs_cast_z_axis{iters}_{tmp_action}.png")

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

        # import ipdb;ipdb.set_trace()



        ## diffusion is trained with z axis sliced image
        self.ensemble_obs_model.cast_2d_image_to_box_color(img=ensemble_image,config={"axis":"z"})



        ensemble_image_z = self.ensemble_obs_model.get_2d_image(axis="z")
        ensemble_image_x = self.ensemble_obs_model.get_2d_image(axis="x")
        ensemble_image_y = self.ensemble_obs_model.get_2d_image(axis="y")



        # # ensemble_image_z_cv2 = pil_to_cv2(numpy_to_pil(ensemble_image_z))
        # # ensemble_image_x_cv2 = pil_to_cv2(numpy_to_pil(ensemble_image_x))
        # # ensemble_image_y_cv2 = pil_to_cv2(numpy_to_pil(ensemble_image_y))

        # # # hoge = Image.fromarray(cv2_hsv_mask(ensemble_image_x_cv2)).convert("RGB")
        # # cost_map_z = np.asarray(Image.fromarray(cv2_hsv_mask(ensemble_image_z_cv2)).convert("RGB"))/255.0
        # # cost_map_x = np.asarray(Image.fromarray(cv2_hsv_mask(ensemble_image_x_cv2)).convert("RGB"))/255.0
        # # cost_map_y = np.asarray(Image.fromarray(cv2_hsv_mask(ensemble_image_y_cv2)).convert("RGB"))/255.0




        # ### default
        # # target_mask = np.asarray([0.2,0.8,0.8])
        # # image_mask_config = {"target_mask":target_mask,
        # #                     "target_mask_lb":target_mask-np.asarray([0.1,0.1,0.1]),
        # #                     "target_mask_ub":target_mask+np.asarray([0.7,0.2,0.2])}


        # target_mask = np.asarray([0.8,0.8,0.2])
        # image_mask_config = {"target_mask":target_mask,
        #                     "target_mask_lb":target_mask-np.asarray([0.1,0.1,0.1]),
        #                     "target_mask_ub":target_mask+np.asarray([0.2,0.2,0.7])}


        # # pil_image_save_from_numpy(ensemble_image_z,"./ensemble_z.png")
        # # pil_image_save_from_numpy(ensemble_image_x,"./ensemble_x.png")
        # # pil_image_save_from_numpy(ensemble_image_y,"./ensemble_y.png")

        # cost_map_z = self.color_range_mask(ensemble_image_z,image_mask_config)
        # # pil_image_save_from_numpy(cost_map_z,"./cost_map_z.png")


        # cost_map_x = self.color_range_mask(ensemble_image_x,image_mask_config)
        # # pil_image_save_from_numpy(cost_map_x,"./cost_map_x.png")


        # cost_map_y = self.color_range_mask(ensemble_image_y,image_mask_config)
        # # pil_image_save_from_numpy(cost_map_y,"./cost_map_y.png")




        ensemble_images = {"image_x":ensemble_image_x,
                            "image_y":ensemble_image_y,
                            "image_z":ensemble_image_z,}


        target_mask_y = np.asarray([0.8,0.8,0.2])
        image_mask_config_y = {"target_mask":target_mask_y,
                            "target_mask_lb":target_mask_y-np.asarray([0.1,0.1,0.1]),
                            "target_mask_ub":target_mask_y+np.asarray([0.2,0.2,0.6])}

        target_mask_b = np.asarray([0.2,0.8,0.8])
        image_mask_config_b = {"target_mask":target_mask_b,
                            "target_mask_lb":target_mask_b-np.asarray([0.1,0.1,0.1]),
                            "target_mask_ub":target_mask_b+np.asarray([0.7,0.2,0.2])}


        target_mask_r = np.asarray([0.8,0.2,0.2])
        image_mask_config_r = {"target_mask":target_mask_r,
                            "target_mask_lb":target_mask_r-np.asarray([0.1,0.1,0.1]),
                            "target_mask_ub":target_mask_r+np.asarray([0.2,0.6,0.6])}



        cost_map_yellow = self.get_color_mask_image(images=ensemble_images,mask_config=image_mask_config_y)
        cost_map_blue   = self.get_color_mask_image(images=ensemble_images,mask_config=image_mask_config_b)
        cost_map_red    = self.get_color_mask_image(images=ensemble_images,mask_config=image_mask_config_r)

        # cost_map_x = cost_map_red["image_x"]+cost_map_blue["image_x"]
        # cost_map_y = cost_map_red["image_y"]+cost_map_blue["image_y"]
        # cost_map_z = cost_map_red["image_z"]+cost_map_blue["image_z"]


        cost_map_x = cost_map_blue["image_x"]
        cost_map_y = cost_map_blue["image_y"]
        cost_map_z = cost_map_blue["image_z"]



        # import ipdb;ipdb.set_trace()




        ## transform mini batch image shape do not need transform axis, then permute if fixed "z"
        ensemble_image_x_mini_batch = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_x,permute="z")
        ensemble_image_y_mini_batch = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_y,permute="z")
        ensemble_image_z_mini_batch = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_z,permute="z")


        # cost_x = ensemble_image_x_mini_batch.mean(3).sum(1).sum(1)
        # cost_y = ensemble_image_y_mini_batch.mean(3).sum(1).sum(1)
        # cost_z = ensemble_image_z_mini_batch.mean(3).sum(1).sum(1)

        cost_x = ensemble_image_x_mini_batch.sum(3).sum(1).sum(1)
        cost_y = ensemble_image_y_mini_batch.sum(3).sum(1).sum(1)
        cost_z = ensemble_image_z_mini_batch.sum(3).sum(1).sum(1)



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

        infos ={"ensemble_image":{"z":ensemble_image_z,
                                  "x":ensemble_image_x,
                                  "y":ensemble_image_y,}}



        slice_range_z =self.get_slice_range(cost=cost_z,axis="z",observation_history=observation_history)
        slice_range_x =self.get_slice_range(cost=cost_x,axis="x",observation_history=observation_history)
        slice_range_y =self.get_slice_range(cost=cost_y,axis="y",observation_history=observation_history)

        slice_range_candidates = [slice_range_z,slice_range_x,slice_range_y]


        # # 最も長いリスト
        slice_range = max(slice_range_candidates, key=len)

        # slice_range_ = random.choice(slice_range_candidates)
        # if isinstance(slice_range_, list):
        #     slice_range  = [slice_range_[0]]
        # else:
        #     slice_range  = [slice_range_]





        if self.policy_config["ctrl_mode"] == "random" :
            ### random action
            # random_number = random.randint(1, 5)
            # random_slice_range_candidate = random.sample(range(0, 47 + 1), random_number)
            # observation_history_keys = list(observation_history.keys())
            # slice_range = [x for x in random_slice_range_candidate if x not in observation_history_keys]

            random_number = random.randint(1, 5)
            random_slice_range_candidate = [i for i in range(0, 48)]
            observation_history_keys = list(observation_history.keys())
            slice_range_candidates = [x for x in random_slice_range_candidate if x not in observation_history_keys]
            slice_range = random.sample(slice_range_candidates, random_number)

            # print(random_number)
            # print(random_slice_range_candidate)
            print(slice_range)

            # import ipdb;ipdb.set_trace()
            # action_idx  =  random.choice(list(sort_action_candidate.keys()))
            # slice_range = [action_idx]


        return slice_range, sort_action_candidate, infos



    def set_oracle_obs(self,oracle_obs_image):
        self.oracle_image_z = oracle_obs_image