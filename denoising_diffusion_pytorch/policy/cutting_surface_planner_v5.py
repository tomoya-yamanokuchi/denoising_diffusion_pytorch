


import random
import numpy as np
import torch
from PIL import Image

from denoising_diffusion_pytorch.utils.normalization import LimitsNormalizer
from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np
from denoising_diffusion_pytorch.utils.pil_utils import numpy_to_pil ,cv2_hsv_mask ,pil_to_cv2
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils,load_yaml


class cutting_surface_planner():


    def __init__(self, diffusion, sample_image_num, obs_model ,config):

        # self.ensemble_obs_model     = voxel_cut_handler(grid_config=grid_config, mesh_components=mesh_components,zero_initialize=True)
        self.ensemble_obs_model     = obs_model
        self.diffusion              = diffusion
        self.sample_image_num       = sample_image_num
        self.policy_config          = config
        self.split_obs_config       ={}



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


    def get_split_idx(self,cost,axis,observation_history):

            if axis == "z":
                offset = 0
            elif axis=="x":
                offset = self.ensemble_obs_model.voxel_hander.box_array.grid_3dim_size[0]
            elif axis =="y":
                offset = self.ensemble_obs_model.voxel_hander.box_array.grid_3dim_size[0]+self.ensemble_obs_model.voxel_hander.box_array.grid_3dim_size[0]


            # top, btm = self.find_nonzero_indices(cost)
            # top_split_pos = [offset+top]
            # btm_split_pos = [offset+btm]

            # observation_history_keys = list(observation_history.keys())
            # sprit_range_top = [x for x in top_split_pos if x not in observation_history_keys]
            # sprit_range_btm = [x for x in btm_split_pos if x not in observation_history_keys]

            # if len(sprit_range_top)==0 and len(sprit_range_btm)==0:
            #     slice_range =[-1]
            # elif sprit_range_top == sprit_range_btm:
            #     slice_range     =  sprit_range_top
            # else:
            #     slice_range =[-1]

            # import ipdb;ipdb.set_trace()

            # indices_of_ones = [i for i, val in enumerate(cost) if val == 1]

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

            # if type(slice_range) != list :
            #     return slice_range.tolist()
            # else:
            return slice_range


    def find_nonzero_indices(self, arr):

        ## find indices array > cost threshold
        cost_threshold = 0
        start_index = np.argmax((arr>cost_threshold)!=0)
        end_index = len(arr) - np.argmax((arr[::-1]>cost_threshold) != 0) - 1

        return start_index, end_index


    # def find_false_true_false_indices(self,bool_list):
    #     # indices = []
    #     # for i in range(len(bool_list) - 2):
    #     #     if bool_list[i] == False and bool_list[i + 1] == True and bool_list[i + 2] == False:
    #     #         indices.append(i+1)
    #     # return indices

    #     indices = np.zeros(len(bool_list))
    #     for i in range(len(bool_list) - 2):
    #         if bool_list[i] == False and bool_list[i + 1] == True and bool_list[i + 2] == False:
    #             # indices.append(i+1)
    #             indices [i+1] = 1.0
    #     return indices


    # def find_false_true_false_indices(self,lst):
    #     prev_false_index = None

    #     for i, item in enumerate(lst):
    #         if not item:  # False の場合
    #             if prev_false_index is not None:
    #                 lst[prev_false_index] = 1
    #                 prev_false_index = None
    #         else:  # True の場合
    #             if prev_false_index is not None:
    #                 lst[prev_false_index] = 1
    #             prev_false_index = i

    #     # 最後のFalseの間にTrueがある場合に対応するためにチェック
    #     if prev_false_index is not None:
    #         lst[prev_false_index] = 0

    #     # 1以外の要素を0に変更
    #     lst = [0 if item != 1 else item for item in lst]

    #     return lst

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




    # def get_optimal_act(self,ensemble_image,obs_config):
    def get_optimal_act(self,slice_img_,observation_history, env2,tmp_action,iters,save_path):

        # obs,reward,done,info = env2.step(action_idx=tmp_action)

        obs,reward,done,info = env2.step(action_idx=tmp_action, partial_obs = self.split_obs_config)

        slice_img = obs["sequential_obs"]["z"]
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

            if self.policy_config["ctrl_mode"] == "no_cond":
                cond = None
            else:
                cond = {0:{ "idx":torch.where(normalized_cond>-1.0),
                            "val":normalized_cond}}


            ## infer image and calculate ensemble image
            sample_image        = self.diffusion.model.sample(batch_size=self.sample_image_num, return_all_timesteps=True, cond = cond ).detach().cpu()
            batch_images        = (torch.permute(sample_image,(0,1,3,4,2))*255.0).numpy().astype(np.uint8)
            last_step_images    = batch_images[:,-1,:,:,:]
            # import ipdb;ipdb.set_trace()

            ensemble_image      = last_step_images.mean(0)/255.0

            for k in range(last_step_images.shape[0]):
                raw_pred_image_save_path = save_path+f"/raw_pred_image/step_{iters}"
                create_folder(raw_pred_image_save_path)
                pil_image_save_from_numpy(last_step_images[k]/255.0,raw_pred_image_save_path+f"/ensemble_z_{k}.png")



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




        pil_image_save_from_numpy(ensemble_image_z,"./ensemble_z.png")
        pil_image_save_from_numpy(ensemble_image_x,"./ensemble_x.png")
        pil_image_save_from_numpy(ensemble_image_y,"./ensemble_y.png")


        # import ipdb;ipdb.set_trace()

        #############################################################
        ## get cost of each segmented images
        ##############################################################

        ## transform mini batch image shape do not need transform axis, then permute if fixed "z"
        cost_x_y = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_yellow["image_x"],permute="z").sum(3).sum(1).sum(1)
        cost_y_y = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_yellow["image_y"],permute="z").sum(3).sum(1).sum(1)
        cost_z_y = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_yellow["image_z"],permute="z").sum(3).sum(1).sum(1)


        ## transform mini batch image shape do not need transform axis, then permute if fixed "z"
        cost_x_b = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_blue["image_x"],permute="z").sum(3).sum(1).sum(1)
        cost_y_b = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_blue["image_y"],permute="z").sum(3).sum(1).sum(1)
        cost_z_b = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_blue["image_z"],permute="z").sum(3).sum(1).sum(1)


        ## transform mini batch image shape do not need transform axis, then permute if fixed "z"
        cost_x_r = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_red["image_x"],permute="z").sum(3).sum(1).sum(1)
        cost_y_r = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_red["image_y"],permute="z").sum(3).sum(1).sum(1)
        cost_z_r = self.ensemble_obs_model.voxel_hander.get_2d_image_to_mini_batch_image(cost_map_red["image_z"],permute="z").sum(3).sum(1).sum(1)




        ###############################################
        ## get split position for parts splitting
        ##################################################
        # split_candidate_x = self.find_false_true_false_indices(self.compare_lists_for_zero(cost_x_b,cost_x_r))
        # split_candidate_y = self.find_false_true_false_indices(self.compare_lists_for_zero(cost_y_b,cost_y_r))
        # split_candidate_z = self.find_false_true_false_indices(self.compare_lists_for_zero(cost_z_b,cost_z_r))



        split_cost_th_candidate  ={ "x":np.unique(cost_x_b+cost_x_r)[1],
                                    "y":np.unique(cost_y_b+cost_y_r)[1],
                                    "z":np.unique(cost_z_b+cost_z_r)[1],}

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
            slice_range_z =self.get_slice_range(cost=cost_z_b+cost_z_r,axis="z",observation_history=observation_history)
            slice_range_x =self.get_slice_range(cost=cost_x_b+cost_x_r,axis="x",observation_history=observation_history)
            slice_range_y =self.get_slice_range(cost=cost_y_b+cost_y_r,axis="y",observation_history=observation_history)
            slice_range_candidates = [slice_range_z,slice_range_x,slice_range_y]
            # 最も長いリスト
            slice_range = max(slice_range_candidates, key=len)

            # import ipdb;ipdb.set_trace()

        elif split_index!=-1:
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