

import ray
from tqdm import tqdm
from copy import copy
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import dismantling_env
from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.os_utils import save_json ,get_folder_name,create_folder,pickle_utils,get_path
from denoising_diffusion_pytorch.utils.voxel_render import pv_voxel_render,pv_voxel_render_parallel
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy ,pil_image_load_to_numpy,numpy_to_pil

import numpy as np

# ray.init(log_to_driver=False,num_cpus=24) # for ros
ray.init(log_to_driver=False) # for ros
# ray.init(log_to_driver=False) # for ro



if __name__ == '__main__':


    #///////////////////////////////////////////////////
    tags = "epsilon_greedy_00"
    # tags = "no_cond"
    # tags = "oracle_obs"
    # tags = "random"


    # root_folder_ = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456789_clip_ucb_raw_0.5_v12_1/"
    # root_folder_ = "/home/tomoya-y/denoising_diffusion_pytorch/logs/Image_diffusion_2D/eval/T8_partial_obs_PT2000_B_T8_partial_obs__a___v12_1_for_paper_render"
    # root_folder_ = "/home/dev/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/eval/T8_partial_obs_PT2000_B_T8_partial_obs__a___v12_1_for_paper_render"
    root_folder_ = "/home/dev/workspace/dataset/nedo_dismantling_log/Image_diffusion_2D/eval/T8_partial_obs_PT100000_B_T8_partial_obs__a___v12_1_for_paper_render"
    #///////////////////////////////////////////////////

    root_folder  = root_folder_+f'/{tags}'

    save_prefix = "no_axis_w_cutting_plane3"

    # dim_2D = 512
    # dim_3D = 64

    dim_2D = 64
    dim_3D = 16
    s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),"side_length":dim_3D}


    # dim_2D = 343
    # dim_3D = 49
    # s_grid_config = {"bounds":(-0.3,0.3,-0.3,0.3,-0.3,0.3), 'side_length':dim_3D}


    save_name = f"dim_{dim_3D}_{save_prefix}"

    # __new__ を使って __init__ を呼ばずにインスタンスを作成
    cutting_env = object.__new__(dismantling_env)
    action_table = cutting_env.get_action_table(s_grid_config)


    model_type_folders = get_folder_name(root_folder)
    for j in range(len(model_type_folders)):

        episodes_folder = root_folder+"/"+model_type_folders[j]
        episodes        = get_folder_name(episodes_folder)

        # for i in tqdm(range(len(episodes))):
        # for i in tqdm(range(3,6)):
        for i in tqdm(range(1)):
        # for i in tqdm(range(3)):

                data_folder     = episodes_folder+"/"+episodes[i]
                print(f"load_data:{data_folder}")
                save_folder_    = data_folder+ "/3d_cutting_process"
                create_folder(save_folder_)

                load_data = pickle_utils().load(load_path=data_folder+f"/rollout_data.pickle")

                oracle_2d_map               = pil_image_load_to_numpy(data_folder+"/oracle_obs_cast_z_axis0.png",resize=(dim_2D,dim_2D))
                cutting_process_2d_map      = load_data["observations"]
                action                      = load_data["actions"]

                ##########################
                ## cutting process setting for paper
                ##########################
                cutting_process_2d_map_ = np.concatenate([oracle_2d_map[None,:,:,:]*0.0, cutting_process_2d_map], axis=0)
                action_ = np.concatenate([np.asarray([0]),action])
                step_num, width, height, channel = cutting_process_2d_map_.shape
                cutting_process_2d_map = np.empty((int(step_num*2), width, width, channel), dtype=cutting_process_2d_map_.dtype)
                cutting_process_2d_map[0::2] = cutting_process_2d_map_        # 偶数番目に元画像
                cutting_process_2d_map[1::2] = cutting_process_2d_map_.copy() # 奇数番目にコピー
                action = np.empty((int(step_num*2)), dtype=action_.dtype)
                action[0::2] = action_        # 偶数番目に元画像
                action[1::2] = action_.copy() # 奇数番目にコピー
                action = np.roll(action, -1) # shift the action array to align with the cutting process



                # 配列に追加（concatenate）
                cutting_process_2d_map = np.concatenate([cutting_process_2d_map, cutting_process_2d_map[-1:]], axis=0)
                action = np.concatenate([action,action[-1:]],axis=0)
                print(f"action_idx:{action}")

                # import ipdb;ipdb.set_trace()


                cutting_process_2d_map_resized = []
                for i in range(cutting_process_2d_map.shape[0]):
                    cutting_process_2d_map_tmp = numpy_to_pil(cutting_process_2d_map[i])
                    bb =cutting_process_2d_map_tmp.resize((dim_2D,dim_2D))
                    cutting_process_2d_map_resized.append(np.asarray(bb)/255.0)

                cutting_process_2d_map = np.asarray(cutting_process_2d_map_resized)
                # cutting_process_2d_map_flip = np.where(cutting_process_2d_map==oracle_2d_map,np.asarray([0.,0.,0.]),oracle_2d_map)
                cutting_process_2d_map_flip = np.where((cutting_process_2d_map>=oracle_2d_map-0.05) & (cutting_process_2d_map<=oracle_2d_map+0.05),np.asarray([0.,0.,0.]),oracle_2d_map)
                cutting_process_2d_map_flip = cutting_process_2d_map_flip*255.0
                pil_image_save_from_numpy(cutting_process_2d_map_flip[-1]/255.0,data_folder+f"/last_remain_voxels.png")

                # import ipdb;ipdb.set_trace()


                for i in range(cutting_process_2d_map.shape[0]):
                    ## mask the over cutting voxels
                    over_cutting_voxels = np.all(oracle_2d_map == np.asarray([0.2,0.8,0.8]), axis=-1) & np.all(cutting_process_2d_map_flip[i]/255.0 == [0, 0, 0], axis=-1)
                    ocv_result = (cutting_process_2d_map_flip[i]/255.0).copy()
                    # ocv_result[over_cutting_voxels]=np.asarray([255/255,20/255,147/255])
                    ocv_result[over_cutting_voxels]=np.asarray([148/255,0.0,211/255]) #paper
                    # ocv_result[over_cutting_voxels]=np.asarray([148/255,0.0,211/255])
                    cutting_process_2d_map_flip[i]=ocv_result*255.0
                pil_image_save_from_numpy(ocv_result,data_folder+f"/last_remain_voxels_w_ocv_masked{i}.png")


                # previous ver
                pv_voxel_render_parallel().render_cutting_process_v3(save_path=save_folder_,
                                                                        s_grind_config=s_grid_config,
                                                                        action = action,
                                                                        action_table = action_table,
                                                                        sample_images=cutting_process_2d_map_flip,
                                                                        save_tag=save_name)

                # pv_voxel_render_parallel().render_cutting_process_as_pcl_v1(save_path=save_folder_,
                #                                                         s_grind_config=s_grid_config,
                #                                                         action = action,
                #                                                         action_table = action_table,
                #                                                         sample_images=cutting_process_2d_map_flip,
                #                                                         save_tag=save_name)

                del data_folder

    # import ipdb;ipdb.set_trace()
