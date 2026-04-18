
import os
import ray
from tqdm import tqdm
from copy import copy
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import dismantling_env
from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.os_utils import save_json ,get_folder_name,create_folder,pickle_utils,get_path
from denoising_diffusion_pytorch.utils.voxel_render import pv_voxel_render,pv_voxel_render_parallel
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy ,pil_image_load_to_numpy,numpy_to_pil
from scripts.plotter.plot_internal_structure_heatmap import get_each_predicted_image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# ray.init(log_to_driver=False,num_cpus=24) # for ros
ray.init(log_to_driver=False) # for ros
# ray.init(log_to_driver=False) # for ro


def get_2d_image_to_mini_batch_image_for_render(dim_2d, dim_3d, image, permute):
    """Converts a 2D image into a mini-batch image representation.

    Args:
        image (np.ndarray): The input 2D image. (v*np.sqrt(v), v*np.sqrt(v), 3). v is grind dim.
        permute (str): The permutation type for the image ('x', 'y', or 'z').

    Returns:
        np.ndarray: The mini-batch image. (v, v, v, 3). v is grind dim.

    """


    grid_2dim    = dim_2d
    grid_3dim    = dim_3d
    batch_img_len = int(grid_2dim/grid_3dim)

    batch_2d_image_ = np.zeros((grid_3dim, grid_3dim, grid_3dim, 3))
    k = 0
    for j in range(batch_img_len):
        for i in range(batch_img_len):
            batch_2d_image_[k] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
            k = k+1


    if permute == "z":
        batch_2d_image  = batch_2d_image_
    elif permute == "y":
        batch_2d_image  = batch_2d_image_.transpose(1,0,2,3)
    elif permute == "x":
        batch_2d_image  = batch_2d_image_.transpose(2,1,0,3)

    return batch_2d_image





def get_mini_batch_image_to_2d_image_for_render(dim_2d, dim_3d, batch_3d_image, permute):
    """
    Converts a mini-batch 3D image representation into a 2D image (flattened along z-axis order).

    Args:
        batch_3d_image (np.ndarray): The input 3D mini-batch image. Shape (v, v, v, 3)
        permute (str): The permutation type applied to the original image ('x', 'y', or 'z')

    Returns:
        np.ndarray: The reconstructed 2D image in z-axis slice order. Shape (v*sqrt(v), v*sqrt(v), 3)
    """
    grid_2dim = dim_2d
    grid_3dim = dim_3d
    batch_img_len = int(grid_2dim / grid_3dim)

    # 逆permute: もとのz基準に戻す
    if permute == "z":
        batch_z_order = batch_3d_image
    elif permute == "y":
        batch_z_order = batch_3d_image.transpose(1, 0, 2, 3)
    elif permute == "x":
        batch_z_order = batch_3d_image.transpose(2, 1, 0, 3)
    else:
        raise ValueError(f"Invalid permute argument: {permute}")

    # z軸順に並んだスライスを2Dに展開
    image = np.zeros((grid_2dim, grid_2dim, 3))
    k = 0
    for j in range(batch_img_len):
        for i in range(batch_img_len):
            image[j*grid_3dim:(j+1)*grid_3dim, i*grid_3dim:(i+1)*grid_3dim] = batch_z_order[k]
            k += 1

    return image




# def create_white_to_black_cmap(center_index, total=16):
#     """
#     Create a white-to-black colormap centered at a specific index.
    
#     Args:
#         center_index (int): The index where the color is pure white.
#         total (int): Total number of indices (default 17, i.e., 0 to 16).
    
#     Returns:
#         ListedColormap: A custom colormap.
#     """
#     assert 0 <= center_index < total, "center_index must be in range [0, total-1]"

#     colors = []

#     for i in range(total):
#         dist = abs(i - center_index)
#         max_dist = max(center_index, total - 1 - center_index)
#         intensity = 1.0 - dist / max_dist  # 1.0（白）→ 0.0（黒）へ
#         colors.append((intensity, intensity, intensity))  # グレースケールRGB

#     return ListedColormap(colors)




# def create_centered_cmap(center_index=5,length=16):
#     cmap = np.zeros((length, 3))  # RGB

#     for i in range(length):
#         dist = abs(i - center_index) / max(center_index, length - 1 - center_index)
#         value = 1.0 - dist  # 白が中心、遠いほど黒へ
#         cmap[i] = [value, value, value]  # グレースケール

#     return cmap




def create_centered_cmap(length=16, center_index=8):
    cmap = np.zeros((length, 3))

    # 左側: 白 -> 青
    left_len = center_index
    for i in range(left_len + 1):
        t = i / left_len if left_len > 0 else 0
        color = (1 - t) * np.array([1, 1, 1]) + t * np.array([0.2, 0.2, 0.2])
        # color = (1 - t) * np.array([1, 1, 1]) + t * np.array([138/255, 43/255, 226/255])  # 青色
        cmap[center_index - i] = color

    # 右側: 白 -> 赤
    right_len = length - center_index - 1
    for i in range(1, right_len + 1):
        t = i / right_len if right_len > 0 else 0
        color = (1 - t) * np.array([1, 1, 1]) + t * np.array([0.2, 0.2, 0.2])
        # color = (1 - t) * np.array([1, 1, 1]) + t * np.array([138/255, 43/255, 226/255])
        cmap[center_index + i] = color

    return cmap



if __name__ == '__main__':


    #///////////////////////////////////////////////////
    tags = "epsilon_greedy_00"
    # tags = "no_cond"
    # tags = "oracle_obs"
    # tags = "random"

    policy_name = "1D_diffusion" #vaeac, "" , 1D_diffusion

    # root_folder = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac_plans/dataset_4142435161_13901k_v1/dataset_4_3_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a0_long_vaeac_cal_cost_mean_ucb_9_t20_8_3_tmp/{tags}"
    # root_folder = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac_plans/dataset_4142435161_13901k_v1/dataset_4_3_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a0_long_vaeac_cal_cost_mean_ucb_9_t20_8_3_tmp1/{tags}"
    # root_folder = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac_plans/dataset_4142435161_13901k_v1/dataset_4_3_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a0_long_vaeac_cal_cost_mean_ucb_9_t20_8_3_tmp2/{tags}"

    # root_folder = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac_plans/dataset_4142435161_13901k_v1/dataset_4_3_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a0_long_vaeac_cal_cost_mean_ucb_9_t20_8_3_tmp2/{tags}"

    # root_folder_ = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456_clip_ucb_raw_0.6_v11_11"
    # root_folder_ = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT200000_B32_T8_partial_obs_vaeac_a123456_clip_ucb_raw_0.5_v11_11"
    # root_folder_ = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456_clip_ucb_raw_0.5_v13_1"

    # root_folder_ = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456789_clip_ucb_raw_0.5_v12_1/"
    # root_folder_ = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT200000_B32_T8_partial_obs_vaeac_a123456789_clip_ucb_raw_0.5_v12_1/"
    # root_folder_ = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT200000_B32_T8_partial_obs_diffusion_1D_a123456789_clip_ucb_raw_0.5_v12_1/"



    # root_folder_ = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456789_clip_ucb_raw_0.5_v12_1/"
    # root_folder_ = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456789_clip_ucb_raw_0.5_v12_1_for_paper_render/"

    # root_folder_ = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT100000_B32_T8_partial_obs_vaeac_a123456789_clip_ucb_raw_0.5_v13_2_tmp/"
    root_folder_ = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT100000_B32_T8_partial_obs_diffusion_1D_a123456789_clip_ucb_raw_0.5_v13_2/"

    # root_folder_ = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_vaeac_a123456_clip_ucb_raw_0.5_v13_2/"
    # root_folder_  = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456_clip_ucb_raw_0.5_v13_1/"
    # root_folder_  = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456_clip_ucb_raw_0.5_v14_1/"

    root_folder  = root_folder_+f'/{tags}'



    dim_2D = 64
    dim_3D = 16
    s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),"side_length":dim_3D}



    # __new__ を使って __init__ を呼ばずにインスタンスを作成
    cutting_env = object.__new__(dismantling_env)
    action_table = cutting_env.get_action_table(s_grid_config)


    model_type_folders = get_folder_name(root_folder)
    # for j in range(len(model_type_folders)):
    # for j in range(len(model_type_folders)):
    # for j in range(0,1):
    # for j in range(5,6):
    for j in range(7,8):

        episodes_folder = root_folder+"/"+model_type_folders[j]
        episodes        = get_folder_name(episodes_folder)

        # for i in tqdm(range(len(episodes))):
        for i in tqdm(range(3,6)):
        # for i in tqdm(range(1)):
        # for i in tqdm(range(3)):
        
        
                save_prefix = f"00_{policy_name}_{tags}_obj{j}_ep{i}_no_axis_w_cutting_plane7_3"
                save_name = f"dim_{dim_3D}_{save_prefix}"

                data_folder     = episodes_folder+"/"+episodes[i]
                print(f"load_data:{data_folder}")
                save_folder_    = data_folder+ "/3d_cutting_process"
                create_folder(save_folder_)

                load_data = pickle_utils().load(load_path=data_folder+f"/rollout_data.pickle")

                oracle_2d_map               = pil_image_load_to_numpy(data_folder+"/oracle_obs_cast_z_axis0.png",resize=(dim_2D,dim_2D))
                cutting_process_2d_map      = load_data["observations"]
                action                      = load_data["actions"]



                ######################################
                ## load cutting cost map if file exists
                ######################################
                cost_map_logs = {}
                for m in range(action.shape[0]):
                    file_path = data_folder + f"/{m}_cost_map_logs.pickle"
                    if os.path.exists(file_path):
                        cost_map_logs[m] = pickle_utils().load(file_path)
                    else:
                        print(f"Not find file: {file_path}")


                #####################################################################################################
                # cutting process setting for paper
                # Duplicate each step in the cutting process (original + copy).
                # Shift the action array left by one to align with the cutting process timeline.
                # Repeat the last image and action once more to match the total number of steps for visualization.
                #####################################################################################################
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



                cutting_process_2d_map_resized = []
                for p in range(cutting_process_2d_map.shape[0]):
                    cutting_process_2d_map_tmp = numpy_to_pil(cutting_process_2d_map[p])
                    bb =cutting_process_2d_map_tmp.resize((dim_2D,dim_2D))
                    cutting_process_2d_map_resized.append(np.asarray(bb)/255.0)

                cutting_process_2d_map = np.asarray(cutting_process_2d_map_resized)
                # cutting_process_2d_map_flip = np.where(cutting_process_2d_map==oracle_2d_map,np.asarray([0.,0.,0.]),oracle_2d_map)
                cutting_process_2d_map_flip = np.where((cutting_process_2d_map>=oracle_2d_map-0.05) & (cutting_process_2d_map<=oracle_2d_map+0.05),np.asarray([0.,0.,0.]),oracle_2d_map)
                cutting_process_2d_map_flip = cutting_process_2d_map_flip*255.0
                pil_image_save_from_numpy(cutting_process_2d_map_flip[-1]/255.0,data_folder+f"/last_remain_voxels.png")



                # ########################################################
                # ### overwrite the ensemble predicted image ver
                # ########################################################
                # data_folder         = episodes_folder+"/"+episodes[i]+"/raw_pred_image"
                # raw_pred_image_dirs = get_folder_name(data_folder)
                # all_predicted_images_in_episode = []
                # for k in range(len(raw_pred_image_dirs)):
                #     raw_pred_image_dir = data_folder+"/"+raw_pred_image_dirs[k]
                #     # Get each predicted image
                #     all_predicted_images = get_each_predicted_image(raw_pred_image_dir, target_ext=".png")
                #     #convert to numpy array
                #     all_predicted_images = np.array(all_predicted_images)
                #     all_predicted_images_in_episode.append(all_predicted_images.mean(0))
                # all_predicted_images_in_episode.append(all_predicted_images.mean(0))
                # all_predicted_images_in_episode = np.array(all_predicted_images_in_episode)
                # all_predicted_images_in_episode = np.repeat(all_predicted_images_in_episode, repeats=2, axis=0)
                # last = all_predicted_images_in_episode[-1:]  
                # cutting_process_2d_map_flip = np.concatenate([all_predicted_images_in_episode, last], axis=0)
                # # import ipdb;ipdb.set_trace()


                ########################################################
                ### overwrite the action cost map ver
                ########################################################
                # for k in range(cutting_process_2d_map_flip.shape[0]):
                #     action_pos = action_table[action[k]]
                #     mini_batch_image = get_2d_image_to_mini_batch_image_for_render(dim_2D, dim_3D, cutting_process_2d_map_flip[k], permute=action_pos["axis"])
                #     # cmaps = create_white_to_black_cmap(center_index=action_pos["loc"], total=dim_3D)
                #     cmaps = create_centered_cmap(center_index=action_pos["loc"],length=dim_3D)
                #     for slices in range(mini_batch_image.shape[0]):
                #         if action_pos["loc"] !=0:
                #             # color = np.asarray(cmaps(slices / (dim_3D - 1))[:3])
                #             mini_batch_image[slices] = cmaps[slices]*255.0
                #     cutting_process_2d_map_flip[k] =get_mini_batch_image_to_2d_image_for_render(dim_2D, dim_3D, mini_batch_image, permute=action_pos["axis"])


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

                # cutting_plane_with_cost_map={}
                # for d in range(action.shape[0]):
                #     if d!=0 and d%2==0 and d<=action.shape[0]-2:
                #         aa = int(d/2-1)
                #         print(d)
                #         print(aa)
                #         cutting_plane_with_cost_map_one_step={}
                #         for i in range(len(action_table)):
                #             cutting_plane_with_cost_map_one_step[i] = {}
                            
                #             ###########################################
                #             ## get cutting cost from cost map and bias
                #             ###########################################
                #             axis_name   = action_table[i]["axis"]
                #             loc_idx     = action_table[i]["loc"]
                #             cutting_plane_with_cost_map_one_step[i]["cost"] = cost_map_logs[aa]["raw_cost"]['cost_b'][f"{axis_name}_axis"].mean(0)[loc_idx]+10
                #         # import ipdb;ipdb.set_trace()

                #         ##################################################
                #         ### bias the cutting cost around the cutting volume
                #         ##################################################
                #         slice_candidate = cost_map_logs[aa]["slice_candidate"]
                #         for axis_candidates in slice_candidate.values():
                #             for value in axis_candidates:
                #                 # print(value)
                #                 cutting_plane_with_cost_map_one_step[value]["cost"]=cutting_plane_with_cost_map_one_step[value]["cost"]-3

                #         ###############################################
                #         ### set the cutting cost of the executed action to 0
                #         ##########################
                #         cutting_pos  = cost_map_logs[aa]['slice_range'][-1]
                #         cutting_plane_with_cost_map_one_step[cutting_pos]["cost"]=0
                #         # import ipdb;ipdb.set_trace()

                #         all_candidates = []

                #         for axis_candidates in cost_map_logs[aa]["slice_candidate"].values():
                #             all_candidates.extend(axis_candidates)
                #         import ipdb;ipdb.set_trace()
                            


                # data = cutting_plane_with_cost_map[16]
                # cmap = plt.cm.viridis
                # updated_data = normalize_cost_and_assign_color(data, cmap)
                # cost_z_bool = np.where(data["z_axis"]>0,1,0)
                # import ipdb;ipdb.set_trace()


                # # cutting costmap_ver
                # pv_voxel_render_parallel().render_cutting_process_v4(save_path=save_folder_,
                #                                                         s_grind_config=s_grid_config,
                #                                                         action = action,
                #                                                         action_table = action_table,
                #                                                         cost_map_logs=cost_map_logs,
                #                                                         sample_images=cutting_process_2d_map_flip,
                #                                                         save_tag=save_name)

                # pv_voxel_render_parallel().render_cutting_process_as_pcl_v1(save_path=save_folder_,
                #                                                         s_grind_config=s_grid_config,
                #                                                         action = action,
                #                                                         action_table = action_table,
                #                                                         sample_images=cutting_process_2d_map_flip,
                #                                                         save_tag=save_name)

                del data_folder#

    # import ipdb;ipdb.set_trace()
