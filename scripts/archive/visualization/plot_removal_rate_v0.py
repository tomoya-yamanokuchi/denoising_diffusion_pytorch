

import ray
from tqdm import tqdm
from copy import copy
from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.os_utils import save_json ,get_folder_name,create_folder,pickle_utils,get_path
from denoising_diffusion_pytorch.utils.voxel_render import pv_voxel_render,pv_voxel_render_parallel
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy ,pil_image_load_to_numpy

import numpy as np




def color_range_mask(image, mask_config):
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












if __name__ == '__main__':


    tags      = "4_3"
    time_step = 10


    # root_folder  = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_4_3_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a41_long_diffusion_cal_cost_mean_ucb_9_t20_8_1/{tags}"

    # root_folder         = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_{tags}_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a41_long_diffusion_cal_cost_mean_ucb_9_t20_8_1/epsilon_greedy_00/Boxy_1/episode_2"
    root_folder         = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_6_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_diffusion_cal_cost_mean_ucb_9_t20_8_1_tmp/oracle_obs/Boxy_1/episode_0"
    cond_image_path     = root_folder+f"/oracle_obs_cast_z_axis0.png"
    ensemble_image_path = root_folder+f"/{time_step}_seq_obs_cast_z_axis{time_step}_0.png"


    current_image  = pil_image_load_to_numpy(cond_image_path)
    ensemble_image = pil_image_load_to_numpy(ensemble_image_path)


    target_mask_b = np.asarray([0.2,0.8,0.8])
    image_mask_config_b = {"target_mask":target_mask_b,
                        "target_mask_lb":target_mask_b-np.asarray([0.1,0.1,0.1]),
                        "target_mask_ub":target_mask_b+np.asarray([0.7,0.2,0.2])}


    target_mask_G        =  np.asarray([0.0,0.0,0.0])
    image_mask_config  = {"target_mask":target_mask_G,
                                        "target_mask_lb":target_mask_G-np.asarray([0.1,0.1,0.1]),
                                        "target_mask_ub":target_mask_G+np.asarray([0.1,0.1,0.1]),}

    oracle_pixels= color_range_mask(current_image,image_mask_config_b).mean(2).sum()
    black_pixels= color_range_mask(ensemble_image,image_mask_config).mean(2).sum()

    black_image     = color_range_mask(ensemble_image,image_mask_config)
    current_region  = np.where(black_image==1.0,current_image,0.0)

    # total_volume = np.count_nonzero(current_region.sum(2) != 0.0)
    # current_target_volumes = color_range_mask(current_region,image_mask_config_b).mean(2).sum()


    total_volumes         = np.count_nonzero(current_region.mean(2) == 1.0)
    current_target_volume = np.count_nonzero(color_range_mask(current_region,image_mask_config_b).sum(2) != 0.0)

    aa= current_target_volume/(total_volumes+current_target_volume)*100
    import ipdb;ipdb.set_trace()

    # cutting_process_2d_map_flip = np.where((cond_image==np.asarray([0,0,0])),ensemble_image,np.asarray([0.,0.,0.]))
    # pil_image_save_from_numpy(cutting_process_2d_map_flip, root_folder+f"/pred_ensemble_z_{time_step}.png")
    # import ipdb;ipdb.set_trace()




    #             oracle_2d_map               = pil_image_load_to_numpy(data_folder+"/oracle_obs_cast_z_axis0.png")
    #             cutting_process_2d_map      = load_data["observations"]
    #             # cutting_process_2d_map_flip = np.where(cutting_process_2d_map==oracle_2d_map,np.asarray([0.,0.,0.]),oracle_2d_map)
    #             cutting_process_2d_map_flip = np.where((cutting_process_2d_map>=oracle_2d_map-0.05) & (cutting_process_2d_map<=oracle_2d_map+0.05),np.asarray([0.,0.,0.]),oracle_2d_map)
    #             cutting_process_2d_map_flip = cutting_process_2d_map_flip*255.0
                
    #             pil_image_save_from_numpy(cutting_process_2d_map_flip[-1]/255.0,data_folder+f"/last_remain_voxels.png")

    #             s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),
    #                                 "side_length":16}


    #             # import ipdb;ipdb.set_trace()

    #             pv_voxel_render_parallel().render_cutting_process_v3(save_path=save_folder_,
    #                                                                     s_grind_config=s_grid_config,
    #                                                                     sample_images=cutting_process_2d_map_flip)

    #             del data_folder

    # # import ipdb;ipdb.set_trace()10_seq_obs_cast_z_axis10_0.png
