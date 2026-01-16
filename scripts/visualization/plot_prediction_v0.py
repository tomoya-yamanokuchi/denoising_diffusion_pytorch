

import ray
from tqdm import tqdm
from copy import copy
from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.os_utils import save_json ,get_folder_name,create_folder,pickle_utils,get_path
from denoising_diffusion_pytorch.utils.voxel_render import pv_voxel_render,pv_voxel_render_parallel
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy ,pil_image_load_to_numpy

import numpy as np

if __name__ == '__main__':


    tags      = "4_3"
    time_step = 14


    # root_folder  = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_4_3_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a41_long_diffusion_cal_cost_mean_ucb_9_t20_8_1/{tags}"

    root_folder         = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_{tags}_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a41_long_diffusion_cal_cost_mean_ucb_9_t20_8_1/epsilon_greedy_00/Boxy_1/episode_2"
    cond_image_path     = root_folder+f"/conditions/seq_obs_cast_{time_step}_axis_z_0.png"
    ensemble_image_path = root_folder+f"/{time_step}_ensemble_z_axis{time_step}_0.png"


    cond_image  = pil_image_load_to_numpy(cond_image_path)
    ensemble_image = pil_image_load_to_numpy(ensemble_image_path)

    cutting_process_2d_map_flip = np.where((cond_image==np.asarray([0,0,0])),ensemble_image,np.asarray([0.,0.,0.]))
    pil_image_save_from_numpy(cutting_process_2d_map_flip, root_folder+f"/pred_ensemble_z_{time_step}.png")
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

    # # import ipdb;ipdb.set_trace()
