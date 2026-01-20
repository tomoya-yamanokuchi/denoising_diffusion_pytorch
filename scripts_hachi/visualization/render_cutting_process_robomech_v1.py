
import os
import ray
from tqdm import tqdm
from copy import copy
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import dismantling_env
from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.os_utils import save_json ,get_folder_name,create_folder,pickle_utils,get_path
from denoising_diffusion_pytorch.utils.voxel_render import pv_voxel_render,pv_voxel_render_parallel
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy ,pil_image_load_to_numpy,numpy_to_pil
from PIL import Image

import numpy as np

# ray.init(log_to_driver=False,num_cpus=24) # for ros
ray.init(log_to_driver=False) # for ros
# ray.init(log_to_driver=False) # for ro






if __name__ == '__main__':


    # root_folder        ="/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/PT18_T1000_D64_test_v0_fix_start/oracle_obs/"
    # root_folder        = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT20000_T1000_D64_test_v1_fix_start/epsilon_greedy_00"
    # root_folder        = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT6000000_T1000_D64_test_v5_fix_start_single_step/epsilon_greedy_00"
    # root_folder ="/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT600000_T1000_D64_test_v5_fix_start_single_step/random

    # root_folder ="/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT100000_T1000_D64_test_v5_fix_start_multi_step/epsilon_greedy_00"
    # root_folder ="/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT600000_T1000_D64_test_v6_fix_start_multi_step/epsilon_greedy_00"

    # root_folder ="/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step/epsilon_greedy_00"

    # root_folder ="/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step/oracle_obs"
    # root_folder ="/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_no_separate/oracle_obs"
    # root_folder ="/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_2_target/oracle_obs"
    # root_folder ="/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_/epsilon_greedy_00"
    # root_folder   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step__2_target/epsilon_greedy_00"
    # root_folder   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1/dataset_4_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_diffusion_8/epsilon_greedy_00"
    # root_folder   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1/dataset_4_2_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_diffusion_8/epsilon_greedy_00"
    # root_folder   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1/dataset_4_2_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_vaeac_8_1/epsilon_greedy_00"
    # root_folder = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k4/dataset_4_3_eval/PT190000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_vaeac_8_1/epsilon_greedy_00"

    # root_folder    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k4/dataset_4_2_eval/PT190000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_vaeac_8_1/epsilon_greedy_00"
    # root_folder    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k4/dataset_4_2_eval/PT190000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_vaeac_8_1/epsilon_greedy_00"

    #///////////////////////////////////////////////////
    # tags = "epsilon_greedy_00"
    # tags = "no_cond"
    tags = "oracle_obs"
    # tags = "random"


    # root_folder  = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_5_1_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_diffusion_9_t20_1/{tags}"
    # root_folder  = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_6_1_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_vaeac_9_t20_1/{tags}"
    # root_folder  = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_4_3_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_diffusion_9_t20_1/{tags}"

    # root_folder  = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v2/dataset_4_3_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_diffusion_cal_cost_mean_ucb_9_t20_7/{tags}"


    # root_folder  = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_4_3_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a41_long_diffusion_cal_cost_mean_ucb_9_t20_8_1/{tags}"
    # root_folder  = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans_1d/dataset_4142435161_13901k_v1/dataset_6_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_diffusion_1D_cal_cost_mean_ucb_9_t20_8_1/{tags}"

    # root_folder = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac_plans/dataset_4142435161_13901k_v1/dataset_6_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_vaeac_cal_cost_mean_ucb_9_t20_8_3/{tags}"
    # root_folder = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_6_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_diffusion_cal_cost_mean_ucb_9_t20_8_3/{tags}"

    # root_folder = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac_plans/dataset_4142435161_13901k_v1/dataset_4_3_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a0_long_vaeac_cal_cost_mean_ucb_9_t20_8_3_tmp/{tags}"
    # root_folder = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac_plans/dataset_4142435161_13901k_v1/dataset_4_3_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a0_long_vaeac_cal_cost_mean_ucb_9_t20_8_3_tmp1/{tags}"
    root_folder = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac_plans/dataset_4142435161_13901k_v1/dataset_4_3_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a0_long_vaeac_cal_cost_mean_ucb_9_t20_8_3_tmp2/{tags}"



    # dim_2D = 256
    # dim_3D = 32

    # dim_2D = 256
    # dim_3D = 64

    dim_2D = 512
    dim_3D = 64

    # dim_2D = 343
    # dim_3D = 49

    # load_file = "gt_img_rs_001.png"
    load_file = "gt_img_512.png"

    # render_type = "Render_0"
    # render_type = "Render_1"
    render_type = "Render_3"


# Render0: {plotter.add_points(points, color = [0.8,0.8,0.8], point_size = 0.8, opacity = 0.3): False,
#            view_offset_val : -0.1 ,
#           null_shape_transparency = 0.0}



# Render1: {plotter.add_points(points, color = [0.8,0.8,0.8], point_size = 0.8, opacity = 0.3): False,
#            view_offset_val : 0.2 ,
#           null_shape_transparency = 0.008}


# Render1: {plotter.add_points(points, color = [0.8,0.8,0.8], point_size = 0.8, opacity = 0.3): True,
#            view_offset_val : 0.2 ,
#           null_shape_transparency = 0.008}


    s_grid_config = {"bounds":(-0.3,0.3,-0.3,0.3,-0.3,0.3), 'side_length':dim_3D}
    save_name = f"{render_type}"

    # __new__ を使って __init__ を呼ばずにインスタンスを作成
    cutting_env = object.__new__(dismantling_env)
    action_table = cutting_env.get_action_table(s_grid_config)

    data_folder ="/home/haxhi/workspace/denoising_diffusion_pytorch/logs/for_robomech2025/"
    print(f"load_data:{data_folder}")
    save_tag = os.path.splitext(load_file)[0]
    save_folder_    = data_folder+ f"/3d_cutting_process/{save_tag}_2D{dim_2D}_3D{dim_3D}/{render_type}/"
    create_folder(save_folder_)

    action = np.asarray([ 0, 0])
    # image = pil_image_load_to_numpy(data_folder+f"/gt_img_rs_001.png",resize=(dim_2D,dim_2D),channel_type="RGB")
    # image = pil_image_load_to_numpy(data_folder+f"{load_file}",resize=(dim_2D,dim_2D),channel_type="RGB")
    image = pil_image_load_to_numpy(data_folder+f"{load_file}",resize=None,channel_type="RGB")
    cutting_process_2d_map_flip = np.tile(image, (action.shape[0], 1, 1, 1))*255.0


    # import pyvista as pv
    # import copy
    # from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array
    # tmp_mesh = pv.Box(bounds=(s_grid_config["bounds"][0],
    #                         s_grid_config["bounds"][1],
    #                         s_grid_config["bounds"][2],
    #                         s_grid_config["bounds"][3],
    #                         s_grid_config["bounds"][4],
    #                         s_grid_config["bounds"][5],
    #                         ))

    # box_array_handler   = pv_box_array(grid_config=s_grid_config)
    # _                   = box_array_handler.cast_mesh_to_box_array(mesh=copy.copy(tmp_mesh))
    # box_arrays_data     = box_array_handler.get_box_array_data()

    # box_array_handler.cast_2d_image_to_box_color(image=cutting_process_2d_map_flip[0],permute="z")

    # import ipdb;ipdb.set_trace()

    pv_voxel_render_parallel().render_cutting_process_as_pcl_v1(save_path=save_folder_,
                                                            s_grind_config=s_grid_config,
                                                            action = action,
                                                            action_table = action_table,
                                                            sample_images=cutting_process_2d_map_flip,
                                                            save_tag=save_name)

    exit()

