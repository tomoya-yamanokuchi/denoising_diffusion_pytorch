

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


def get_action_table(grid_config):
    """_summary_
        define slice action index

    Args:
        grid_config (dict)
    Returns:
        action table (dict): {i:{"axis":data_order[val],"loc":j}})
        i    : Serial number of the action index
        axis : axis name
        loc  : slice index
        In the current configuration, Data_order is unified as [“Z”, “X”, “Y”].
    """


    """Creates an action table that maps action indices to slice operations. In the current configuration, Data_order is unified as [“z”, “x”, “y”].

    Args:
        grid_config (dict): Configuration dictionary for the voxel grid.

    Returns:
        dict: A table mapping action indices to action descriptions.
            Each action includes the axis (e.g., "z", "x", "y") and the slice location.

    Examples:
        >>> action table (dict): {i:{"axis":data_order[val],"loc":j}})
        >>> i    : Serial number of the action index
        >>> axis : axis name
        >>> loc  : slice index
    """

    image_length = grid_config["side_length"]
    action_table  = {}

    i   = 0
    # data_order = ["x","y","z"]
    data_order = ["z","x","y"]
    # data_order = ["z","y","x"]
    for val in range(len(data_order)):
        for j in range(image_length):
            action_table.update({i:{"axis":data_order[val],"loc":j}})
            i+=1

    return action_table



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


    # dim_2D = 512
    # dim_3D = 64

    dim_2D = 343
    dim_3D = 49
    save_name = f"dim_{dim_3D}_no_dot"
    s_grid_config = {"bounds":(-0.3,0.3,-0.3,0.3,-0.3,0.3), 'side_length':dim_3D}

    # __new__ を使って __init__ を呼ばずにインスタンスを作成
    cutting_env = object.__new__(dismantling_env)
    action_table = cutting_env.get_action_table(s_grid_config)


    model_type_folders = get_folder_name(root_folder)
    for j in range(len(model_type_folders)):

        episodes_folder = root_folder+"/"+model_type_folders[j]
        episodes        = get_folder_name(episodes_folder)

        # for i in tqdm(range(len(episodes))):
        for i in tqdm(range(1)):
        # for i in tqdm(range(5)):

                data_folder     = episodes_folder+"/"+episodes[i]
                print(f"load_data:{data_folder}")
                save_folder_    = data_folder+ "/3d_cutting_process"
                create_folder(save_folder_)

                load_data = pickle_utils().load(load_path=data_folder+f"/rollout_data.pickle")

                oracle_2d_map               = pil_image_load_to_numpy(data_folder+"/oracle_obs_cast_z_axis0.png",resize=(dim_2D,dim_2D))
                cutting_process_2d_map      = load_data["observations"]
                action                      = load_data["actions"]
                print(action)

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

                # s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),"side_length":16}
                # s_grid_config = {"bounds":(-0.3,0.3,-0.3,0.3,-0.3,0.3), 'side_length':32}

                # previous ver
                # pv_voxel_render_parallel().render_cutting_process_v3(save_path=save_folder_,
                #                                                         s_grind_config=s_grid_config,
                #                                                         sample_images=cutting_process_2d_map_flip)

                pv_voxel_render_parallel().render_cutting_process_as_pcl_v1(save_path=save_folder_,
                                                                        s_grind_config=s_grid_config,
                                                                        action = action,
                                                                        action_table = action_table,
                                                                        sample_images=cutting_process_2d_map_flip,
                                                                        save_tag=save_name)

                del data_folder

    # import ipdb;ipdb.set_trace()
