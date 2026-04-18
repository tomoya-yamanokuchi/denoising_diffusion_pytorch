


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.style.use('ggplot')



from denoising_diffusion_pytorch.utils.os_utils import load_json







def get_merged_dict(data_dir,file_name):
    data_dict = {}
    for key,data in data_dir.items():
        data_path = os.path.join(data,file_name)
        data = load_json(data_path)
        data_dict[key]=data
    data_dict["data_path"] =  data_dir
    return data_dict












if __name__ == '__main__':

    save_tag   = "v1"



    root_dir1       = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_5_1_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_diffusion_9_t20_1"
    root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_5_1_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_vaeac_9_t20_1"


    # root_dir1       = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_6_1_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_diffusion_9_t20_1"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_6_1_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_vaeac_9_t20_1"

    # root_dir1       = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_4_3_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_diffusion_9_t20_1"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_4_3_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_vaeac_9_t20_1"


    data_dir_4 ={
        "data_0":os.path.join(root_dir1,"epsilon_greedy_00"),
        }
    data_dir_3 ={
        "data_0":os.path.join(root_dir1,"oracle_obs"),
        }
    data_dir_2 ={
        "data_0":os.path.join(root_dir1,"no_cond"),
        }

    data_dir_1 ={
        "data_0":os.path.join(root_dir1,"random"),
        }

    data_dict_4 = get_merged_dict(data_dir=data_dir_4,file_name="post_processed_data.json")
    data_dict_3 = get_merged_dict(data_dir=data_dir_3,file_name="post_processed_data.json")
    data_dict_2 = get_merged_dict(data_dir=data_dir_2,file_name="post_processed_data.json")
    data_dict_1 = get_merged_dict(data_dir=data_dir_1,file_name="post_processed_data.json")


    data_dir_4_1    = {
                        "data_0":os.path.join(root_dir4_1,"epsilon_greedy_00"),
                    }
    data_dict_5 = get_merged_dict(data_dir=data_dir_4_1,file_name="post_processed_data.json")




    # data_list =[data_dict_4,data_dict_4_1]





    for idx,val in enumerate(data_dict_5["data_0"].keys()):

        save_name = os.path.normpath(data_dict_4["data_path"]["data_0"]+f"/../task_performance_comparison_{val}_{save_tag}")
        with open(f"{save_name}.txt", "w") as file:

            print(f"Random          & {data_dict_1['data_0'][val]['success_rate']} & {np.round(data_dict_1['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_1['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_1['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_1['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
            print(f"Vaeac           & {data_dict_5['data_0'][val]['success_rate']} & {np.round(data_dict_5['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_5['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_5['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_5['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
            print(f"Proposed-Nocond & {data_dict_2['data_0'][val]['success_rate']} & {np.round(data_dict_2['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_2['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_2['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_2['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
            print(f"Proposed        & {data_dict_4['data_0'][val]['success_rate']} & {np.round(data_dict_4['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_4['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_4['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_4['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
            print(f"Proposed-GT     & {data_dict_3['data_0'][val]['success_rate']} & {np.round(data_dict_3['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_3['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_3['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_3['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
            print(f"================================================================================",file=file)
            print(f"Random          & {np.round(data_dict_1['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_1['data_0'][val]['reward_std'],2)} & {np.round(data_dict_1['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_1['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_1['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_1['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
            print(f"Vaeac           & {np.round(data_dict_5['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_5['data_0'][val]['reward_std'],2)} & {np.round(data_dict_5['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_5['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_5['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_5['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
            print(f"Proposed-Nocond & {np.round(data_dict_2['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_2['data_0'][val]['reward_std'],2)} & {np.round(data_dict_2['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_2['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_2['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_2['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
            print(f"Proposed        & {np.round(data_dict_4['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_4['data_0'][val]['reward_std'],2)} & {np.round(data_dict_4['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_4['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_4['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_4['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
            print(f"Proposed-GT     & {np.round(data_dict_3['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_3['data_0'][val]['reward_std'],2)} & {np.round(data_dict_3['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_3['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_3['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_3['data_0'][val]['removal_performance_std'][-1],2)}",file=file)



    # import ipdb;ipdb.set_trace()

