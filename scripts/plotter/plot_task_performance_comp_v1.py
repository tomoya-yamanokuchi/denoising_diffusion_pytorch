


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

    save_tag   = "B32_t20_1_6000"



    # root_dir4       = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_5_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_45_12900k_v1_dataset_5_eval/PT40000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_3"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1_dataset_5_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_3"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1_dataset_4_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_3"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1_dataset_4_3_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_3"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1_dataset_4_2_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_4"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1/dataset_4_3_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_diffusion_8"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1/dataset_4_3_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_diffusion_9_t0"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1/dataset_4_2_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_diffusion_8"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1/dataset_4_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_diffusion_9"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k4/dataset_5_1_eval/PT190000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_diffusion_8_1"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v1/dataset_4_3_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_diffusion_8_1"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v2/dataset_4_3_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_diffusion_9_3"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v2/dataset_4_3_eval/PT200000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_diffusion_9_t50"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_4_3_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_diffusion_9_t20_1"
    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_4_3_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_diffusion_9_t20_1"

    # root_dir4     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v2/dataset_6_1_eval/PT200000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_diffusion_9_t20_1"
    root_dir4 = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v2/dataset_4_3_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_diffusion_cal_cost_mean_ucb_9_t20_7"


    data_dir_4      = {
                        "data_0":os.path.join(root_dir4,"epsilon_greedy_00"),
                        }
    # data_dict_4 = get_merged_dict(data_dir=data_dir_4,file_name="post_processed_data.json")
    # data_dict_4 = get_merged_dict(data_dir=data_dir_4,file_name="post_processed_data_v3.json")
    data_dict_4 = get_merged_dict(data_dir=data_dir_4,file_name="post_processed_data_v4.json")


    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_5_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_cvae_3"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1_dataset_5_eval/PT40000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_cvae_3"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1_dataset_4_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_cvae_3"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1_dataset_4_3_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_cvae_4"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1_dataset_4_2_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_cvae_4"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1/dataset_5_1_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_vaeac_8"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1/dataset_4_3_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_vaeac_8"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k1/dataset_4_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_vaeac_9"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_41424351_12900k4/dataset_4_3_eval/PT190000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_vaeac_8_1"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v1/dataset_4_3_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_vaeac_8_1"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v2/dataset_4_3_eval/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_vaeac_9_1"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v2/dataset_4_3_eval/PT200000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_vaeac_9_t"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_4_3_eval/PT200000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long_vaeac_9"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_4_3_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_vaeac_9_t20_1"

    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v2/dataset_6_1_eval/PT200000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_vaeac_9_t20_1"
    root_dir4_1      = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v2/dataset_4_3_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_vaeac_cal_cost_mean_ucb_9_t20_7"


    data_dir_4_1    = {
                        "data_0":os.path.join(root_dir4_1,"epsilon_greedy_00"),
                    }
    # data_dict_4_1 = get_merged_dict(data_dir=data_dir_4_1,file_name="post_processed_data.json")
    # data_dict_4_1 = get_merged_dict(data_dir=data_dir_4_1,file_name="post_processed_data_v3.json")
    data_dict_4_1 = get_merged_dict(data_dir=data_dir_4_1,file_name="post_processed_data_v4.json")



    data_list =[data_dict_4,data_dict_4_1]


    for idx,val in enumerate(data_dict_4_1["data_0"].keys()):



        fig = plt.figure(figsize=(int(16*1.5), int(4*1.5)))
        # fig = plt.figure()

        ax1     =    fig.add_subplot(1,3,1)
        ax2     =    fig.add_subplot(1,3,2)
        ax3     =    fig.add_subplot(1,3,3)

        width = 0.2
        shift = width/2.5
        # color_set       = ["#66cdaa","#9370db","#4682b4"]
        # color_set       = ["#66cdaa","#9370db","#4682b4","#f08080"]
        color_set       = ["#9370db"]



        # heights = [data_dict_1["data_0"][val]["reward_mean"], data_dict_2["data_0"][val]["reward_mean"], data_dict_3["data_0"][val]["reward_mean"]]
        # std     = [data_dict_1["data_0"][val]["reward_std"] , data_dict_2["data_0"][val]["reward_std"] , data_dict_3["data_0"][val]["reward_std"] ]
        heights = [data_dict_4["data_0"][val]["reward_mean"]]
        std     = [data_dict_4["data_0"][val]["reward_std"] ]
        bars    = np.arange(len(heights))-0.16
        ax1.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = color_set, capsize=10,width=-0.3)


        heights = [data_dict_4_1["data_0"][val]["reward_mean"]]
        std     = [data_dict_4_1["data_0"][val]["reward_std"] ]
        bars    = np.arange(len(heights))+0.16
        ax1.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = "#f08080", capsize=10,width=-0.3)


        heights = [data_dict_4["data_0"][val]["removal_trans"][-1],   ]
        std     = [data_dict_4["data_0"][val]["removal_trans_std"][-1]]
        bars    = np.arange(len(heights))-0.16
        ax2.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = color_set, capsize=10, width=0.3)


        heights = [data_dict_4_1["data_0"][val]["removal_trans"][-1],   ]
        std     = [data_dict_4_1["data_0"][val]["removal_trans_std"][-1]]
        bars    = np.arange(len(heights))+0.16
        ax2.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = "#f08080" ,capsize=10, width=0.3)


        heights = [data_dict_4["data_0"][val]["removal_performance"][-1]    ]
        std     = [data_dict_4["data_0"][val]["removal_performance_std"][-1]]
        bars    = np.arange(len(heights))-0.16
        ax3.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = color_set, capsize=10, width=0.3)


        heights = [data_dict_4_1["data_0"][val]["removal_performance"][-1]    ]
        std     = [data_dict_4_1["data_0"][val]["removal_performance_std"][-1]]
        bars    = np.arange(len(heights))+0.16
        ax3.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = "#f08080", capsize=10, width=0.3)


        ax1.set_xlabel(" ",fontsize=18,color="black")
        ax1.set_ylabel("Target shape cutting cost",fontsize=18,color="black")
        ax1.tick_params(labelbottom=False, labelsize=22,labelcolor="black")


        ax2.set_xlabel("",fontsize=18, color="black")
        ax2.set_ylabel("Target shape remaining rate [%]",fontsize=18,color="black")
        ax2.tick_params(labelbottom=False, labelsize=22,labelcolor="black")


        ax3.set_xlabel("",fontsize=18, color="black")
        ax3.set_ylabel("Removal performance",fontsize=18,color="black")
        ax3.tick_params(labelbottom=False, labelsize=22,labelcolor="black")


        fig.suptitle(data_dir_4["data_0"]+"\n"+
                    data_dir_4_1["data_0"],
                    fontsize= 5)


        # import ipdb;ipdb.set_trace()
        for i in range(len(data_list)):
                save_name = os.path.normpath(data_list[i]["data_path"]["data_0"]+f"/../learning_performance_comparison_{val}_{save_tag}")
                # plt.savefig(save_name+".pdf")
                # plt.savefig(save_name+".png")
                plt.savefig(save_name+"_D18000_V1062000.png")


        plt.show()
        plt.clf()
        plt.close()