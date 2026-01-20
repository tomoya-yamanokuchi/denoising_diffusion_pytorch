


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

    # save_tag   = "v7"
    # save_tag   = "v2_2"
    # save_tag   = "v8_9"
    # save_tag   = "v10_2"
    # save_tag   = "v10_2_cond_test"
    # save_tag   = "v10_2_learing_performance comparison"
    save_tag   = "v10_2_mask_cond_test_repaint_cond_v4"
    # save_tag   = "v10_2_mask_cond_test"
    





    # root_dir1       = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_5_1_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_diffusion_9_t20_1"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_5_1_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_vaeac_9_t20_1"


    # root_dir1       = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_6_1_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_diffusion_9_t20_1"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_6_1_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_vaeac_9_t20_1"

    # root_dir1       = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_4_3_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_diffusion_9_t20_1"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_4_3_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_vaeac_9_t20_1"



    # root_dir1       = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_5_1_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_diffusion_9_t20_1"
    # root_dir4_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_5_1_eval/PT600000_T1000_D64_B32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_vaeac_9_t20_1"

    # root_dir1   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v2/dataset_6_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_diffusion_cal_cost_mean_ucb_9_t20_7"
    # root_dir4_1 = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v2/dataset_6_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_vaeac_cal_cost_mean_ucb_9_t20_7_2"


    ###########################################
    #######         latest folder       #######
    ###########################################

    # root_dir1   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_3/dataset_4_3_eval/PT102000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp5"
    # root_dir2   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_3/dataset_4_3_eval/PT102000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp6"
    
    # root_dir1   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_3/dataset_4_3_eval/PT102000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp7_1"
    # root_dir2   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_3/dataset_4_3_eval/PT102000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp7_2"
    # root_dir3   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_3/dataset_4_3_eval/PT102000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp7_3"


    
    # root_dir1   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_3/dataset_4_3_eval/PT102000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp9_6"
    # root_dir2   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_3/dataset_4_3_eval/PT102000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp9_6"
    # root_dir3   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_3/dataset_4_3_eval/PT102000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp9_7"


    # root_dir1   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_1/dataset_4_3_eval/PT81000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp10_1"
    # root_dir2   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_1/dataset_4_3_eval/PT81000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp10_2"
    # root_dir3 = root_dir2


    # root_dir1   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_1/dataset_4_3_eval/PT81000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp10_1"
    # root_dir2   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_1/dataset_4_3_eval/PT81000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp10_1_tmp"
    # root_dir3   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_1/dataset_4_3_eval/PT81000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp10_2"


    # root_dir1   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_3/dataset_4_3_eval/PT102000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp11_1"
    # root_dir2   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_3/dataset_4_3_eval/PT102000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp11_1"
    # root_dir3   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_3/dataset_4_3_eval/PT102000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp11_1"


    # root_dir1   = "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_1/dataset_4_3_eval/PT50000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp10_1"
    # root_dir2 =   "/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_1/dataset_4_3_eval/PT81000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp10_1"
    # root_dir3   = root_dir2


    # root_dir1   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_5_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_diffusion_cal_cost_mean_ucb_9_t20_8_3"
    # root_dir2   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_5_1_eval/PT20000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_conditional_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp12_1"
    # root_dir3   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v1_3/dataset_5_1_eval/PT102000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a19_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp11_1"


    # root_dir1   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_5_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_diffusion_cal_cost_mean_ucb_9_t20_8_3"
    # root_dir5 = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac_plans/dataset_4142435161_13901k_v1/dataset_5_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_vaeac_cal_cost_mean_ucb_9_t20_8_3"
    # root_dir9 = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans_1d/dataset_4142435161_13901k_v1/dataset_5_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_diffusion_1D_cal_cost_mean_ucb_9_t20_8_3"
    # root_dir10   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_5_1_eval/PT20000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_conditional_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp12_1"


    root_dir1   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_6_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_diffusion_cal_cost_mean_ucb_9_t20_8_3"
    root_dir5 = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/vaeac_plans/dataset_4142435161_13901k_v1/dataset_6_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_vaeac_cal_cost_mean_ucb_9_t20_8_3"
    root_dir9 = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans_1d/dataset_4142435161_13901k_v1/dataset_6_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_diffusion_1D_cal_cost_mean_ucb_9_t20_8_3"
    root_dir10   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_6_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_conditional_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp13_3"
    root_dir11   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_6_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_diffusion_cal_cost_mean_ucb_9_t20_8_3_tmp13_2"


    data_dir_1 ={
        "data_0":os.path.join(root_dir1,"random"),
        }

    data_dir_2 ={
        "data_0":os.path.join(root_dir10,"no_cond"),
        }

    data_dir_3 ={
        "data_0":os.path.join(root_dir1,"oracle_obs"),
        }

    data_dir_4 ={
        "data_0":os.path.join(root_dir11,"epsilon_greedy_00"),
        # "data_0":os.path.join(root_dir1,"epsilon_greedy_00"),
        }



    data_dir_5 ={
        "data_0":os.path.join(root_dir5,"epsilon_greedy_00"),
        }


    data_dir_9 ={
        "data_0":os.path.join(root_dir9,"epsilon_greedy_00"),
        }

    data_dir_10 ={
        "data_0":os.path.join(root_dir10,"epsilon_greedy_00"),
        }



    data_dict_1 = get_merged_dict(data_dir=data_dir_1,file_name="post_processed_data_v5.json")
    data_dict_2 = get_merged_dict(data_dir=data_dir_2,file_name="post_processed_data_v5.json")
    data_dict_3 = get_merged_dict(data_dir=data_dir_3,file_name="post_processed_data_v5.json")
    data_dict_4 = get_merged_dict(data_dir=data_dir_4,file_name="post_processed_data_v5.json")
    data_dict_5 = get_merged_dict(data_dir=data_dir_5,file_name="post_processed_data_v5.json")
    data_dict_9 = get_merged_dict(data_dir=data_dir_9,file_name="post_processed_data_v5.json")
    data_dict_10 = get_merged_dict(data_dir=data_dir_10,file_name="post_processed_data_v5.json")



    # data_list =[data_dict_1,data_dict_2]
    data_list =[data_dict_1,data_dict_2,data_dict_3,data_dict_4,data_dict_5,data_dict_9,data_dict_10]
    # data_list =[data_dict_1,data_dict_2,]


    for idx,val in enumerate(data_dict_1["data_0"].keys()):

        fig = plt.figure(figsize=(22, 16))

        ax1     =    fig.add_subplot(3,2,1)
        ax2     =    fig.add_subplot(3,2,2)
        ax3     =    fig.add_subplot(3,2,3)
        ax4     =    fig.add_subplot(3,2,4)
        ax5     =    fig.add_subplot(3,2,5)
        ax6     =    fig.add_subplot(3,2,6)


        width = 0.3
        shift = width/2
        color_set       = ["#66cdaa","#9370db","#4682b4", "#f4a460",'#808080',"#db7093" , "blue"]
        # color_set       = ["#66cdaa","#9370db", "#f4a460", "#db7093",'#808080']
        # color_set       = ["#66cdaa",'#808080']
        # color_set       = ["#66cdaa","#9370db",'#808080']

        # import ipdb;ipdb.set_trace()


        # heights = [data_dict_1["data_0"][val]["reward_mean"], data_dict_2["data_0"][val]["reward_mean"],data_dict_3["data_0"][val]["reward_mean"]]
        # std     = [data_dict_1["data_0"][val]["reward_std"] , data_dict_2["data_0"][val]["reward_std"] ,data_dict_3["data_0"][val]["reward_std"] ]
        heights = [d["data_0"][val]["reward_mean"] for d in data_list]
        std     = [d["data_0"][val]["reward_std"]  for d in data_list]
        bars    = np.arange(len(heights))
        ax1.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = color_set, capsize=10)


        for i in range(len(data_list)):

            cost_mean = np.asarray(data_list[i]["data_0"][val]["removal_trans"])
            cost_std  = np.asarray(data_list[i]["data_0"][val]["removal_trans_std"])
            iteration = np.arange(cost_mean.shape[0])

            lb_std = cost_mean-cost_std
            ub_std = cost_mean+cost_std

            colors = color_set[i]
            ax2.plot(iteration,cost_mean,linestyle = '--', color =colors)
            ax2.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
            ax2.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
            ax2.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)



            cost_mean = np.asarray(data_list[i]["data_0"][val]["removal_performance"])
            cost_std  = np.asarray(data_list[i]["data_0"][val]["removal_performance_std"])
            iteration = np.arange(cost_mean.shape[0])

            lb_std = cost_mean-cost_std
            ub_std = cost_mean+cost_std

            colors = color_set[i]
            ax3.plot(iteration,cost_mean,linestyle = '--', color =colors)
            ax3.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
            ax3.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
            ax3.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)

            # import ipdb;ipdb.set_trace()
            if "no_cond" in data_list[i]["data_path"]["data_0"] or "epsilon_greedy_00" in data_list[i]["data_path"]["data_0"] :

                if "vaeac" in data_list[i]["data_path"]["data_0"] is not True:
                    pass
                elif "diffusion_plans_1d" in data_list[i]["data_path"]["data_0"] is not True:
                    pass
                else:
                    cost_mean = np.asarray(data_list[i]["data_0"][val]["image_pred_loss"])
                    cost_std  = np.asarray(data_list[i]["data_0"][val]["image_pred_loss_std"])
                    # cost_mean = np.asarray(data_list[i]["data_0"][val]["image_pred_loss_mode"])
                    # cost_std  = np.asarray(data_list[i]["data_0"][val]["image_pred_loss_mode_std"])
                    iteration = np.arange(cost_mean.shape[0])

                    lb_std = cost_mean-cost_std
                    ub_std = cost_mean+cost_std

                    colors = color_set[i]
                    ax4.plot(iteration,cost_mean,linestyle = '--', color =colors)
                    ax4.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
                    ax4.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
                    ax4.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)
                    ax4.set_xticks( np.arange(start=0, stop=15, step=2))


            cost_mean = np.asarray(data_list[i]["data_0"][val]["augnet_loss"])
            cost_std  = np.asarray(data_list[i]["data_0"][val]["augnet_loss_std"])
            iteration = np.arange(cost_mean.shape[0])

            lb_std = cost_mean-cost_std
            ub_std = cost_mean+cost_std

            colors = color_set[i]
            ax5.plot(iteration,cost_mean,linestyle = '--', color =colors)
            ax5.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
            ax5.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
            ax5.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)


            cost_mean = np.asarray(data_list[i]["data_0"][val]["augnet_loss_2"])
            cost_std  = np.asarray(data_list[i]["data_0"][val]["augnet_loss_2_std"])
            iteration = np.arange(cost_mean.shape[0])

            lb_std = cost_mean-cost_std
            ub_std = cost_mean+cost_std

            colors = color_set[i]
            ax6.plot(iteration,cost_mean,linestyle = '--', color =colors)
            ax6.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
            ax6.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
            ax6.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)


        # ax1.set_xlabel(" ",fontsize=18,color="black")
        # ax1.set_ylabel("Target shape cutting cost",fontsize=18,color="black")
        # ax1.tick_params(labelbottom=False, labelsize=22,labelcolor="black")


        # ax2.set_xlabel("Time step ",fontsize=18, color="black")
        # ax2.set_ylabel("Target shape remaining rate [%]",fontsize=18,color="black")
        # ax2.tick_params(labelsize=22,labelcolor="black")
        # ax2.set_ylim(0,110)



        # ax3.set_xlabel("Time step ",fontsize=18, color="black")
        # ax3.set_ylabel("Removal performance",fontsize=18,color="black")
        # ax3.tick_params(labelsize=24,labelcolor="black")


        # ax4.set_xlabel("Time step ",fontsize=18, color="black")
        # ax4.set_ylabel("Reconstruction loss",fontsize=18,color="black")
        # ax4.tick_params(labelsize=24,labelcolor="black")
        ax4.tick_params(labelsize=24,labelcolor="black")
        


        # ax5.set_xlabel("Time step ",fontsize=18, color="black")
        # # ax5.set_ylabel("augnet loss",fontsize=18,color="black")
        # ax5.set_ylabel("SSIM loss",fontsize=18,color="black")
        # ax5.tick_params(labelsize=22,labelcolor="black")



        # ax6.set_xlabel("Time step ",fontsize=18, color="black")
        # ax6.set_ylabel("augnet loss",fontsize=18,color="black")
        # ax6.tick_params(labelsize=22,labelcolor="black")


        title = []
        for i in range(len(data_list)):
            title.append(data_list[i]["data_path"]["data_0"])
        title = "\n".join(title)


        fig.suptitle(title,
                    fontsize= 8)


        for i in range(len(data_list)):
                save_name = os.path.normpath(data_list[i]["data_path"]["data_0"]+f"/../reward_comparison_{val}_{save_tag}")
                # plt.savefig(save_name+".png")
                plt.savefig(save_name+".pdf")
                

        plt.show()
        plt.clf()
        plt.close()

        # import ipdb;ipdb.set_trace()

        for i in range(len(data_list)):
            save_name = os.path.normpath(data_list[i]["data_path"]["data_0"]+f"/../task_performance_comparison_{val}_{save_tag}")
            with open(f"{save_name}.txt", "w") as file:

                print(f"{title}",file = file)
                print(f"Random          & {data_dict_1['data_0'][val]['success_rate']} & {np.round(data_dict_1['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_1['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_1['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_1['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
                print(f"Vaeac           & {data_dict_5['data_0'][val]['success_rate']} & {np.round(data_dict_5['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_5['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_5['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_5['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
                print(f"Proposed-Nocond & {data_dict_2['data_0'][val]['success_rate']} & {np.round(data_dict_2['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_2['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_2['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_2['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
                print(f"Proposed        & {data_dict_4['data_0'][val]['success_rate']} & {np.round(data_dict_4['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_4['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_4['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_4['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
                print(f"Proposed-GT     & {data_dict_3['data_0'][val]['success_rate']} & {np.round(data_dict_3['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_3['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_3['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_3['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
                print(f"================================================================================",file=file)
                print(f"Random          & {np.round(data_dict_1['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_1['data_0'][val]['reward_std'],2)} & {np.round(data_dict_1['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_1['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_1['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_1['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
                print(f"Vaeac           & {np.round(data_dict_5['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_5['data_0'][val]['reward_std'],2)} & {np.round(data_dict_5['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_5['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_5['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_5['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
                print(f"3D_Diffusion    & {np.round(data_dict_9['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_9['data_0'][val]['reward_std'],2)} & {np.round(data_dict_9['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_9['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_9['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_9['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
                print(f"Proposed-Nocond & {np.round(data_dict_2['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_2['data_0'][val]['reward_std'],2)} & {np.round(data_dict_2['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_2['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_2['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_2['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
                print(f"Proposed        & {np.round(data_dict_4['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_4['data_0'][val]['reward_std'],2)} & {np.round(data_dict_4['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_4['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_4['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_4['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
                print(f"Proposed w/ mask& {np.round(data_dict_10['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_10['data_0'][val]['reward_std'],2)} & {np.round(data_dict_10['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_10['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_10['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_10['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
                print(f"Proposed-GT     & {np.round(data_dict_3['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_3['data_0'][val]['reward_std'],2)} & {np.round(data_dict_3['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_3['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_3['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_3['data_0'][val]['removal_performance_std'][-1],2)}",file=file)



    # import ipdb;ipdb.set_trace()



    # aa = np.asarray([data_dict_4["data_0"]["Boxy_0"]["image_pred_loss"],
    #                  data_dict_4["data_0"]["Boxy_1"]["image_pred_loss"],
    #                  data_dict_4["data_0"]["Boxy_2"]["image_pred_loss"]])

    # bb = np.asarray([data_dict_2["data_0"]["Boxy_0"]["image_pred_loss"],
    #                  data_dict_2["data_0"]["Boxy_1"]["image_pred_loss"],
    #                  data_dict_2["data_0"]["Boxy_2"]["image_pred_loss"]])


    # fig = plt.figure(figsize=(25, 15))

    # ax1     =    fig.add_subplot(3,2,1)

    # cost_mean = aa.mean(0)
    # cost_std  = aa.std(0)
    # iteration = np.arange(cost_mean.shape[0])

    # lb_std = cost_mean-cost_std
    # ub_std = cost_mean+cost_std

    # ax1.plot(iteration,cost_mean,linestyle = '--', color =colors)
    # ax1.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
    # ax1.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
    # ax1.fill_between(iteration, ub_std, lb_std,alpha=0.5,color ="r")
    
    # cost_mean = bb.mean(0)
    # cost_std  = bb.std(0)
    # iteration = np.arange(cost_mean.shape[0])

    # lb_std = cost_mean-cost_std
    # ub_std = cost_mean+cost_std

    # ax1.plot(iteration,cost_mean,linestyle = '--', color =colors)
    # ax1.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
    # ax1.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
    # ax1.fill_between(iteration, ub_std, lb_std,alpha=0.5,color ="b")
    
    # plt.show()