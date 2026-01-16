


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# import seaborn as sns


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
    # import ipdb;ipdb.set_trace()

    data_dict["data_path"] =  data_dir
    return data_dict












if __name__ == '__main__':




    # root_dir9_0    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT600000_T1000_D64_test_v6_fix_start_multi_step"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT600000_T1000_D64_test_v6_fix_start_multi_step"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_3_partial_obs"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_3_full_obs"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_3_1_partial_obs"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_3_2_full_obs_a32"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_3_2_partial_obs_a32"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_3_3_partial_obs_a44"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_3_3_full_obs_a44"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_3_3_partial_obs_a3"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_3_3_partial_obs_a_r"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_3_4_full_obs_a32_long"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_3_5_full_obs_a4_long"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_3_5_partial_obs_a4_long"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v2/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_full_obs_a4_long"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v2/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_5_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_0_0_partial_obs_a32_long"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13900k_v5/dataset_6_1_eval/PT80000_T1000_D64_32_test_v5_fix_start_multi_step_0_0_partial_obs_a4_long_diffusion_9_t20_1"
    root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1/dataset_6_1_eval/PT200000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_diffusion_cal_cost_mean_ucb_9_t20_8_1"
    root_dir9_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans_1d/dataset_4142435161_13901k_v1/dataset_6_1_eval/PT20000_T1000_D64_B32_test_v6_fix_start_multi_step_partial_obs_a4_long_diffusion_1D_cal_cost_mean_ucb_9_t20_8_1"

    
    
    


    # root_dir9_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step__no_separate"
    # root_dir9_2     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step__2_target"




    save_tag    = "v3"
    # save_tag    = "tmp"
    # save_tag    = "0_partial"



    data_dir_6 ={
        "data_0":os.path.join(root_dir9_0,"oracle_obs"),
        }
    data_dir_5 ={
        "data_0":os.path.join(root_dir9_0,"epsilon_greedy_00"),
        }

    data_dir_4 ={
        "data_0":os.path.join(root_dir9_1,"epsilon_greedy_00"),
        }

    data_dir_3 ={
        "data_0":os.path.join(root_dir9_0,"epsilon_greedy_00"),
        }

    data_dir_2 ={
        "data_0":os.path.join(root_dir9_0,"no_cond"),
        }

    data_dir_1 ={
        "data_0":os.path.join(root_dir9_0,"random"),
        }


    data_dict_1 = get_merged_dict(data_dir=data_dir_1,file_name="post_processed_data_v5.json")
    data_dict_2 = get_merged_dict(data_dir=data_dir_2,file_name="post_processed_data_v5.json")
    data_dict_3 = get_merged_dict(data_dir=data_dir_3,file_name="post_processed_data_v5.json")
    data_dict_4 = get_merged_dict(data_dir=data_dir_4,file_name="post_processed_data_v5.json")
    data_dict_5 = get_merged_dict(data_dir=data_dir_5,file_name="post_processed_data_v5.json")
    data_dict_6 = get_merged_dict(data_dir=data_dir_6,file_name="post_processed_data_v5.json")



    # data_list =[data_dict_1,data_dict_2]
    # data_list =[data_dict_1,data_dict_2,data_dict_3,data_dict_4,data_dict_5]
    data_list =[data_dict_1,data_dict_2,data_dict_3,data_dict_4,data_dict_5,data_dict_6]
    # data_list =[data_dict_5]


    # import ipdb;ipdb.set_trace()



    for idx,val in enumerate(data_dict_1["data_0"].keys()):



        fig = plt.figure(figsize=(35, 20))

        ax1     =    fig.add_subplot(3,3,1)
        ax2     =    fig.add_subplot(3,3,2)
        ax3     =    fig.add_subplot(3,3,3)
        ax4     =    fig.add_subplot(3,3,4)
        ax5     =    fig.add_subplot(3,3,5)
        ax6     =    fig.add_subplot(3,3,6)
        ax7     =    fig.add_subplot(3,3,7)
        ax8     =    fig.add_subplot(3,3,8)
        ax9     =    fig.add_subplot(3,3,9)


        width = 0.3
        shift = width/2
        color_set       = ["#66cdaa","#9370db","#4682b4", "#f4a460", "#db7093",'#808080']
        # color_set       = ["#66cdaa","#9370db", "#f4a460", "#db7093",'#808080']
        # color_set       = ["#66cdaa",'#808080']


        heights = [data_dict_1["data_0"][val]["reward_mean"], data_dict_2["data_0"][val]["reward_mean"], data_dict_3["data_0"][val]["reward_mean"],data_dict_4["data_0"][val]["reward_mean"],data_dict_5["data_0"][val]["reward_mean"],data_dict_6["data_0"][val]["reward_mean"]]
        std     = [data_dict_1["data_0"][val]["reward_std"] , data_dict_2["data_0"][val]["reward_std"] , data_dict_3["data_0"][val]["reward_std"] ,data_dict_4["data_0"][val]["reward_std"] ,data_dict_5["data_0"][val]["reward_std"] ,data_dict_6["data_0"][val]["reward_std"] ]
        # heights = [data_dict_1["data_0"][val]["reward_mean"], data_dict_2["data_0"][val]["reward_mean"], data_dict_3["data_0"][val]["reward_mean"],data_dict_4["data_0"][val]["reward_mean"],data_dict_5["data_0"][val]["reward_mean"]]
        # std     = [data_dict_1["data_0"][val]["reward_std"] , data_dict_2["data_0"][val]["reward_std"] , data_dict_3["data_0"][val]["reward_std"] ,data_dict_4["data_0"][val]["reward_std"] ,data_dict_5["data_0"][val]["reward_std"] ]
        # heights = [data_dict_1["data_0"][val]["reward_mean"], data_dict_2["data_0"][val]["reward_mean"]]
        # std     = [data_dict_1["data_0"][val]["reward_std"] , data_dict_2["data_0"][val]["reward_std"] ]
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


            cost_mean = np.asarray(data_list[i]["data_0"][val]["image_pred_loss"])
            cost_std  = np.asarray(data_list[i]["data_0"][val]["image_pred_loss_std"])
            iteration = np.arange(cost_mean.shape[0])

            lb_std = cost_mean-cost_std
            ub_std = cost_mean+cost_std

            colors = color_set[i]
            ax4.plot(iteration,cost_mean,linestyle = '--', color =colors)
            ax4.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
            ax4.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
            ax4.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)



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




            # cost_mean = np.asarray(data_list[i]["data_0"][val]["pred_center_x"])
            # cost_std  = np.asarray(data_list[i]["data_0"][val]["pred_center_x_std"])
            # iteration = np.arange(cost_mean.shape[0])

            # lb_std = cost_mean-cost_std
            # ub_std = cost_mean+cost_std

            # colors = color_set[i]
            # ax7.plot(iteration,cost_mean,linestyle = '--', color =colors)
            # ax7.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
            # ax7.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
            # ax7.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)



            # cost_mean = np.asarray(data_list[i]["data_0"][val]["pred_center_y"])
            # cost_std  = np.asarray(data_list[i]["data_0"][val]["pred_center_y_std"])
            # iteration = np.arange(cost_mean.shape[0])

            # lb_std = cost_mean-cost_std
            # ub_std = cost_mean+cost_std

            # colors = color_set[i]
            # ax8.plot(iteration,cost_mean,linestyle = '--', color =colors)
            # ax8.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
            # ax8.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
            # ax8.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)



            # cost_mean = np.asarray(data_list[i]["data_0"][val]["pred_center_z"])
            # cost_std  = np.asarray(data_list[i]["data_0"][val]["pred_center_z_std"])
            # iteration = np.arange(cost_mean.shape[0])

            # lb_std = cost_mean-cost_std
            # ub_std = cost_mean+cost_std

            # colors = color_set[i]
            # ax9.plot(iteration,cost_mean,linestyle = '--', color =colors)
            # ax9.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
            # ax9.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
            # ax9.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)



            # print(f"pred_center_x | mean {data_list[i]['data_0'][val]['pred_center_x'][-1]} | std {data_list[i]['data_0'][val]['pred_center_x_std'][-1]}")
            # print(f"pred_center_y | mean {data_list[i]['data_0'][val]['pred_center_y'][-1]} | std {data_list[i]['data_0'][val]['pred_center_y_std'][-1]}")
            # print(f"pred_center_z | mean {data_list[i]['data_0'][val]['pred_center_z'][-1]} | std {data_list[i]['data_0'][val]['pred_center_z_std'][-1]}")

            # # print(f"center_x | mean {data_list[i]['data_0'][val]['oracle_center_x'][-1]} |")
            # # print(f"center_y | mean {data_list[i]['data_0'][val]['oracle_center_y'][-1]} |")
            # # print(f"center_z | mean {data_list[i]['data_0'][val]['oracle_center_z'][-1]} |")




            # import ipdb;ipdb.set_trace()


        ax1.set_xlabel(" ",fontsize=18,color="black")
        ax1.set_ylabel("Target shape cutting cost",fontsize=18,color="black")
        ax1.tick_params(labelbottom=False, labelsize=22,labelcolor="black")


        ax2.set_xlabel("Time step ",fontsize=18, color="black")
        ax2.set_ylabel("Target shape remaining rate [%]",fontsize=18,color="black")
        ax2.tick_params(labelsize=22,labelcolor="black")
        ax2.set_ylim(0,110)



        ax3.set_xlabel("Time step ",fontsize=18, color="black")
        ax3.set_ylabel("Removal performance",fontsize=18,color="black")
        ax3.tick_params(labelsize=22,labelcolor="black")


        ax4.set_xlabel("Time step ",fontsize=18, color="black")
        ax4.set_ylabel("Reconstruction loss",fontsize=18,color="black")
        ax4.tick_params(labelsize=22,labelcolor="black")


        ax5.set_xlabel("Time step ",fontsize=18, color="black")
        ax5.set_ylabel("augnet loss",fontsize=18,color="black")
        ax5.tick_params(labelsize=22,labelcolor="black")



        ax6.set_xlabel("Time step ",fontsize=18, color="black")
        ax6.set_ylabel("augnet loss",fontsize=18,color="black")
        ax6.tick_params(labelsize=22,labelcolor="black")


        ax7.tick_params(labelsize=22,labelcolor="black")
        ax8.tick_params(labelsize=22,labelcolor="black")
        ax9.tick_params(labelsize=22,labelcolor="black")

        ax7.set_ylim(0,5)
        ax8.set_ylim(0,5)
        ax9.set_ylim(0,5)
        


        title = []
        for i in range(len(data_list)):
            title.append(data_list[i]["data_path"]["data_0"])
        title = "\n".join(title)


        fig.suptitle(title,
                    fontsize= 8)


        # fig.suptitle(data_dir_1["data_0"]+"\n"+
        #             data_dir_2["data_0"],
        #             fontsize= 5)


        # import ipdb;ipdb.set_trace()
        # data_list =[data_dict_5]
        for i in range(len(data_list)):
                save_name = os.path.normpath(data_list[i]["data_path"]["data_0"]+f"/../reward_comparison_{val}_{save_tag}")
                plt.savefig(save_name+".png")
                plt.savefig(save_name+".pdf")
        
        # import ipdb;ipdb.set_trace()


        plt.show()
        plt.clf()
        plt.close()