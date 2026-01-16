


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter

import matplotlib.ticker as ticker

from denoising_diffusion_pytorch.utils.os_utils import load_json







def get_merged_dict(data_dir,file_name):
    data_dict = {}
    for key,data in data_dir.items():
        data_path = os.path.join(data,file_name)
        data = load_json(data_path)
        data_dict[key]=data
    data_dict["data_path"] =  data_dir
    return data_dict



# class FixedOrderFormatter(ScalarFormatter):
#     def __init__(self, order_of_mag=0, useOffset=True, useMathText=True):
#         self._order_of_mag = order_of_mag
#         ScalarFormatter.__init__(self, useOffset=useOffset,
#                                  useMathText=useMathText)
#     def _set_orderOfMagnitude(self, range):
#         self.orderOfMagnitude = self._order_of_mag






class FixedOrderFormatter(ScalarFormatter):
    def __init__(self, order_of_mag=0, useMathText=True, format_str="%.1f"):
        super().__init__(useMathText=useMathText)
        self._order_of_mag = order_of_mag
        self.format_str = format_str
        self.set_powerlimits((0, 0))  # 常に指数表記

    def _set_order_of_magnitude(self):
        """Fix the order of magnitude (e.g., -3 for ×10⁻³)."""
        self.orderOfMagnitude = self._order_of_mag

    def __call__(self, x, pos=None):
        return self.format_str % (x * 10**(-self._order_of_mag))


if __name__ == '__main__':





    # import numpy as np
    # import matplotlib.pyplot as plt
    # # pの範囲を定義（ベルヌーイ分布の範囲）
    # p = np.linspace(0, 1, 500)

    # # μ = p, var = p(1 - p), std = sqrt(var)
    # mu = p
    # var = p * (1 - p)
    # std = np.sqrt(var)

    # # μ + var, μ + std
    # mu_plus_var = mu + var
    # mu_plus_std = mu + std

    # # グラフ描画
    # plt.figure(figsize=(10, 6))
    # plt.plot(p, mu_plus_var, label=r'$\mu + \mathrm{var}$', color='blue')
    # plt.plot(p, mu_plus_std, label=r'$\mu + \mathrm{std}$', color='green')
    # plt.axhline(0.3, color='red', linestyle='--', label='y = 0.3')
    # plt.xlabel('Bernoulli parameter $p$')
    # plt.ylabel('Value')
    # plt.title('Comparison of $\mu + \mathrm{var}$ and $\mu + \mathrm{std}$ in Bernoulli Distribution')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # exit()
    plt.style.use('ggplot')

    # is_save = False
    is_save = True

    # save_tag   = "v2_19_partial_obs_a123456789_clip_ucb_raw_0.5_13_4_cond_comp" # latest tag


    save_tag   = "v2_19_partial_obs_a123456_clip_ucb_raw_0.5_14_1_cond_comp" # latest tag



    #############
    ## simple config
    #############
    ##################
    ## for journal result
    ##################
    # root_dir1   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456789_clip_ucb_raw_0.5_v12_1"
    # root_dir1_  = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456789_clip_ucb_raw_0.5_v12_1"
    # root_dir2   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT100000_B32_T8_partial_obs_vaeac_a123456789_clip_ucb_raw_0.5_v13_2_tmp"
    # root_dir3   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT100000_B32_T8_partial_obs_diffusion_1D_a123456789_clip_ucb_raw_0.5_v13_2"



    #############
    ## real config
    #############

    #################
    # for journal result
    #################
    # root_dir1   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456_clip_ucb_raw_0.5_v13_1"
    root_dir1_  = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456_clip_ucb_raw_0.5_v14_1"
    # root_dir2   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_vaeac_a123456_clip_ucb_raw_0.5_v13_2"


    ##########################################
    ## simple model
    ##########################################
    # data_dir_1 ={
    #     "data_0":os.path.join(root_dir1,"random"),
    #     }

    data_dir_2 ={
        "data_0":os.path.join(root_dir1_,"no_cond"),
        }


    # data_dir_3 ={
    #     "data_0":os.path.join(root_dir1,"oracle_obs"),
    #     }


    # data_dir_4 ={
    #     "data_0":os.path.join(root_dir2,"epsilon_greedy_00"),
    #     }


    data_dir_5 ={
        # "data_0":os.path.join(root_dir1,"epsilon_greedy_00"),
        "data_0":os.path.join(root_dir1_,"epsilon_greedy_00"),
        }



    data_dict_1 = get_merged_dict(data_dir=data_dir_2,file_name="post_processed_data_v5.json")
    data_dict_5 = get_merged_dict(data_dir=data_dir_5,file_name="post_processed_data_v5.json")

    # keys_to_delete = {'Object_2','Object_3','Object_4','Object_5','Object_7','Object_9'} # simple-model
    keys_to_delete = {'Object_1','Object_3','Object_5'} # complex-model

    for key in keys_to_delete:
        data_dict_1["data_0"].pop(key, None)
        data_dict_5["data_0"].pop(key, None)


    data_list =[data_dict_1,data_dict_5]
    # color_set = ['#5499c7', '#f5b041', '#bb8fce'] #blue, orange, purple #for simple model
    # maker_color = ['#447ca1', '#c69435', '#9771aa']

    color_set = ['#5499c7', '#bb8fce', '#f5b041'] #blue, purple, orange #for complex model
    maker_color = ['#447ca1', '#9771aa', '#c69435']

    # color_set = ['#5499c7', '#bb8fce','#f5b041',]
    # color_set = ['#1f77b4', '#ff7f0e', '#9467bd']
    # color_set = ['#1f77b4', '#f5b041', '#9b59b6']
    # color_set = ['#FF4B00', '#005AFF', '#03AF7A'] #blue, orange, purple
    # color_set = ['#073467', '#ffbc36', '#ea6d6d'] #blue, orange, purple
    

    title = []

    # fig = plt.figure(figsize=(32, 16))
    fig = plt.figure(figsize=(22, 16))
    ax1     =    fig.add_subplot(3,2,1)
    ax2     =    fig.add_subplot(3,2,2)
    ax3     =    fig.add_subplot(3,2,3)
    ax4     =    fig.add_subplot(3,2,4)
    ax5     =    fig.add_subplot(3,2,5)
    ax6     =    fig.add_subplot(3,2,6)

    for i in range(len(data_list)):
        for idx,val in enumerate(data_dict_1["data_0"].keys()):
            if "epsilon_greedy" in data_list[i]["data_path"]["data_0"]:
                alpa = 0.25
                marker = "o"
                marker_size = 13+2
            else:
                alpa = 0.3
                # marker = "s"
                marker = "X"
                marker_size = 16+2

            cost_mean = np.asarray(data_list[i]["data_0"][val]["image_pred_loss"])
            cost_std  = np.asarray(data_list[i]["data_0"][val]["image_pred_loss_std"])
            iteration = np.arange(cost_mean.shape[0])

            lb_std = cost_mean-cost_std
            ub_std = cost_mean+cost_std

            colors = color_set[idx]
            ax4.plot(iteration,cost_mean, marker, linestyle = '--', color =colors, ms=marker_size,
                        markerfacecolor=maker_color[idx], markeredgecolor=[221/255, 221/255, 221/255],
                        markeredgewidth=1.5)
            ax4.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=alpa)
            ax4.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=alpa)
            ax4.fill_between(iteration, ub_std, lb_std,alpha=alpa+0.2,color =colors)
            ax4.set_xticks( np.arange(start=0, stop=8, step=2))

            title.append(val+"_"+Path(data_list[i]["data_path"]["data_0"]).name+"_"+colors)


    ax4.tick_params(labelsize=24,labelcolor="black")
    ax4.yaxis.set_major_formatter(FixedOrderFormatter(-3, useMathText=True, format_str="%.1f"))
    ax4.yaxis.get_offset_text().set_fontsize(16)  # フォントサイズを14に
    title = "\n".join(title)
    fig.suptitle(title,fontsize= 8)


    if is_save is True:
        save_name = os.path.normpath(data_list[0]["data_path"]["data_0"]+f"/../reward_comparison_{save_tag}")
        plt.savefig(save_name+".pdf")


    plt.show()
    plt.clf()
    plt.close()

    # import ipdb;ipdb.set_trace()
    exit()

    # for idx,val in enumerate(data_dict_1["data_0"].keys()):

    #     # fig = plt.figure(figsize=(22, 16))
    #     fig = plt.figure(figsize=(32, 16))

    #     ax1     =    fig.add_subplot(3,2,1)
    #     ax2     =    fig.add_subplot(3,2,2)
    #     ax3     =    fig.add_subplot(3,2,3)
    #     ax4     =    fig.add_subplot(3,2,4)
    #     ax5     =    fig.add_subplot(3,2,5)
    #     ax6     =    fig.add_subplot(3,2,6)


    #     width = 0.3
    #     shift = width/2




    #     # heights = [data_dict_1["data_0"][val]["reward_mean"], data_dict_2["data_0"][val]["reward_mean"],data_dict_3["data_0"][val]["reward_mean"]]
    #     # std     = [data_dict_1["data_0"][val]["reward_std"] , data_dict_2["data_0"][val]["reward_std"] ,data_dict_3["data_0"][val]["reward_std"] ]
    #     heights = [d["data_0"][val]["reward_mean"] for d in data_list]
    #     std     = [d["data_0"][val]["reward_std"]  for d in data_list]
    #     bars    = np.arange(len(heights))
    #     ax1.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = color_set, capsize=10)


    #     for i in range(len(data_list)):

    #         cost_mean = np.asarray(data_list[i]["data_0"][val]["removal_trans"])
    #         cost_std  = np.asarray(data_list[i]["data_0"][val]["removal_trans_std"])
    #         iteration = np.arange(cost_mean.shape[0])

    #         lb_std = cost_mean-cost_std
    #         ub_std = cost_mean+cost_std

    #         colors = color_set[i]
    #         ax2.plot(iteration,cost_mean,linestyle = '--', color =colors)
    #         ax2.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
    #         ax2.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
    #         ax2.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)



    #         cost_mean = np.asarray(data_list[i]["data_0"][val]["removal_performance"])
    #         cost_std  = np.asarray(data_list[i]["data_0"][val]["removal_performance_std"])
    #         iteration = np.arange(cost_mean.shape[0])

    #         lb_std = cost_mean-cost_std
    #         ub_std = cost_mean+cost_std

    #         colors = color_set[i]
    #         ax3.plot(iteration,cost_mean,linestyle = '--', color =colors)
    #         ax3.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
    #         ax3.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
    #         ax3.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)

    #         # import ipdb;ipdb.set_trace()
    #         if "no_cond" in data_list[i]["data_path"]["data_0"] or "epsilon_greedy_00" in data_list[i]["data_path"]["data_0"] :

    #             if "vaeac" in data_list[i]["data_path"]["data_0"] is not True:
    #                 pass
    #             elif "1D" in data_list[i]["data_path"]["data_0"] is not True:
    #                 pass
    #             else:
    #                 cost_mean = np.asarray(data_list[i]["data_0"][val]["image_pred_loss"])
    #                 cost_std  = np.asarray(data_list[i]["data_0"][val]["image_pred_loss_std"])
    #                 # cost_mean = np.asarray(data_list[i]["data_0"][val]["image_pred_loss_mode"])
    #                 # cost_std  = np.asarray(data_list[i]["data_0"][val]["image_pred_loss_mode_std"])
    #                 iteration = np.arange(cost_mean.shape[0])

    #                 lb_std = cost_mean-cost_std
    #                 ub_std = cost_mean+cost_std

    #                 colors = color_set[i]
    #                 ax4.plot(iteration,cost_mean,linestyle = '--', color =colors)
    #                 ax4.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
    #                 ax4.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
    #                 ax4.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)
    #                 ax4.set_xticks( np.arange(start=0, stop=8, step=2))


    #         cost_mean = np.asarray(data_list[i]["data_0"][val]["augnet_loss"])
    #         cost_std  = np.asarray(data_list[i]["data_0"][val]["augnet_loss_std"])
    #         iteration = np.arange(cost_mean.shape[0])

    #         lb_std = cost_mean-cost_std
    #         ub_std = cost_mean+cost_std

    #         colors = color_set[i]
    #         ax5.plot(iteration,cost_mean,linestyle = '--', color =colors)
    #         ax5.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
    #         ax5.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
    #         ax5.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)



    #         if "no_cond" in data_list[i]["data_path"]["data_0"] or "epsilon_greedy_00" in data_list[i]["data_path"]["data_0"] :

    #             if "vaeac" in data_list[i]["data_path"]["data_0"] is not True:
    #                 pass
    #             elif "1D" in data_list[i]["data_path"]["data_0"] is not True:
    #                 pass
    #             else:
    #                 cost_mean = np.asarray(data_list[i]["data_0"][val]["augnet_loss"])
    #                 cost_std  = np.asarray(data_list[i]["data_0"][val]["augnet_loss_std"])
    #                 iteration = np.arange(cost_mean.shape[0])

    #                 lb_std = cost_mean-cost_std
    #                 ub_std = cost_mean+cost_std

    #                 colors = color_set[i]
    #                 ax6.plot(iteration,cost_mean,linestyle = '--', color =colors)
    #                 ax6.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
    #                 ax6.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
    #                 ax6.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)


    #     # ax1.set_xlabel(" ",fontsize=18,color="black")
    #     # ax1.set_ylabel("Target shape cutting cost",fontsize=18,color="black")
    #     # ax1.tick_params(labelbottom=False, labelsize=22,labelcolor="black")


    #     # ax2.set_xlabel("Time step ",fontsize=18, color="black")
    #     # ax2.set_ylabel("Target shape remaining rate [%]",fontsize=18,color="black")
    #     # ax2.tick_params(labelsize=22,labelcolor="black")
    #     # ax2.set_ylim(0,110)



    #     # ax3.set_xlabel("Time step ",fontsize=18, color="black")
    #     # ax3.set_ylabel("Removal performance",fontsize=18,color="black")
    #     # ax3.tick_params(labelsize=24,labelcolor="black")


    #     # ax4.set_xlabel("Time step ",fontsize=18, color="black")
    #     # ax4.set_ylabel("Reconstruction loss",fontsize=18,color="black")
    #     # ax4.tick_params(labelsize=24,labelcolor="black")
    #     ax4.tick_params(labelsize=24,labelcolor="black")
    #     # ax4.yaxis.set_major_formatter(FixedOrderFormatter(-3, useMathText=True, format_str="%.1f"))
    #     # ax4.yaxis.get_offset_text().set_fontsize(14)  # フォントサイズを14に


    #     # ax5.set_xlabel("Time step ",fontsize=18, color="black")
    #     # # ax5.set_ylabel("augnet loss",fontsize=18,color="black")
    #     # ax5.set_ylabel("SSIM loss",fontsize=18,color="black")
    #     # ax5.tick_params(labelsize=22,labelcolor="black")



    #     # ax6.set_xlabel("Time step ",fontsize=18, color="black")
    #     # ax6.set_ylabel("augnet loss",fontsize=18,color="black")
    #     ax6.tick_params(labelsize=24,labelcolor="black")


    #     title = []
    #     for i in range(len(data_list)):
    #         title.append(data_list[i]["data_path"]["data_0"])
    #     title = "\n".join(title)


    #     fig.suptitle(title,
    #                 fontsize= 8)

    #     if is_save is True:
    #         for i in range(len(data_list)):
    #                 save_name = os.path.normpath(data_list[i]["data_path"]["data_0"]+f"/../reward_comparison_{val}_{save_tag}")
    #                 plt.savefig(save_name+".pdf")

    #     plt.show()
    #     plt.clf()
    #     plt.close()


    #     # import ipdb;ipdb.set_trace()

    #     if is_save is True:
    #         for i in range(len(data_list)):
    #             save_name = os.path.normpath(data_list[i]["data_path"]["data_0"]+f"/../task_performance_comparison_{val}_{save_tag}")
    #             with open(f"{save_name}.txt", "w") as file:

    #                 print(f"{title}",file = file)
    #                 print(f"Random          & {data_dict_1['data_0'][val]['success_rate']} & {np.round(data_dict_1['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_1['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_1['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_1['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
    #                 print(f"Vaeac           & {data_dict_5['data_0'][val]['success_rate']} & {np.round(data_dict_5['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_5['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_5['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_5['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
    #                 print(f"Proposed-Nocond & {data_dict_2['data_0'][val]['success_rate']} & {np.round(data_dict_2['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_2['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_2['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_2['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
    #                 print(f"Proposed        & {data_dict_4['data_0'][val]['success_rate']} & {np.round(data_dict_4['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_4['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_4['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_4['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
    #                 print(f"Proposed-GT     & {data_dict_3['data_0'][val]['success_rate']} & {np.round(data_dict_3['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_3['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_3['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_3['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
    #                 print(f"================================================================================",file=file)
    #                 print(f"Random          & {np.round(data_dict_1['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_1['data_0'][val]['reward_std'],2)} & {np.round(data_dict_1['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_1['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_1['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_1['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
    #                 print(f"Vaeac           & {np.round(data_dict_4['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_4['data_0'][val]['reward_std'],2)} & {np.round(data_dict_4['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_4['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_4['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_4['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
    #                 print(f"3D-Diffusion    & {np.round(data_dict_9['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_9['data_0'][val]['reward_std'],2)} & {np.round(data_dict_9['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_9['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_9['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_9['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
    #                 print(f"Proposed-Nocond & {np.round(data_dict_2['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_2['data_0'][val]['reward_std'],2)} & {np.round(data_dict_2['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_2['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_2['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_2['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
    #                 print(f"Proposed        & {np.round(data_dict_5['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_5['data_0'][val]['reward_std'],2)} & {np.round(data_dict_5['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_5['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_5['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_5['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
    #                 print(f"Proposed-GT     & {np.round(data_dict_3['data_0'][val]['reward_mean'],2)} $\pm${np.round(data_dict_3['data_0'][val]['reward_std'],2)} & {np.round(data_dict_3['data_0'][val]['removal_trans'][-1],2)} $\pm${np.round(data_dict_3['data_0'][val]['removal_trans_std'][-1],2)} & {np.round(data_dict_3['data_0'][val]['removal_performance'][-1],2)} $\pm${np.round(data_dict_3['data_0'][val]['removal_performance_std'][-1],2)}",file=file)
