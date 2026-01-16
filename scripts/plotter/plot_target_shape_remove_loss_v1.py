


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


    # root_dir = "/home/haxhi/workspace/"
    root_dir   = f"/home/haxhi/workspace/denoising-diffusion-pytorch/logs/Image_diffusion_2D/diffusion_plans/PT18_T1000_D64_test_4/"
    save_tag   = "0"

    data_dir_5 ={
        "data_0":os.path.join(root_dir,"epsilon_greedy_00"),
        }
    data_dir_4 ={
        "data_0":os.path.join(root_dir,"epsilon_greedy_001"),
        }

    data_dir_3 ={
        "data_0":os.path.join(root_dir,"epsilon_greedy_01"),
        }

    data_dir_2 ={
        "data_0":os.path.join(root_dir,"epsilon_greedy_05"),
        }

    data_dir_1 ={
        "data_0":os.path.join(root_dir,"random"),
        }


    data_dict_1 = get_merged_dict(data_dir=data_dir_1,file_name="post_processed_data.json")
    data_dict_2 = get_merged_dict(data_dir=data_dir_2,file_name="post_processed_data.json")
    data_dict_3 = get_merged_dict(data_dir=data_dir_3,file_name="post_processed_data.json")
    data_dict_4 = get_merged_dict(data_dir=data_dir_4,file_name="post_processed_data.json")
    data_dict_5 = get_merged_dict(data_dir=data_dir_5,file_name="post_processed_data.json")


    data_list =[data_dict_1,data_dict_2,data_dict_3,data_dict_4,data_dict_5]

    import ipdb;ipdb.set_trace()




    fig = plt.figure(figsize=(20, 10))


    ax1     =    fig.add_subplot(2,2,1)
    ax2     =    fig.add_subplot(2,2,2)
    # ax3     =    fig.add_subplot(2,2,3)

    width = 0.3
    shift = width/2
    color_set       = ["#66cdaa","#9370db","#f4a460", "#db7093",'#808080']




    heights = [data_dict_1["data_0"]["reward_mean"], data_dict_2["data_0"]["reward_mean"], data_dict_3["data_0"]["reward_mean"],data_dict_4["data_0"]["reward_mean"],data_dict_5["data_0"]["reward_mean"]]
    std     = [data_dict_1["data_0"]["reward_std"] , data_dict_2["data_0"]["reward_std"] , data_dict_3["data_0"]["reward_std"] ,data_dict_4["data_0"]["reward_std"] ,data_dict_5["data_0"]["reward_std"] ]
    bars    = np.arange(len(heights))
    ax1.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = color_set, capsize=10)



    for i in range(len(data_list)):


        cost_mean = np.asarray(data_list[i]["data_0"]["removal_trans"])
        cost_std  = np.asarray(data_list[i]["data_0"]["removal_trans_std"])
        iteration = np.arange(cost_mean.shape[0])

        lb_std = cost_mean-cost_std
        ub_std = cost_mean+cost_std

        colors = color_set[i]
        ax2.plot(iteration,cost_mean,linestyle = '--', color =colors)
        ax2.plot(iteration,ub_std,color =colors,linewidth=0.8,alpha=0.3)
        ax2.plot(iteration,lb_std,color =colors,linewidth=0.8,alpha=0.3)
        ax2.fill_between(iteration, ub_std, lb_std,alpha=0.5,color =colors)




    ax1.set_xlabel(" ",fontsize=18,color="black")
    ax1.set_ylabel("Target shape cutting cost",fontsize=18,color="black")
    ax1.tick_params(labelbottom=False, labelsize=22,labelcolor="black")


    ax2.set_xlabel("Time step ",fontsize=18, color="black")
    ax2.set_ylabel("Target shape remaining rate [%]",fontsize=18,color="black")
    ax2.tick_params(labelsize=22,labelcolor="black")


    # fig.suptitle(data_dir_1["data_0"]+"\n"+
    #             data_dir_2["data_0"]+"\n"+
    #             data_dir_3["data_0"]+"\n"+
    #             data_dir_4["data_0"],
    #             fontsize= 5)


    for i in range(len(data_list)):
        # for key,val in enumerate(data_list[i]):
            save_name = os.path.normpath(data_list[i]["data_path"]["data_0"]+f"/reward_comparison_{save_tag}")
            plt.savefig(save_name+".pdf")




    plt.show()