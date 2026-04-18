


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
    return data_dict












if __name__ == '__main__':


    # root_dir = "/home/haxhi/workspace/"
    root_dir   = f"./results/voxel_image_64"
    save_tag   = "mse_loss"

    data_dir_1 ={
        "data_0":os.path.join(root_dir,"eval_13_soft_v2"),
        }

    data_dir_2 ={
        "data_0":os.path.join(root_dir,"eval_13_mid_v2"),
        }

    data_dir_3 ={
        "data_0":os.path.join(root_dir,"eval_13_hard_v2"),
        }

    data_dir_4 ={
        "data_0":os.path.join(root_dir,"eval_13_composite_v2_2"),
        }

    data_dir_5 ={
        "data_0":os.path.join(root_dir,"eval_13_composite_v2_1"),
        }



    data_dir_6 ={
        "data_0":os.path.join(root_dir,"eval_10_hard_v2"),
        }

    data_dir_7 ={
        "data_0":os.path.join(root_dir,"eval_6_hard_v2"),
        }

    data_dict_1 = get_merged_dict(data_dir=data_dir_1,file_name="post_processed_data.json")
    data_dict_2 = get_merged_dict(data_dir=data_dir_2,file_name="post_processed_data.json")
    data_dict_3 = get_merged_dict(data_dir=data_dir_3,file_name="post_processed_data.json")
    data_dict_4 = get_merged_dict(data_dir=data_dir_4,file_name="post_processed_data.json")
    data_dict_5 = get_merged_dict(data_dir=data_dir_5,file_name="post_processed_data.json")
    data_dict_6 = get_merged_dict(data_dir=data_dir_6,file_name="post_processed_data.json")
    data_dict_7 = get_merged_dict(data_dir=data_dir_7,file_name="post_processed_data.json")







    fig = plt.figure(figsize=(20, 20))

    width = 0.3
    shift = width/2
    color_set       = ["#66cdaa","#9370db","#f4a460"]


    ax1     =    fig.add_subplot(2,2,1)
    ax2     =    fig.add_subplot(2,2,2)
    ax3     =    fig.add_subplot(2,2,3)


    # df = pd.DataFrame({
    #     'data_0': data_dict_1["data_0"]["image_loss"],
    #     'data_1': data_dict_2["data_0"]["image_loss"],
    #     'data_2': data_dict_3["data_0"]["image_loss"]
    # })
    # df_melt = pd.melt(df)

    # sns.violinplot(x='variable', y='value', data=df_melt, ax=ax1)



    heights = [data_dict_1["data_0"]["image_loss_mean"], data_dict_2["data_0"]["image_loss_mean"], data_dict_3["data_0"]["image_loss_mean"]]
    std     = [data_dict_1["data_0"]["image_loss_std"] , data_dict_2["data_0"]["image_loss_std"] , data_dict_3["data_0"]["image_loss_std"]]
    bars    = np.arange(len(heights))
    ax1.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = color_set, capsize=10)



    heights = [data_dict_4["data_0"]["image_loss_mean"], data_dict_5["data_0"]["image_loss_mean"]]
    std     = [data_dict_4["data_0"]["image_loss_std"] , data_dict_5["data_0"]["image_loss_std"]]
    bars    = np.arange(len(heights))
    ax2.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = ["steelblue","tan"], capsize=10)

    # import ipdb;ipdb.set_trace()


    heights = [data_dict_7["data_0"]["image_loss_mean"], data_dict_6["data_0"]["image_loss_mean"], data_dict_3["data_0"]["image_loss_mean"]]
    std     = [data_dict_7["data_0"]["image_loss_std"] , data_dict_6["data_0"]["image_loss_std"] , data_dict_3["data_0"]["image_loss_std"]]
    bars    = np.arange(len(heights))
    ax3.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = ["gray","olive", "mediumvioletred"], capsize=10)


    ax1.set_xlabel(" ",fontsize=20,color="black")
    ax1.set_ylabel("MSE",fontsize=20,color="black")
    ax1.tick_params(labelbottom=False, labelsize=22,labelcolor="black")


    ax2.set_xlabel(" ",fontsize=20,color="black")
    ax2.set_ylabel("MSE",fontsize=20,color="black")
    ax2.tick_params(labelbottom=False, labelsize=22,labelcolor="black")


    ax3.set_xlabel(" ",fontsize=20,color="black")
    ax3.set_ylabel("MSE",fontsize=20,color="black")
    ax3.tick_params(labelbottom=False, labelsize=22,labelcolor="black")



    fig.suptitle(data_dir_1["data_0"]+"\n"+
                data_dir_2["data_0"]+"\n"+
                data_dir_3["data_0"]+"\n"+
                data_dir_4["data_0"]+"\n"+
                data_dir_5["data_0"]+"\n"+
                data_dir_6["data_0"]+"\n"+
                data_dir_7["data_0"]+"\n",fontsize= 5)


    for key,val in enumerate(data_dir_1):
        save_name = os.path.normpath(data_dir_1[val]+f"/images_loss_comparison_{save_tag}")
        plt.savefig(save_name+".png")
        plt.savefig(save_name+".pdf")

    for key,val in enumerate(data_dir_2):
        save_name = os.path.normpath(data_dir_2[val]+f"/images_loss_comparison_{save_tag}")
        plt.savefig(save_name+".png")
        plt.savefig(save_name+".pdf")


    for key,val in enumerate(data_dir_3):
        save_name = os.path.normpath(data_dir_3[val]+f"/images_loss_comparison_{save_tag}")
        plt.savefig(save_name+".png")
        plt.savefig(save_name+".pdf")


    plt.show()