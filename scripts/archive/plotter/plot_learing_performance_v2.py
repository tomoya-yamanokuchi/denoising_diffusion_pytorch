


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




    # root_dir1   = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/PT1_T1000_D64_test_v0_fix_start/"
    # root_dir2   = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/PT2_T1000_D64_test_v0_fix_start/"
    # root_dir3   = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/PT8_T1000_D64_test_v0_fix_start/"
    # root_dir4   = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/PT478_T1000_D64_test_v1_fix_start/"


    save_tag   = "1"



    root_dir4    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT600000_T1000_D64_test_v5_fix_start_single_step"
    root_dir3    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT100000_T1000_D64_test_v5_fix_start_single_step"
    root_dir2    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT500_T1000_D64_test_v5_fix_start_single_step"
    root_dir1    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT100_T1000_D64_test_v5_fix_start_single_step"





    data_dir_4 ={
        "data_0":os.path.join(root_dir4,"epsilon_greedy_00"),
        }


    data_dir_3 ={
        "data_0":os.path.join(root_dir3,"epsilon_greedy_00"),
        }


    data_dir_2 ={
        "data_0":os.path.join(root_dir2,"epsilon_greedy_00"),
        }

    data_dir_1 ={
        "data_0":os.path.join(root_dir1,"epsilon_greedy_00"),
        }




    data_dict_1 = get_merged_dict(data_dir=data_dir_1,file_name="post_processed_data.json")
    data_dict_2 = get_merged_dict(data_dir=data_dir_2,file_name="post_processed_data.json")
    data_dict_3 = get_merged_dict(data_dir=data_dir_3,file_name="post_processed_data.json")
    data_dict_4 = get_merged_dict(data_dir=data_dir_4,file_name="post_processed_data.json")




    root_dir4_1    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT600000_T1000_D64_test_v5_fix_start_multi_step"
    root_dir3_1    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT100000_T1000_D64_test_v5_fix_start_multi_step"
    root_dir2_1    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT500_T1000_D64_test_v5_fix_start_multi_step"
    root_dir1_1    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT100_T1000_D64_test_v5_fix_start_multi_step"



    data_dir_4 ={
        "data_0":os.path.join(root_dir4_1,"epsilon_greedy_00"),
        }


    data_dir_3 ={
        "data_0":os.path.join(root_dir3_1,"epsilon_greedy_00"),
        }


    data_dir_2 ={
        "data_0":os.path.join(root_dir2_1,"epsilon_greedy_00"),
        }

    data_dir_1 ={
        "data_0":os.path.join(root_dir1_1,"epsilon_greedy_00"),
        }




    data_dict_1_1 = get_merged_dict(data_dir=data_dir_1,file_name="post_processed_data.json")
    data_dict_2_1 = get_merged_dict(data_dir=data_dir_2,file_name="post_processed_data.json")
    data_dict_3_1 = get_merged_dict(data_dir=data_dir_3,file_name="post_processed_data.json")
    data_dict_4_1 = get_merged_dict(data_dir=data_dir_4,file_name="post_processed_data.json")






    # data_list =[data_dict_1,data_dict_2]
    # data_list =[data_dict_1,data_dict_2,data_dict_3]
    data_list =[data_dict_1,data_dict_2,data_dict_3,data_dict_4]





    for idx,val in enumerate(data_dict_1["data_0"].keys()):



        fig = plt.figure(figsize=(20, 10))

        ax1     =    fig.add_subplot(2,2,1)
        ax2     =    fig.add_subplot(2,2,2)
        ax3     =    fig.add_subplot(2,2,3)

        width = 0.3
        shift = width/2
        # color_set       = ["#66cdaa","#9370db","#4682b4"]
        # color_set       = ["#66cdaa","#9370db","#4682b4","#f08080"]
        color_set       = ["#9370db","#9370db","#9370db","#9370db"]



        # heights = [data_dict_1["data_0"][val]["reward_mean"], data_dict_2["data_0"][val]["reward_mean"], data_dict_3["data_0"][val]["reward_mean"]]
        # std     = [data_dict_1["data_0"][val]["reward_std"] , data_dict_2["data_0"][val]["reward_std"] , data_dict_3["data_0"][val]["reward_std"] ]
        heights = [data_dict_1["data_0"][val]["reward_mean"], data_dict_2["data_0"][val]["reward_mean"], data_dict_3["data_0"][val]["reward_mean"], data_dict_4["data_0"][val]["reward_mean"]]
        std     = [data_dict_1["data_0"][val]["reward_std"] , data_dict_2["data_0"][val]["reward_std"] , data_dict_3["data_0"][val]["reward_std"] , data_dict_4["data_0"][val]["reward_std"] ]
        bars    = np.arange(len(heights))-0.15
        ax1.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = color_set, capsize=10,width=-0.3)

        heights = [data_dict_1_1["data_0"][val]["reward_mean"], data_dict_2_1["data_0"][val]["reward_mean"], data_dict_3_1["data_0"][val]["reward_mean"], data_dict_4_1["data_0"][val]["reward_mean"]]
        std     = [data_dict_1_1["data_0"][val]["reward_std"] , data_dict_2_1["data_0"][val]["reward_std"] , data_dict_3_1["data_0"][val]["reward_std"] , data_dict_4_1["data_0"][val]["reward_std"] ]
        bars    = np.arange(len(heights))+0.15
        ax1.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = "#f08080", capsize=10,width=-0.3)





        heights = [data_dict_1["data_0"][val]["removal_trans"][-1],         data_dict_2["data_0"][val]["removal_trans"][-1],       data_dict_3["data_0"][val]["removal_trans"][-1]      ,data_dict_4["data_0"][val]["removal_trans"][-1],   ]
        std     = [data_dict_1["data_0"][val]["removal_trans_std"][-1],     data_dict_2["data_0"][val]["removal_trans_std"][-1],   data_dict_3["data_0"][val]["removal_trans_std"][-1]  ,data_dict_4["data_0"][val]["removal_trans_std"][-1]]
        bars    = np.arange(len(heights))-0.15
        ax2.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = color_set, capsize=10, width=0.3)


        heights = [data_dict_1_1["data_0"][val]["removal_trans"][-1],         data_dict_2_1["data_0"][val]["removal_trans"][-1],       data_dict_3_1["data_0"][val]["removal_trans"][-1]      ,data_dict_4_1["data_0"][val]["removal_trans"][-1],   ]
        std     = [data_dict_1_1["data_0"][val]["removal_trans_std"][-1],     data_dict_2_1["data_0"][val]["removal_trans_std"][-1],   data_dict_3_1["data_0"][val]["removal_trans_std"][-1]  ,data_dict_4_1["data_0"][val]["removal_trans_std"][-1]]
        bars    = np.arange(len(heights))+0.15
        ax2.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = "#f08080" ,capsize=10, width=0.3)





        heights = [data_dict_1["data_0"][val]["removal_performance"][-1],         data_dict_2["data_0"][val]["removal_performance"][-1],       data_dict_3["data_0"][val]["removal_performance"][-1]    , data_dict_4["data_0"][val]["removal_performance"][-1]    ]
        std     = [data_dict_1["data_0"][val]["removal_performance_std"][-1],     data_dict_2["data_0"][val]["removal_performance_std"][-1],   data_dict_3["data_0"][val]["removal_performance_std"][-1], data_dict_4["data_0"][val]["removal_performance_std"][-1]]
        bars    = np.arange(len(heights))-0.15
        ax3.bar(bars, heights, yerr=std, align='center', alpha=1.0, ecolor='black',color = color_set, capsize=10, width=0.3)


        heights = [data_dict_1_1["data_0"][val]["removal_performance"][-1],         data_dict_2_1["data_0"][val]["removal_performance"][-1],       data_dict_3_1["data_0"][val]["removal_performance"][-1]    , data_dict_4_1["data_0"][val]["removal_performance"][-1]    ]
        std     = [data_dict_1_1["data_0"][val]["removal_performance_std"][-1],     data_dict_2_1["data_0"][val]["removal_performance_std"][-1],   data_dict_3_1["data_0"][val]["removal_performance_std"][-1], data_dict_4_1["data_0"][val]["removal_performance_std"][-1]]
        bars    = np.arange(len(heights))+0.15
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





        fig.suptitle(data_dir_1["data_0"]+"\n"+
                    data_dir_2["data_0"],
                    fontsize= 5)


        # import ipdb;ipdb.set_trace()
        for i in range(len(data_list)):
                save_name = os.path.normpath(data_list[i]["data_path"]["data_0"]+f"/../learning_performance_comparison_{val}_{save_tag}")
                plt.savefig(save_name+".pdf")


        plt.show()
        plt.clf()
        plt.close()