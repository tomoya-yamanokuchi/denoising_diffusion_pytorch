


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


    # root_dir = "/home/haxhi/workspace/"
    # root_dir   = f"/home/haxhi/workspace/denoising-diffusion-pytorch/logs/Image_diffusion_2D/diffusion_plans/PT18_T1000_D64_test_6/"
    # root_dir   = f"/home/haxhi/workspace/denoising-diffusion-pytorch/logs/Image_diffusion_2D/diffusion_plans/PT5_T1000_D64_test_7_fix_start/"
    # root_dir   = f"/home/haxhi/workspace/denoising-diffusion-pytorch/logs/Image_diffusion_2D/diffusion_plans/PT10_T1000_D64_test_7_fix_start/"
    # root_dir   = f"/home/haxhi/workspace/denoising-diffusion-pytorch/logs/Image_diffusion_2D/diffusion_plans/PT18_T1000_D64_test_7_fix_start/"
    # root_dir   = f"/home/haxhi/workspace/denoising-diffusion-pytorch/logs/Image_diffusion_2D/diffusion_plans/PT18_T1000_D64_test_8_fix_start/"
    # root_dir   = f"/home/haxhi/workspace/denoising-diffusion-pytorch/logs/Image_diffusion_2D/diffusion_plans/PT10_T1000_D64_test_8_fix_start/"
    # root_dir   = f"/home/haxhi/workspace/denoising-diffusion-pytorch/logs/Image_diffusion_2D/diffusion_plans/PT5_T1000_D64_test_8_fix_start/"
    # root_dir   = f"/home/haxhi/workspace/denoising-diffusion-pytorch/logs/Image_diffusion_2D/diffusion_plans/PT2_T1000_D64_test_8_fix_start/"
    # root_dir1   = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/PT18_T1000_D64_test_v0_fix_start/"
    # root_dir    = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/PT18_T1000_D64_test_v0_fix_start/"
    # root_dir2   = f"/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/PT478_T1000_D64_test_v1_fix_start/"

    # root_dir    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT20000_T1000_D64_test_v1_fix_start"

    # root_dir2_6    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT60000_T1000_D64_test_v3_fix_start"
    # root_dir2_0    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT60000_T1000_D64_test_v2_fix_start"
    # root_dir2_1    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT25000_T1000_D64_test_v2_fix_start"
    # root_dir2_2    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT1000_T1000_D64_test_v2_fix_start"
    # root_dir2_3    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v2/PT500_T1000_D64_test_v2_fix_start"

    # # root_dir3    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT25000_T1000_D64_test_v1_fix_start"
    # root_dir4    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT1000_T1000_D64_test_v1_fix_start"
    # root_dir5    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v2/PT500_T1000_D64_test_v1_fix_start"
    # root_dir6    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v2/PT100_T1000_D64_test_v1_fix_start"


    # root_dir3_0    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT60000_T1000_D64_test_v4_fix_start"
    # root_dir3_1    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT25000_T1000_D64_test_v4_fix_start"
    # root_dir3_2    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT500_T1000_D64_test_v4_fix_start"
    # root_dir4_0    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_1/PT18_T1000_D64_test_v4_fix_start"
    # root_dir4_2    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_1/PT1_T1000_D64_test_v4_fix_start"
    # root_dir4_1    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_1/PT5_T1000_D64_test_v4_fix_start"



    # root_dir5_0    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v1/PT280000_T1000_D64_test_v1_fix_start"
    # root_dir5_1    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v1/PT20000_T1000_D64_test_v1_fix_start"
    # root_dir5_2    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v1/PT160000_T1000_D64_test_v1_fix_start"
    # root_dir5_3    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT1000_T1000_D64_test_v1_fix_start"


    # root_dir6_0    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT60000_T1000_D64_test_v5_fix_start"
    # root_dir6_1    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT500_T1000_D64_test_v5_fix_start"
    # root_dir6_2    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_2_12900k_v1/PT200_T1000_D64_test_v5_fix_start"

    # root_dir7_0    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v3/PT600000_T1000_D64_test_v1_fix_start"

    # root_dir8_0    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v3/PT600000_T1000_D64_test_v8_fix_start"
    # root_dir8_1    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT1000_T1000_D64_test_v8_fix_start"



    # root_dir9_0    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT600000_T1000_D64_test_v6_fix_start_multi_step"
    # root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT600000_T1000_D64_test_v6_fix_start_multi_step"
    root_dir9_0     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step_"
    # root_dir9_1     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step__no_separate"
    # root_dir9_2     = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4_12900k_v1/PT80000_T1000_D64_test_v5_fix_start_multi_step__2_target"


    # root_dir5_2    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v1/PT160000_T1000_D64_test_v1_fix_start"
    # root_dir5_3    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_3_12900k_v2/PT1000_T1000_D64_test_v1_fix_start"


    # save_tag    = "0"
    save_tag    = "0_partial"




    # data_dir_2 ={
    #     "data_0":os.path.join(root_dir,"epsilon_greedy_01"),
    #     }

    # data_dir_1 ={
    #     "data_0":os.path.join(root_dir,"random"),
    #     }




    # data_dir_6 ={
    #     "data_0":os.path.join(root_dir3_0,"oracle_obs"),
    #     }
    # data_dir_5 ={
    #     # "data_0":os.path.join(root_dir,"epsilon_greedy_001"),
    #     "data_0":os.path.join(root_dir3_0,"epsilon_greedy_00"),
    #     # "data_0":os.path.join(root_dir2,"epsilon_greedy_00"),
    #     }

    # data_dir_4 ={
    #     # "data_0":os.path.join(root_dir,"epsilon_greedy_01"),
    #     # "data_0":os.path.join(root_dir,"epsilon_greedy_05"),
    #     "data_0":os.path.join(root_dir3_1,"epsilon_greedy_00"),
    #     }

    # data_dir_3 ={
    #     # "data_0":os.path.join(root_dir,"epsilon_greedy_01"),
    #     "data_0":os.path.join(root_dir3_2,"epsilon_greedy_00"),
    #     }


    # data_dir_2 ={
    #     # "data_0":os.path.join(root_dir,"epsilon_greedy_01"),
    #     "data_0":os.path.join(root_dir3_1,"no_cond"),
    #     }

    # data_dir_1 ={
    #     "data_0":os.path.join(root_dir3_0,"random"),
    #     }



    # data_dir_6 ={
    #     "data_0":os.path.join(root_dir6_0,"oracle_obs"),
    #     }
    # data_dir_5 ={
    #     "data_0":os.path.join(root_dir6_0,"epsilon_greedy_00"),
    #     }

    # data_dir_4 ={
    #     "data_0":os.path.join(root_dir6_1,"epsilon_greedy_00"),
    #     }

    # data_dir_3 ={
    #     "data_0":os.path.join(root_dir6_2,"epsilon_greedy_00"),
    #     }

    # data_dir_2 ={
    #     "data_0":os.path.join(root_dir6_0,"no_cond"),
    #     }

    # data_dir_1 ={
    #     "data_0":os.path.join(root_dir6_0,"random"),
    #     }



    # data_dir_6 ={
    #     "data_0":os.path.join(root_dir8_0,"oracle_obs"),
    #     }
    # data_dir_5 ={
    #     "data_0":os.path.join(root_dir8_0,"epsilon_greedy_00"),
    #     }

    # data_dir_4 ={
    #     "data_0":os.path.join(root_dir8_1,"epsilon_greedy_00"),
    #     }

    # data_dir_3 ={
    #     "data_0":os.path.join(root_dir8_1,"epsilon_greedy_00"),
    #     }

    # data_dir_2 ={
    #     "data_0":os.path.join(root_dir8_0,"no_cond"),
    #     }

    # data_dir_1 ={
    #     "data_0":os.path.join(root_dir8_0,"random"),
    #     }






    data_dir_6 ={
        "data_0":os.path.join(root_dir9_0,"oracle_obs"),
        }
    data_dir_5 ={
        "data_0":os.path.join(root_dir9_0,"epsilon_greedy_00"),
        }

    data_dir_4 ={
        "data_0":os.path.join(root_dir9_0,"epsilon_greedy_00"),
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


    data_dict_1 = get_merged_dict(data_dir=data_dir_1,file_name="post_processed_data.json")
    data_dict_2 = get_merged_dict(data_dir=data_dir_2,file_name="post_processed_data.json")
    data_dict_3 = get_merged_dict(data_dir=data_dir_3,file_name="post_processed_data.json")
    data_dict_4 = get_merged_dict(data_dir=data_dir_4,file_name="post_processed_data.json")
    data_dict_5 = get_merged_dict(data_dir=data_dir_5,file_name="post_processed_data.json")
    data_dict_6 = get_merged_dict(data_dir=data_dir_6,file_name="post_processed_data.json")



    # data_list =[data_dict_1,data_dict_2]
    # data_list =[data_dict_1,data_dict_2,data_dict_3,data_dict_4,data_dict_5]
    data_list =[data_dict_1,data_dict_2,data_dict_3,data_dict_4,data_dict_5,data_dict_6]
    # data_list =[data_dict_5]


    # import ipdb;ipdb.set_trace()



    for idx,val in enumerate(data_dict_1["data_0"].keys()):



        fig = plt.figure(figsize=(20, 10))

        ax1     =    fig.add_subplot(2,2,1)
        ax2     =    fig.add_subplot(2,2,2)
        ax3     =    fig.add_subplot(2,2,3)

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