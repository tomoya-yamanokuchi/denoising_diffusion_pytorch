
import os
import numpy as np
import json
import cv2

# import torch
# from torch import nn
# import imgsim
from tqdm import tqdm
# from collections import Counter
# import torch.nn.functional as F

from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.os_utils import get_folder_name,pickle_utils,MyEncoder,get_path
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_load_to_numpy,pil_image_save_from_numpy
from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np
from denoising_diffusion_pytorch.utils.pil_utils import numpy_to_pil ,cv2_hsv_mask ,pil_to_cv2,color_range_mask






if __name__ == '__main__':

    #---------------------------------- setup ----------------------------------#

    # tag = "epsilon_greedy_00"
    # tag = "oracle_obs"
    tag = "no_cond"
    save_folders = f"/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_1/dataset_SheetSander_024_eval/PT70000_B32_T8_fix_start_multi_step_both_cs_obs_a17_diffusion_cal_cost_mean_ucb_v1_3/{tag}"
    # save_folders = f"/home/user/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_1/dataset_SheetSander_024_eval/PT70000_B16_T8_fix_start_multi_step_both_cs_obs_a17_vaeac_cal_cost_mean_ucb_v1_3/{tag}"
    
    eval_folders = get_folder_name(save_folders)


    episodes_trj = {}
    pixel_loss_ =[]
    # evaluate folder loop
    for i in range(len(eval_folders)):

        episode_folders = get_folder_name(os.path.join(save_folders,eval_folders[i]))
        episodes_trj[eval_folders[i]]= {}
        
        pixel_loss_r  = []

        for j in range(len(episode_folders)): # episode folder loop

            data_dir  = os.path.join(save_folders, eval_folders[i], episode_folders[j])
            final_step_obs_image  = pil_image_load_to_numpy(data_path=data_dir+"/7_seq_obs_cast_z_axis7_0.png")
            
            
            target_mask_b = np.asarray([0.0,0.0,1.0])
            image_mask_config_b = {"target_mask":target_mask_b,
                    "target_mask_lb":target_mask_b-np.asarray([0.1,0.1,0.1]),
                    "target_mask_ub":target_mask_b+np.asarray([0.1,0.1,0.0])}
            mask_image_blue  = color_range_mask(final_step_obs_image,image_mask_config_b)
            mask_image = mask_image_blue
            target_mask_cutting_cost    = mask_image.mean(2).sum()
            
            pixel_loss_r.append(target_mask_cutting_cost)


        episodes_trj[eval_folders[i]]["observed_target"] = np.asarray(pixel_loss_r).reshape(len(episode_folders),-1)


                # pil_image_save_from_numpy(target_image,f"./{k}_target_image.png")
                # pil_image_save_from_numpy(pred_image,f"./{k}_pred_image.png")


    # NumPy array を list に変換
    episodes_trj_serializable = {
        k: {'observed_target': v['observed_target'].tolist()}
        for k, v in episodes_trj.items()
    }

    # JSON に保存
    with open(save_folders+"/observed_target.json", "w") as f:
        json.dump(episodes_trj_serializable, f, indent=2)

    # post_processed_data =  {}
    # for key,value  in episodes_trj.items():

    #     print(f"load_dirs:{key}")
    #     # import ipdb;ipdb.set_trace()
    #     tmps = value["image_pred_loss"].mean(0).tolist()
    #     print(f"image_loss: {tmps}")
    #     ## save some statics values
    #     non_zero_count  = np.count_nonzero(value["rewards"].sum(1))
    #     episode_num     = len(episode_folders)
    #     bunshi          = episode_num-non_zero_count
    #     task_success = f"{bunshi}/{episode_num}"
    #     post_processed_data[key]={
    #                             "success_rate"          : task_success,
    #                             "reward"                : value["rewards"].sum(1).ravel().tolist(),
    #                             "reward_mean"           : value["rewards"].sum(1).mean(),
    #                             "reward_std"            : value["rewards"].sum(1).std(),
    #                             "reward_var"            : value["rewards"].sum(1).var(),
    #                             "removal_trans"         : (100-value["infos"]).mean(0).tolist(),
    #                             "removal_trans_std"     : (100-value["infos"]).std(0).tolist(),
    #                             "removal_trans_var"     : (100-value["infos"]).var(0).tolist(),
    #                             "removal_performance"         : (value["removal_performance"]).mean(0).tolist(),
    #                             "removal_performance_std"     : (value["removal_performance"]).std(0).tolist(),
    #                             "removal_performance_var"     : (value["removal_performance"]).var(0).tolist(),
    #                             "image_pred_loss"             : value["image_pred_loss"].mean(0).tolist(),
    #                             "image_pred_loss_std"             : value["image_pred_loss"].std(0).tolist(),
    #                             "image_pred_loss_var"             : value["image_pred_loss"].var(0).tolist(),
    #                             "augnet_loss"                   : value["augnet_loss"].mean(0).tolist(),
    #                             "augnet_loss_std"               : value["augnet_loss"].std(0).tolist(),
    #                             "augnet_loss_var"               : value["augnet_loss"].var(0).tolist(),
    #                             "augnet_loss_2"                   : value["augnet_loss_2"].mean(0).tolist(),
    #                             "augnet_loss_2_std"               : value["augnet_loss_2"].std(0).tolist(),
    #                             "augnet_loss_2_var"               : value["augnet_loss_2"].var(0).tolist(),
    #                             "image_pred_loss_mode"                 : value["image_pred_loss_mode"].mean(0).tolist(),
    #                             "image_pred_loss_mode_std"             : value["image_pred_loss_mode"].std(0).tolist(),
    #                             "image_pred_loss_mode_var"             : value["image_pred_loss_mode"].var(0).tolist(),
    #                                 }



    # # serialized_data_path = os.path.normpath(data_dir+"/")
    # # with open(save_folders+"/post_processed_data.json", mode="w") as f:
    # # with open(save_folders+"/post_processed_data_v3.json", mode="w") as f:
    # # with open(save_folders+"/post_processed_data_v4.json", mode="w") as f:
    # with open(save_folders+"/post_processed_data_v5.json", mode="w") as f:
    #     json.dump(post_processed_data, f, indent=8,cls=MyEncoder)

    # pickle_utils().save(dataset=episodes_trj,save_path=save_folders+"/serialized_trj.pickle")


    # import ipdb;ipdb.set_trace()


    # image_loss_np = np.asarray(image_loss)

    # data = {"image_loss":image_loss_np.tolist(),
    #         "image_loss_mean":image_loss_np.mean(),
    #         "image_loss_std" :image_loss_np.std(),
    #         "image_loss_var" :image_loss_np.var(),
    #         "slice_tag"      : slice_tag
    #         }


    # save_name = save_folder+"/post_processed_data.json"
    # # save_yaml(data=data,save_path=save_name)
    # save_json(data=data,save_path=save_name)