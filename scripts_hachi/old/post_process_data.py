
import os
import numpy as np
import json
import cv2

import torch
from torch import nn
import imgsim
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F

from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.os_utils import get_folder_name,pickle_utils,MyEncoder,get_path
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_load_to_numpy,pil_image_save_from_numpy
from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np



def get_augnet_loss(target_image,pred_image):

        vec0 = vtr.vectorize(target_image)
        vec1 = vtr.vectorize(pred_image)

        augnet_loss = imgsim.distance(vec0, vec1)

        return augnet_loss


def count_mode(arr):
    # 出現回数をカウント
    counts = Counter(arr)

    # 最大の出現回数を取得
    max_count = max(counts.values())

    # 最頻値をすべて取得
    modes = [key for key, count in counts.items() if count == max_count]

    # 最頻値が複数ある場合はその平均を計算
    if len(modes) > 1:
        mode_average = np.mean(modes)
    else:
        mode_average = modes[0]

    return mode_average

if __name__ == '__main__':


    # save_folder     = f"./results/voxel_image_64"+f"/eval_{13}_intermediate//"
    # save_folder     = f"./results/voxel_image_64"+f"/eval_{13}_soft_compound_1_2/"
    # save_folder     = f"./results/voxel_image_64"+f"/eval_{13}_mid_v2/"
    # save_folder     = f"./results/voxel_image_64"+f"/eval_{6}_hard_v2/"
    # save_folder     = f"./results/voxel_image_w_multi_color_v1_64_v1"+f"/eval_{6}_hard_v2/"



    #---------------------------------- setup ----------------------------------#


    class Parser(parser):
        dataset: str = 'Image_diffusion_2D'
        config: str =  'config.vae'


    args = Parser().parse_args('diffusion_plan')


    source_dir   = args.savepath
    save_folders = source_dir
    eval_folders = get_folder_name(save_folders)

    mse_loss_fn = nn.MSELoss()
    # mse_loss_fn = nn.L1Loss()

    vtr = imgsim.Vectorizer()

    episodes_trj = {}
    pixel_loss_ =[]
    # evaluate folder loop
    for i in range(len(eval_folders)):

        episode_folders = get_folder_name(os.path.join(save_folders,eval_folders[i]))

        pixel_loss_r  = []
        pixel_loss_r_mode  = []
        augnet_loss_r =[]
        augnet_loss_r_2 =[]

        for j in range(len(episode_folders)): # episode folder loop

            data_dir  = os.path.join(save_folders, eval_folders[i], episode_folders[j])
            load_data = pickle_utils().load(load_path = data_dir+f"/rollout_data.pickle")

            if j != 0:
                for key,value in load_data.items():
                    episodes_trj[eval_folders[i]][key]= np.vstack((episodes_trj[eval_folders[i]][key], load_data[key]))
            else:
                episodes_trj[eval_folders[i]]= load_data


            oracle_image  = pil_image_load_to_numpy(data_path=data_dir+"/oracle_obs_cast_z_axis0.png")
            condition_image_path,condition_image_name = get_path(data_dir+"/conditions",".png")
            condition_image_path_z      = list(filter(lambda item: "axis_z" in item, condition_image_path))



            if args.policy_config["ctrl_mode"] == "oracle_obs" or args.policy_config["ctrl_mode"] == "random" :
            # if args.policy_config["ctrl_mode"] is not None:

                for k in range(len(condition_image_path_z)):
                    condition_image = pil_image_load_to_numpy(condition_image_path_z[k])
                    ensemble_image  = pil_image_load_to_numpy(data_dir+f"/{k}_ensemble_z_axis{k}_0.png")
                    target_image    = oracle_image*(condition_image<=0)
                    pred_image      = ensemble_image*(condition_image<=0)
                    un_condition_mask = np.where(condition_image==0.0)[0].shape[0]/condition_image.shape[2]

                    # import ipdb;ipdb.set_trace()
                    aa = to_torch(target_image*255.0)
                    bb = to_torch(pred_image*255.0)
                    loss = to_np(mse_loss_fn(aa,bb))

                    pixel_loss = loss/un_condition_mask
                    # pixel_loss = loss/1.0
                    pixel_loss_r.append(pixel_loss)
                    pixel_loss_r_mode.append(pixel_loss)

                    target_image_bgr =  cv2.cvtColor((target_image*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    pred_image_bgr   =  cv2.cvtColor((pred_image*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

                    # mssim, ssim = cv2.quality.QualitySSIM_compute(target_image_bgr/255, pred_image_bgr/255)
                    mssim, ssim = 0,0
                    # augnet_loss = 1 - ssim.mean(axis=2)
                    augnet_loss = 0.0


                    # augnet_loss      = get_augnet_loss(target_image=target_image_bgr,pred_image=pred_image_bgr)
                    # augnet_loss      = 10000.0

                    augnet_loss_r.append((augnet_loss*255/un_condition_mask)*1000)
                    # augnet_loss_r.append(augnet_loss)
                    augnet_loss_r_2.append(augnet_loss)


                print(f"augnet_loss:{augnet_loss_r}")


            else:
                # pass
                # action_idx = 0.0
                # action_flag = 0.0
                for k in tqdm(range(len(condition_image_path_z))):
                    # if load_data["actions"][k]!= 0:
                    #     action_idx+=1
                    # elif load_data["actions"][k]== 0 and action_flag == 0.0:
                    #     action_idx+=1
                    #     action_flag+=1
                    # else:
                    #     pass
                    if  args.policy_config["ctrl_mode"] == "no_cond":
                        condition_image = pil_image_load_to_numpy(condition_image_path_z[0])
                    elif args.policy_config["ctrl_mode"] == "epsilon_greedy_00":
                        condition_image = pil_image_load_to_numpy(condition_image_path_z[k])
                    ensemble_image_path,_ = get_path(data_dir+f"/raw_pred_image/step_{k}",".png")
                    pixel_loss_r_tmp = []
                    augnet_loss_r_tmp =[]
                    augnet_loss_r_tmp_2 =[]
                    for p in range(len(ensemble_image_path)):
                        ensemble_image  = pil_image_load_to_numpy(ensemble_image_path[p])
                        target_image    = oracle_image*(condition_image<=0)
                        pred_image      = ensemble_image*(condition_image<=0)
                        un_condition_mask = np.where(condition_image==0)[0].shape[0]/condition_image.shape[2]


                        aa = to_torch(target_image*255.0)
                        bb = to_torch(pred_image*255.0)
                        loss = to_np(mse_loss_fn(aa,bb))


                        pixel_loss = loss/un_condition_mask
                        pixel_loss_r_tmp.append(pixel_loss)


                        target_image_bgr =  cv2.cvtColor((target_image*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        pred_image_bgr   =  cv2.cvtColor((pred_image*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

                        # import ipdb;ipdb.set_trace()

                        # mssim, ssim = cv2.quality.QualitySSIM_compute(target_image_bgr/255, pred_image_bgr/255)
                        # mssim, ssim = 0.0,0.0
                        # augnet_loss= 1 - ssim.mean(axis=2)
                        augnet_loss= 1 - 0

                        psnr = cv2.PSNR(target_image_bgr, pred_image_bgr, R=255)

                        # augnet_loss      = get_augnet_loss(target_image=target_image_bgr,pred_image=pred_image_bgr)
                        # augnet_loss      = 0.0
                        augnet_loss_r_tmp.append((augnet_loss*255/un_condition_mask)*1000)
                        # augnet_loss_r_tmp.append((augnet_loss*255/((16-action_idx)*256))*1000)
                        # augnet_loss_r_tmp.append(augnet_loss)
                        augnet_loss_r_tmp_2.append(augnet_loss)
                        # augnet_loss_r_tmp_2.append(psnr/un_condition_mask)


                    pixel_loss_r_mode.append(count_mode(np.asarray(pixel_loss_r_tmp)))
                    pixel_loss_r.append(np.asarray(pixel_loss_r_tmp).mean())
                    augnet_loss_r.append(np.asarray(augnet_loss_r_tmp).mean())
                    augnet_loss_r_2.append(np.asarray(augnet_loss_r_tmp_2).mean())

                    print(f"pixel_mse_loss         :{np.asarray(pixel_loss_r_tmp).mean()}")
                    print(f"augnet_loss_normalize  :{np.asarray(augnet_loss_r_tmp).mean()}")
                    print(f"augnet_loss_unnormalize:{np.asarray(augnet_loss_r_tmp_2).mean()}")



        # import ipdb;ipdb.set_trace()

        episodes_trj[eval_folders[i]]["image_pred_loss"] = np.asarray(pixel_loss_r).reshape(len(episode_folders),-1)
        episodes_trj[eval_folders[i]]["image_pred_loss_mode"] = np.asarray(pixel_loss_r_mode).reshape(len(episode_folders),-1)
        episodes_trj[eval_folders[i]]["augnet_loss"] = np.asarray(augnet_loss_r).reshape(len(episode_folders),-1)
        episodes_trj[eval_folders[i]]["augnet_loss_2"] = np.asarray(augnet_loss_r_2).reshape(len(episode_folders),-1)
        # import ipdb;ipdb.set_trace()


                # pil_image_save_from_numpy(target_image,f"./{k}_target_image.png")
                # pil_image_save_from_numpy(pred_image,f"./{k}_pred_image.png")




    post_processed_data =  {}
    for key,value  in episodes_trj.items():

        print(f"load_dirs:{key}")
        # import ipdb;ipdb.set_trace()
        tmps = value["image_pred_loss"].mean(0).tolist()
        print(f"image_loss: {tmps}")
        ## save some statics values
        non_zero_count  = np.count_nonzero(value["rewards"].sum(1))
        episode_num     = len(episode_folders)
        bunshi          = episode_num-non_zero_count
        task_success = f"{bunshi}/{episode_num}"
        post_processed_data[key]={
                                "success_rate"          : task_success,
                                "reward"                : value["rewards"].sum(1).ravel().tolist(),
                                "reward_mean"           : value["rewards"].sum(1).mean(),
                                "reward_std"            : value["rewards"].sum(1).std(),
                                "reward_var"            : value["rewards"].sum(1).var(),
                                "removal_trans"         : (100-value["infos"]).mean(0).tolist(),
                                "removal_trans_std"     : (100-value["infos"]).std(0).tolist(),
                                "removal_trans_var"     : (100-value["infos"]).var(0).tolist(),
                                "removal_performance"         : (value["removal_performance"]).mean(0).tolist(),
                                "removal_performance_std"     : (value["removal_performance"]).std(0).tolist(),
                                "removal_performance_var"     : (value["removal_performance"]).var(0).tolist(),
                                "image_pred_loss"             : value["image_pred_loss"].mean(0).tolist(),
                                "image_pred_loss_std"             : value["image_pred_loss"].std(0).tolist(),
                                "image_pred_loss_var"             : value["image_pred_loss"].var(0).tolist(),
                                "augnet_loss"                   : value["augnet_loss"].mean(0).tolist(),
                                "augnet_loss_std"               : value["augnet_loss"].std(0).tolist(),
                                "augnet_loss_var"               : value["augnet_loss"].var(0).tolist(),
                                "augnet_loss_2"                   : value["augnet_loss_2"].mean(0).tolist(),
                                "augnet_loss_2_std"               : value["augnet_loss_2"].std(0).tolist(),
                                "augnet_loss_2_var"               : value["augnet_loss_2"].var(0).tolist(),
                                "image_pred_loss_mode"                 : value["image_pred_loss_mode"].mean(0).tolist(),
                                "image_pred_loss_mode_std"             : value["image_pred_loss_mode"].std(0).tolist(),
                                "image_pred_loss_mode_var"             : value["image_pred_loss_mode"].var(0).tolist(),
                                    }



    # serialized_data_path = os.path.normpath(data_dir+"/")
    # with open(save_folders+"/post_processed_data.json", mode="w") as f:
    # with open(save_folders+"/post_processed_data_v3.json", mode="w") as f:
    # with open(save_folders+"/post_processed_data_v4.json", mode="w") as f:
    with open(save_folders+"/post_processed_data_v5.json", mode="w") as f:
        json.dump(post_processed_data, f, indent=8,cls=MyEncoder)

    pickle_utils().save(dataset=episodes_trj,save_path=save_folders+"/serialized_trj.pickle")


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