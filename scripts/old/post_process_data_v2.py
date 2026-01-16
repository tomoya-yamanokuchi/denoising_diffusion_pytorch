
import os
import numpy as np
import json
import cv2

import torch
from torch import nn
import imgsim
from tqdm import tqdm

from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.os_utils import get_folder_name,pickle_utils,MyEncoder,get_path
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_load_to_numpy,pil_image_save_from_numpy
from denoising_diffusion_pytorch.utils.arrays import to_torch,to_device,to_np



grid_2dim    = 512
grid_3dim    = 64

def get_augnet_loss(target_image,pred_image):

        vec0 = vtr.vectorize(target_image)
        vec1 = vtr.vectorize(pred_image)

        augnet_loss = imgsim.distance(vec0, vec1)

        return augnet_loss







def get_2d_image_to_mini_batch_image(image,permute):
    # box_arrays_data =  self.get_box_array_data()

    grid_2dim    = image.shape[0]
    # grid_3dim    = 16
    batch_img_len = int(grid_2dim/grid_3dim)


    batch_2d_image_ = np.zeros((grid_3dim, grid_3dim, grid_3dim, 3))
    k = 0
    for j in range(batch_img_len):
        for i in range(batch_img_len):
            batch_2d_image_[k] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
            k = k+1


    if permute == "z":
        batch_2d_image  = batch_2d_image_
    elif permute == "y":
        batch_2d_image  = batch_2d_image_.transpose(1,0,2,3)
    elif permute == "x":
        batch_2d_image  = batch_2d_image_.transpose(2,1,0,3)

    return batch_2d_image


def get_box_color_to_2d_image(box_color=None,permute = "x"):

    # grid_2dim    = 64
    # grid_3dim    = 16
    batch_img_len = int(grid_2dim/grid_3dim)

    # box_color[0]    = np.asarray([0.9,0.2,0.2])
    # box_color[15]   = np.asarray([0.9,0.2,0.2])
    # box_color[240]  = np.asarray([0.9,0.2,0.2])
    # box_color[255]  = np.asarray([0.9,0.2,0.2])

    batch_2d_image_ = box_color.reshape(grid_3dim, grid_3dim, grid_3dim, 3)
    cast_image = np.empty((grid_2dim,grid_2dim,3))

    if permute == "z":
        batch_2d_image  = batch_2d_image_
    elif permute == "y":
        batch_2d_image  = batch_2d_image_.transpose(1,0,2,3)
    elif permute == "x":
        batch_2d_image  = batch_2d_image_.transpose(2,1,0,3)

    k = 0
    for j in range(batch_img_len):
        for i in range(batch_img_len):
            cast_image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim] = batch_2d_image[k]
            k = k+1

    return cast_image


def get_color_mask_image(images,mask_config):

    cost_map = {}
    for idx, val in enumerate(images):
        cost_map[val] = color_range_mask(image=images[val],mask_config=mask_config)

    return cost_map


def color_range_mask(image, mask_config):
    """
    2次元画像から指定した色の範囲内であれば1、そうでなければ0を判定する関数
    Args:
        image (numpy.ndarray): 2次元画像（16x16のRGB画像）
        lower_bound (tuple): 色の下限範囲 (R, G, B)
        upper_bound (tuple): 色の上限範囲 (R, G, B)
    Returns:
        numpy.ndarray: 範囲内であれば1、そうでなければ0を持つ2次元マスク
    """
    lower_bound = mask_config["target_mask_lb"]
    upper_bound = mask_config["target_mask_ub"]

    # RGB画像の各チャンネルを個別に比較
    mask_r = (image[:,:,0] >= lower_bound[0]) & (image[:,:,0] <= upper_bound[0])
    mask_g = (image[:,:,1] >= lower_bound[1]) & (image[:,:,1] <= upper_bound[1])
    mask_b = (image[:,:,2] >= lower_bound[2]) & (image[:,:,2] <= upper_bound[2])

    # # 全てのチャンネルで条件を満たしているピクセルを1、それ以外のピクセルを0とする
    # mask = mask_r & mask_g & mask_b
    # return mask.astype(int)


    # # 全てのチャンネルで条件を満たしているピクセルを黒、それ以外のピクセルを白とする
    # mask = np.zeros_like(mask_r, dtype=np.uint8)  # 黒のマスクを初期化
    # mask[mask_r & mask_g & mask_b] = 1.0 # 全てのチャンネルで条件を満たすピクセルに白を設定


    # mask = np.zeros_like(image,dtype=np.uint8)
    # mask[:,:,0]=mask_r*0.0
    # mask[:,:,1]=mask_g*0.0
    # mask[:,:,2]=mask_b*0.0

    # return mask

    # 全てのチャンネルで条件を満たしているピクセルを白、それ以外のピクセルを黒とする
    mask = np.zeros_like(image, dtype=np.uint8)  # 黒のマスクを初期化
    mask[mask_r & mask_g & mask_b] = [1., 1., 1.]  # 全てのチャンネルで条件を満たすピクセルに白を設定
    return mask

def find_false_true_false_indices(lst):
    result = []
    start = -1

    for i in range(len(lst)):
        if lst[i] == False:
            if start != -1 and i - start > 1:
                result.extend(range(start + 1, i))
            start = i
    return np.asarray(result)



def get_center_mass(cost,oracle):


    if len(cost)>=2:
        oracle_x_center_x = (cost[0]+cost[-1])/2.0
    elif len(cost)>=1:
        oracle_x_center_x = cost[0]
    else :
        oracle_x_center_x = oracle
    return oracle_x_center_x



def get_box_center(oracle_image=None ,oracle = None):

    oracle_image_z_mini_batch = get_2d_image_to_mini_batch_image(oracle_image,"z")
    oracle_image_x            = get_box_color_to_2d_image(box_color=oracle_image_z_mini_batch,permute="x")
    oracle_image_y            = get_box_color_to_2d_image(box_color=oracle_image_z_mini_batch,permute="y")



    oracle_images = {   "image_x":oracle_image_x,
                        "image_y":oracle_image_y,
                        "image_z":oracle_image,}

    oracle_cost_tmp = get_color_mask_image(oracle_images,image_mask_config_b)



    oracle_cost_x= find_false_true_false_indices(get_2d_image_to_mini_batch_image(oracle_cost_tmp["image_x"],"z").sum(3).sum(1).sum(1))
    oracle_cost_y= find_false_true_false_indices(get_2d_image_to_mini_batch_image(oracle_cost_tmp["image_y"],"z").sum(3).sum(1).sum(1))
    oracle_cost_z= find_false_true_false_indices(get_2d_image_to_mini_batch_image(oracle_cost_tmp["image_z"],"z").sum(3).sum(1).sum(1))




    # if len(oracle_cost_x)>=2:
    #     oracle_x_center_x = (oracle_cost_x[0]+oracle_cost_x[-1])/2.0
    # elif len(oracle_cost_x)>=1:
    #     oracle_x_center_x = oracle_cost_x[0]
    # else :
    #     oracle_x_center_x = 7.0



    # if len(oracle_cost_y)>=2:
    #     oracle_y_center_y = (oracle_cost_y[0]+oracle_cost_y[-1])/2.0
    # elif len(oracle_cost_y)>=1:
    #     oracle_y_center_y = oracle_cost_y[0]
    # else:
    #     oracle_center_y = 7.0


    # if len(oracle_cost_z)>=2:
    #     oracle_z_center_z = (oracle_cost_z[0]+oracle_cost_z[-1])/2.0
    # else:
    #     oracle_z_center_z = oracle_cost_z[0]

    if oracle is not None:
    
        oracle_x_center_x = get_center_mass(oracle_cost_x,oracle["center_x"])
        oracle_y_center_y = get_center_mass(oracle_cost_y,oracle["center_y"])
        oracle_z_center_z = get_center_mass(oracle_cost_z,oracle["center_z"])

    else:
        
        oracle_x_center_x = get_center_mass(oracle_cost_x,0.0)
        oracle_y_center_y = get_center_mass(oracle_cost_y,0.0)
        oracle_z_center_z = get_center_mass(oracle_cost_z,0.0)




    return {"center_x": oracle_x_center_x,
            "center_y": oracle_y_center_y,
            "center_z": oracle_z_center_z,}




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
    vtr = imgsim.Vectorizer()

    episodes_trj = {}
    pixel_loss_ =[]


    target_mask_b = np.asarray([0.2,0.8,0.8])
    image_mask_config_b = {"target_mask":target_mask_b,
                        "target_mask_lb":target_mask_b-np.asarray([0.1,0.1,0.1]),
                        "target_mask_ub":target_mask_b+np.asarray([0.7,0.2,0.2])}




    # evaluate folder loop
    for i in range(len(eval_folders)):

        episode_folders = get_folder_name(os.path.join(save_folders,eval_folders[i]))

        pixel_loss_r  = []
        augnet_loss_r =[]
        augnet_loss_r_2 =[]
        
        pred_center_x =[]
        pred_center_y =[]
        pred_center_z =[]


        oracle_center_x =[]
        oracle_center_y =[]
        oracle_center_z =[]


        
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




            oracle_center = get_box_center(oracle_image=oracle_image)
            print(f"oracle_center:{oracle_center}")




            if args.policy_config["ctrl_mode"] == "oracle_obs" or args.policy_config["ctrl_mode"] == "random" :

                for k in range(len(condition_image_path_z)):
                    condition_image = pil_image_load_to_numpy(condition_image_path_z[k])
                    ensemble_image  = pil_image_load_to_numpy(data_dir+f"/{k}_ensemble_z_axis{k}_0.png")
                    target_image    = oracle_image*(condition_image<=0)
                    pred_image      = ensemble_image*(condition_image<=0)
                    un_condition_mask = np.where(condition_image==0.0)[0].shape[0]/condition_image.shape[2]

                    aa = to_torch(target_image*255.0)
                    bb = to_torch(pred_image*255.0)
                    loss = to_np(mse_loss_fn(aa,bb))

                    pixel_loss = loss/un_condition_mask
                    pixel_loss_r.append(pixel_loss)
    
                    target_image_bgr =  cv2.cvtColor((target_image*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    pred_image_bgr   =  cv2.cvtColor((pred_image*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    # augnet_loss      = get_augnet_loss(target_image=target_image_bgr,pred_image=pred_image_bgr)
                    augnet_loss      = 0.0
                    
                    augnet_loss_r.append((augnet_loss/un_condition_mask)*1000)
                    augnet_loss_r_2.append(augnet_loss)


                    pred_center = get_box_center(oracle_image=ensemble_image, oracle = oracle_center)
    
                    pred_center_x.append(np.abs(pred_center["center_x"]-pred_center["center_x"]))
                    pred_center_y.append(np.abs(pred_center["center_y"]-pred_center["center_y"]))
                    pred_center_z.append(np.abs(pred_center["center_z"]-pred_center["center_z"]))


                    oracle_center_x.append(oracle_center["center_x"])
                    oracle_center_y.append(oracle_center["center_y"])
                    oracle_center_z.append(oracle_center["center_z"])

                # print(f"oracle_center:{oracle_center}")
                # print(f"augnet_loss:{augnet_loss_r}")


            else:
                for k in tqdm(range(len(condition_image_path_z))):
                    condition_image = pil_image_load_to_numpy(condition_image_path_z[k])
                    ensemble_image_path,_ = get_path(data_dir+f"/raw_pred_image/step_{k}",".png")
                    pixel_loss_r_tmp = []
                    augnet_loss_r_tmp =[]
                    augnet_loss_r_tmp_2 =[]

                    pred_center_x_tmp =[]
                    pred_center_y_tmp =[]
                    pred_center_z_tmp =[]

                    for p in range(len(ensemble_image_path)):
                        ensemble_image  = pil_image_load_to_numpy(ensemble_image_path[p])
                        target_image    = oracle_image*(condition_image<=0)
                        pred_image      = ensemble_image*(condition_image<=0)
                        un_condition_mask = np.where(condition_image==0.0)[0].shape[0]/condition_image.shape[2]

                        aa = to_torch(target_image*255.0)
                        bb = to_torch(pred_image*255.0)
                        loss = to_np(mse_loss_fn(aa,bb))


                        pixel_loss = loss/un_condition_mask
                        pixel_loss_r_tmp.append(pixel_loss)


                        target_image_bgr =  cv2.cvtColor((target_image*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        pred_image_bgr   =  cv2.cvtColor((pred_image*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        # augnet_loss      = get_augnet_loss(target_image=target_image_bgr,pred_image=pred_image_bgr)
                        augnet_loss      = 0.0
                        augnet_loss_r_tmp.append((augnet_loss/un_condition_mask)*1000)
                        augnet_loss_r_tmp_2.append(augnet_loss)
                        


                        pred_center = get_box_center(oracle_image=ensemble_image,oracle = oracle_center)
                        pred_center_x_tmp.append(np.abs(pred_center["center_x"]-oracle_center["center_x"]))
                        pred_center_y_tmp.append(np.abs(pred_center["center_y"]-oracle_center["center_y"]))
                        pred_center_z_tmp.append(np.abs(pred_center["center_z"]-oracle_center["center_z"]))


                    pixel_loss_r.append(np.asarray(pixel_loss_r_tmp).mean())
                    augnet_loss_r.append(np.asarray(augnet_loss_r_tmp).mean())
                    augnet_loss_r_2.append(np.asarray(augnet_loss_r_tmp_2).mean())


                    pred_center_x.append(np.asarray(pred_center_x_tmp).mean())
                    pred_center_y.append(np.asarray(pred_center_y_tmp).mean())
                    pred_center_z.append(np.asarray(pred_center_z_tmp).mean())

                    oracle_center_x.append(oracle_center["center_x"])
                    oracle_center_y.append(oracle_center["center_y"])
                    oracle_center_z.append(oracle_center["center_z"])


                    print(f"augnet_loss:{np.asarray(augnet_loss_r_tmp).mean()}")
                    print(f"augnet_loss:{np.asarray(augnet_loss_r_tmp_2).mean()}")
    
                    print(f"oracle_center:{oracle_center}")
                    print(f"pred_center:{np.asarray(pred_center_x_tmp).mean()}")
                    print(f"pred_center:{np.asarray(pred_center_y_tmp).mean()}")
                    print(f"pred_center:{np.asarray(pred_center_z_tmp).mean()}")


        # import ipdb;ipdb.set_trace()

        episodes_trj[eval_folders[i]]["image_pred_loss"] = np.asarray(pixel_loss_r).reshape(len(episode_folders),-1)
        episodes_trj[eval_folders[i]]["augnet_loss"] = np.asarray(augnet_loss_r).reshape(len(episode_folders),-1)
        episodes_trj[eval_folders[i]]["augnet_loss_2"] = np.asarray(augnet_loss_r_2).reshape(len(episode_folders),-1)


        episodes_trj[eval_folders[i]]["pred_center_x"] = np.asarray(pred_center_x).reshape(len(episode_folders),-1)
        episodes_trj[eval_folders[i]]["pred_center_y"] = np.asarray(pred_center_y).reshape(len(episode_folders),-1)
        episodes_trj[eval_folders[i]]["pred_center_z"] = np.asarray(pred_center_z).reshape(len(episode_folders),-1)

        episodes_trj[eval_folders[i]]["oracle_center_x"] = np.asarray(oracle_center_x).reshape(len(episode_folders),-1)
        episodes_trj[eval_folders[i]]["oracle_center_y"] = np.asarray(oracle_center_y).reshape(len(episode_folders),-1)
        episodes_trj[eval_folders[i]]["oracle_center_z"] = np.asarray(oracle_center_z).reshape(len(episode_folders),-1)


                # pil_image_save_from_numpy(target_image,f"./{k}_target_image.png")
                # pil_image_save_from_numpy(pred_image,f"./{k}_pred_image.png")




    post_processed_data =  {}
    for key,value  in episodes_trj.items():

        print(f"load_dirs:{key}")
        # import ipdb;ipdb.set_trace()
        tmps = value["image_pred_loss"].mean(0).tolist()
        print(f"image_loss: {tmps}")
        ## save some statics values
        post_processed_data[key]={
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
                                "pred_center_x"                   : value["pred_center_x"].mean(0).tolist(),
                                "pred_center_x_std"               : value["pred_center_x"].std(0).tolist(),
                                "pred_center_x_var"               : value["pred_center_x"].var(0).tolist(),
                                "pred_center_y"                   : value["pred_center_y"].mean(0).tolist(),
                                "pred_center_y_std"               : value["pred_center_y"].std(0).tolist(),
                                "pred_center_y_var"               : value["pred_center_y"].var(0).tolist(),
                                "pred_center_z"                   : value["pred_center_z"].mean(0).tolist(),
                                "pred_center_z_std"               : value["pred_center_z"].std(0).tolist(),
                                "pred_center_z_var"               : value["pred_center_z"].var(0).tolist(),
                                "oracle_center_x"                   : value["oracle_center_x"].mean(0).tolist(),
                                "oracle_center_y"                   : value["oracle_center_y"].mean(0).tolist(),
                                "oracle_center_z"                   : value["oracle_center_z"].mean(0).tolist(),

                                    }



    # serialized_data_path = os.path.normpath(data_dir+"/")
    with open(save_folders+"/post_processed_data_v2.json", mode="w") as f:
        json.dump(post_processed_data, f, indent=8,cls=MyEncoder)

    pickle_utils().save(dataset=episodes_trj,save_path=save_folders+"/serialized_trj_v2.pickle")


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