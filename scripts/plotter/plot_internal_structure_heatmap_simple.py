

import numpy as np
import cv2
from denoising_diffusion_pytorch.utils.os_utils import save_json ,get_folder_name,create_folder,pickle_utils,get_path
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy ,pil_image_load_to_numpy,numpy_to_pil
from scripts.plotter.plot_internal_structure_heatmap import create_crosshatch_pattern, prob2cmap_jet,img2cubic,get_each_predicted_image
import os
from tqdm import tqdm



import shutil

def copy_file_to_structure(source_file_path, output_root='output_dir'):
    # 絶対パス化
    source_file_path = os.path.abspath(source_file_path)
    source_filename = os.path.basename(source_file_path)

    # 新しいルートフォルダ作成
    os.makedirs(output_root, exist_ok=True)

    # 5個のサブフォルダ作成
    for i in range(8):
        subfolder = os.path.join(output_root, f'folder_{i}')
        os.makedirs(subfolder, exist_ok=True)

        # 同じファイルを8個コピー（ファイル名を変える）
        for j in range(8):
            new_filename = f"{os.path.splitext(source_filename)[0]}_{j}{os.path.splitext(source_filename)[1]}"
            dst = os.path.join(subfolder, new_filename)
            shutil.copy(source_file_path, dst)

    print(f"Success: {output_root}")




def pre_process_for_acq(reconsts,color_definitions=None):
    """
    各カテゴリ（battery, motorなど）ごとに、RGBの範囲指定に基づくマスクを作成し、
    マスクがTrueになった画素数の割合を返す。
    Parameters:
        reconsts (np.ndarray): shape = (N, H, W, 3), RGB画像群 (uint8)
        color_definitions (dict): target_mask, lb, ub を含む辞書
    Returns:
        dist (dict): 各カテゴリごとの検出割合マップ (shape = [H, W])
    """
    color_definitions = {"battery": {"target_mask"  :np.asarray([0.2,0.8,0.8]),
                                    "target_mask_lb":np.asarray([0.2,0.8,0.8])-np.asarray([0.25,0.25,0.25]),
                                    "target_mask_ub":np.asarray([0.2,0.8,0.8])+np.asarray([0.25,0.25,0.25])}, #np.asarray([0.7,0.2,0.2]
                        "motor":    {"target_mask"  :np.asarray([0.8,0.2,0.2]),
                                    "target_mask_lb":np.asarray([0.8,0.2,0.2])-np.asarray([0.1,0.1,0.1]),
                                    "target_mask_ub":np.asarray([0.8,0.2,0.2])+np.asarray([0.2,0.2,0.2])}, #np.asarray([0.2,0.6,0.6])
                        "pcb":      {"target_mask"   :np.asarray([0.8,0.8,0.2]),
                                    "target_mask_lb":np.asarray([0.8,0.8,0.2])-np.asarray([0.1,0.1,0.1]),
                                    "target_mask_ub":np.asarray([0.8,0.8,0.2])+np.asarray([0.2,0.2,0.6])}
                        }

    dist = {}
    for item_name, color_info in color_definitions.items():
        # 色範囲（0.0〜1.0） → [0〜255] に変換
        lb = np.clip(color_info["target_mask_lb"] * 255, 0, 255).astype(np.uint8)
        ub = np.clip(color_info["target_mask_ub"] * 255, 0, 255).astype(np.uint8)

        mask = np.ones(reconsts.shape[:-1], dtype=bool)
        for i in range(3):  # R, G, B チャンネル
            mask &= (reconsts[..., i] >= lb[i]) & (reconsts[..., i] <= ub[i])

        # 各位置におけるマスクのTrueの割合
        dist[item_name] = np.sum(mask, axis=0) / len(reconsts)


    return dist['battery'], dist['pcb'], dist['motor']
    # return dist


if __name__ == '__main__':


    grid_3dim  = 16

    grid_2dim  = 344
    # grid_2dim  = 1024
    # grid_2dim  = 2048


    obj_name        = 'simple_model'
    axis            = "x"  # "z" or "x"
    tags            = "epsilon_greedy_00"
    # tags            = "no_cond"
    # tags            = "oracle_obs_for_gen_heatmap"



    # root_folder_    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456_clip_ucb_raw_0.5_v13_1/"
    # root_folder_    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456_clip_ucb_raw_0.5_v14_1/"
    # root_folder_   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_vaeac_a123456_clip_ucb_raw_0.5_v13_2"

    # root_folder_  = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456789_clip_ucb_raw_0.5_v12_1"
    # root_folder_   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT100000_B32_T8_partial_obs_vaeac_a123456789_clip_ucb_raw_0.5_v13_2_tmp"
    root_folder_   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/dataset_4142435161_13901k_v1_2/PT100000_B32_T8_partial_obs_diffusion_1D_a123456789_clip_ucb_raw_0.5_v13_2"


    rs_cl_img = img = np.full((grid_2dim, grid_2dim, 3), (255,255,255), dtype=np.uint8)


    root_folder = f"{root_folder_}/{tags}"
    model_type_folder = get_folder_name(root_folder)
    ###########################################################################
    # create sampled images folder for oracle_obs_for_gen_heatmap at once
    ###########################################################################
    # for j in tqdm(range(len(model_type_folder))):
    #     episodes_folder = root_folder+"/"+model_type_folder[j]
    #     episodes        = get_folder_name(episodes_folder)
    #     for i in range(len(episodes)):
    #         # 使用例（source_file_path を任意のファイルに変更して使ってください）
    #         copy_file_to_structure(source_file_path=episodes_folder+f"/episode_{i}/oracle_obs_cast_z_axis0.png",
    #                             output_root=f"{episodes_folder}/episode_{i}/raw_pred_image/")
    # import ipdb;ipdb.set_trace()

    for j in tqdm(range(len(model_type_folder))):
    # for j in range(5,6):

        episodes_folder = root_folder+"/"+model_type_folder[j]
        episodes        = get_folder_name(episodes_folder)

        for i in range(len(episodes)):
            data_folder         = episodes_folder+"/"+episodes[i]+"/raw_pred_image"
            raw_pred_image_dirs = get_folder_name(data_folder)

            all_predicted_images_in_episode = []
            # Loop through each raw predicted image directory
            for k in range(len(raw_pred_image_dirs)):
                observed_image_dir = data_folder+f"/../{k}_seq_obs_cast_z_axis{k}_0.png"
                observed_image = (pil_image_load_to_numpy(observed_image_dir) * 255.0).astype(np.uint8)
                mask = (observed_image == 0).all(axis=2).astype(np.float32)

                raw_pred_image_dir = data_folder+"/"+raw_pred_image_dirs[k]

                # Get each predicted image
                all_predicted_images = get_each_predicted_image(raw_pred_image_dir, target_ext=".png")
                #convert to numpy array
                all_predicted_images = np.array(all_predicted_images)
                # all_predicted_images_in_episode.append(all_predicted_images)
                all_predicted_images_in_episode.append(all_predicted_images* mask[None, :, :, None])  # Apply mask to each predicted image

            # Convert the list of all predicted images in the episode to a numpy array
            all_predicted_images_in_episode = np.array(all_predicted_images_in_episode)
            reconsts = all_predicted_images_in_episode




            dists = {'battery': [], 'pcb': [], 'motor': []}
            # convert 2D images to 3D cubic representations
            # cubics['motor'][0].shape -> (49, 49, 49, 3) if img_size = 49
            for reconst_each_step in reconsts:
                _db, _dp, _dm = pre_process_for_acq(reconst_each_step)
                dists['battery'].append(_db)
                dists['pcb'].append(_dp)
                dists['motor'].append(_dm)



            cubics = {'battery': [], 'pcb': [], 'motor': []}

            for key, dist_list in dists.items():
                for _dist in dist_list:
                    _c = img2cubic(_dist, grid_3dim)
                    cubics[key].append(_c)



            if obj_name == 'simple_model':
                ########################
                # direction = Z version
                #########################
                if axis == 'z':
                    cum_axis = 2
                    num_rot = 2
                ##########################
                ## direction = X version
                ##########################
                elif axis == 'x':
                    cum_axis = 0 if axis == 'x' else 2
                    num_rot =  -1
                else:
                    import ipdb;ipdb.set_trace()

            for part_name in ['battery', 'pcb', 'motor']:
                # for step, label in zip([0, -1], ['Init', 'Last']):
                for step in range(len(cubics[part_name])):
                    each_cubic = cubics[part_name][step]

                    # 奥行き方向の確率の扱いについては最大値で処理 (回転操作はなくしたかったが...)
                    each_dist_img = np.rot90(np.max(each_cubic, axis=cum_axis), num_rot)
                    # if axis == 'x':
                    #     each_dist_img = np.fliplr(each_dist_img)
                    # 確率 -> jet
                    # cv2のjetは最大値と最小値の色の発色が良くなかったので、each_dist_img*0.8+0.1とした。
                    # 上記の処理をしているのでcolormapは注意する必要がある。
                    each_jet = prob2cmap_jet(each_dist_img*0.8+0.1)
                    rs_each_jet = cv2.resize(each_jet, (grid_2dim, grid_2dim), interpolation=cv2.INTER_NEAREST)

                    rs_each_with_mask = cv2.bitwise_or(cv2.bitwise_not(rs_cl_img), rs_each_jet)
                    # rs_each_with_mask = rs_each_jet



                    save_dir = os.path.normpath(raw_pred_image_dir+"/../../internal_stricture_heatmap3")
                    # if os.path.exists(save_dir):
                    #     shutil.rmtree(save_dir)
                    #     print("delete old folder")
                    # else:
                    #     print(f"{save_dir}:cannot find folder")

                    create_folder(save_dir)
                    save_name = f"{save_dir}/{part_name}_{axis}_{step}.png"
                    pil_image_save_from_numpy(rs_each_with_mask/255.0, save_name)


            # # for ground_truth image
            # for part_name in ['battery', 'pcb', 'motor']:
            #     for iter, label in zip([0, -1], ['Init', 'Last']):
            #         each_cubic = cubics[part_name][iter]

            #         # 奥行き方向の確率の扱いについては最大値で処理 (回転操作はなくしたかったが...)
            #         each_dist_img = np.rot90(np.max(each_cubic, axis=cum_axis), num_rot)
            #         # if axis == 'x':
            #             # each_dist_img = np.fliplr(each_dist_img)
            #         # 確率 -> jet
            #         # cv2のjetは最大値と最小値の色の発色が良くなかったので、each_dist_img*0.8+0.1とした。
            #         # 上記の処理をしているのでcolormapは注意する必要がある。
            #         each_jet = prob2cmap_jet(each_dist_img*0.8+0.1)

            #         if part_name == 'battery':
            #             each_jet[np.all(each_jet == [228, 0, 0], axis=-1)]=[0,0,225]
            #             # each_jet[np.all(each_jet == [228, 0, 0], axis=-1)]=hatch_image[np.all(each_jet == [228, 0, 0], axis=-1)]
            #         elif part_name == 'pcb':
            #             each_jet[np.all(each_jet == [228, 0, 0], axis=-1)]=[0,225,0]
            #         elif part_name == 'motor':
            #             each_jet[np.all(each_jet == [228, 0, 0], axis=-1)]=[225,0,0]

            #         each_jet[np.all(each_jet == [0, 0, 232], axis=-1)]=[220,220,220]

            #         rs_each_jet = cv2.resize(each_jet, (grid_2dim, grid_2dim), interpolation=cv2.INTER_NEAREST)

            #         hatch_image = np.asarray(create_crosshatch_pattern(size=(grid_2dim, grid_2dim), spacing=9, line_color=(0, 0, 0), bg_color=(220, 220, 220),line_width=2))
            #         # pil_image_save_from_numpy(hatch_image/255.0, f"./hatch_image.png")
            #         rs_each_jet[np.all(rs_each_jet == [0, 0, 225], axis=-1)] = hatch_image[np.all(rs_each_jet == [0, 0, 225], axis=-1)]

            #         # 二値製品画像によるマスク処理。上下左右はいい感じにトリミングしている。
            #         # rs_each_with_mask = cv2.bitwise_or(cv2.bitwise_not(rs_cl_img), rs_each_jet)[128:-128, 60:-60]
            #         # rs_each_with_mask = cv2.bitwise_or(cv2.bitwise_not(rs_cl_img), rs_each_jet)[95:-95, 50:-50] # 344
            #         rs_each_with_mask = cv2.bitwise_or(cv2.bitwise_not(rs_cl_img), rs_each_jet) #1024

            #         save_dir = os.path.normpath(raw_pred_image_dir+"/../../internal_stricture_heatmap3")
            #         # if os.path.exists(save_dir):
            #         #     shutil.rmtree(save_dir)
            #         #     print("delete old folder")
            #         # else:
            #         #     print(f"{save_dir}:cannot find folder")

            #         create_folder(save_dir)
            #         save_name = f"{save_dir}/{part_name}_{axis}_{label}.png"
            #         pil_image_save_from_numpy(rs_each_with_mask/255.0, save_name)
