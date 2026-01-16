import numpy as np
import matplotlib.pyplot as plt
import cv2
from denoising_diffusion_pytorch.utils.os_utils import save_json ,get_folder_name,create_folder,pickle_utils,get_path
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy ,pil_image_load_to_numpy,numpy_to_pil
import os
import shutil
from PIL import Image, ImageDraw

def prob2cmap_jet(prob: np.ndarray):
    if prob.ndim != 2:
        raise Exception()
    prob_uint8 = (255 * (1 - prob)).astype(np.uint8)
    prob_cmap = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
    return prob_cmap


def img2cubic(img: np.ndarray, img_size: int):
    if img.ndim == 2:
        cubic = np.zeros((img_size, img_size, img_size), dtype=img.dtype)
    elif img.ndim == 3:
        cubic = np.zeros((img_size, img_size, img_size, img.shape[-1]), dtype=img.dtype)
    else:
        raise ValueError(f"img.ndim == {img.ndim}")

    if np.sqrt(img_size) == int(np.sqrt(img_size)):
        for i in range(img_size):
            for j in range(img_size):
                for k in range(img_size):
                    q, mod = divmod(k, int(np.sqrt(img_size)))
                    cubic[img_size - 1 - i, j, k] = img[q * img_size + i, mod * img_size + j]
    elif img_size == 32:
        for i in range(img_size):
            for j in range(img_size):
                for k in range(img_size):
                    q, mod = divmod(k+16, 8)
                    cubic[img_size - 1 - i, j, k] = img[q * img_size + i, mod * img_size + j]
    else:
        raise NotImplementedError()
    return cubic


def get_each_predicted_image(data_dir,target_ext=".png"):
    """
    Get each predicted image from the directory.
    Args:
        data_dir (str): Directory containing the predicted images.
        target_ext (str): The file extension of the images to be retrieved.
    Returns:
        list: List of all predicted image
    """
    raw_pred_image_dirs, _ = get_path(data_dir, target_ext=target_ext)

    all_predicted_images = []
    for i in range(len(raw_pred_image_dirs)):
        pred_image = (pil_image_load_to_numpy(raw_pred_image_dirs[i]) * 255.0).astype(np.uint8)
        all_predicted_images.append(pred_image)
    return all_predicted_images




def pre_process_for_acq(reconsts: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):

    # name2color = {'battery': [0, 0, 255], 'pcb': [0, 255, 0], 'motor': [255, 0, 0]}
    threshold = 250
    name2color = {'battery': [0, 0, threshold], 'pcb': [0, threshold, 0], 'motor': [threshold, 0, 0]}

    dist = {}
    for item_name, color in name2color.items():
        mask = np.ones(reconsts.shape[:-1], bool)
        for i, each_ch in enumerate(color):
            cond_l, cond_h = max(0, (each_ch - (255-threshold))), min(255, each_ch + (255-threshold))
            # mask = cond_l < reconsts[..., i] < cond_h
            mask &= cond_l <= reconsts[..., i]
            mask &= reconsts[..., i] <= cond_h
        dist[item_name] = np.sum(mask, axis=0) / len(reconsts)

    return dist['battery'], dist['pcb'], dist['motor']



def create_crosshatch_pattern(size=(64, 64), spacing=8, line_color=(0, 0, 0), bg_color=(255, 255, 255),line_width=1):
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    width, height = size

    # 右上がり線（/）
    for x in range(-height, width, spacing):
        draw.line((x, 0, x + height, height), fill=line_color, width=line_width)

    # 左上がり線（\）
    for x in range(0, width + height, spacing):
        draw.line((x, 0, x - height, height), fill=line_color, width=line_width)

    return img



if __name__ == '__main__':

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    # rect = plt.Rectangle((0, 0), 1, 1, hatch='xx', fill=False, edgecolor='black')
    # ax.add_patch(rect)
    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    # plt.gca().set_aspect('equal')
    # ax.axis('off')
    # plt.savefig("./hatch.png")
    # plt.show()

    # dst = cv2.imread('./hatch.png') # ファイルから読み出し
    # dst = dst[:,:,::-1] # BGR->RGB
    # plt.clf()


    grid_3dim  = 49
    # grid_2dim  = 344
    grid_2dim  = 1024
    # grid_2dim  = 2048
    
    
    # img       = cv2.imread(f'/home/haxhi/workspace/denoising_diffusion_pytorch/scripts/plotter/sheetsander_z_bin.png')
    img       = cv2.imread(f'/home/haxhi/workspace/denoising_diffusion_pytorch/scripts/plotter/sheetsander_x_bin.png')

    rs_cl_img = cv2.resize(img, (grid_2dim, grid_2dim), interpolation=cv2.INTER_NEAREST)

    obj_name        = 'sheetsander'
    tags            = "epsilon_greedy_00"
    # tags            = "no_cond"
    # tags            = "oracle_obs_for_gen_heatmap"



    # root_folder_    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456_clip_ucb_raw_0.5_v13_1/"
    root_folder_    = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456_clip_ucb_raw_0.5_v14_1/"
    # root_folder_   = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_vaeac_a123456_clip_ucb_raw_0.5_v13_2"


    root_folder = f"{root_folder_}/{tags}"
    model_type_folder = get_folder_name(root_folder)




    for j in range(len(model_type_folder)):
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


            # この辺はどうにかしたかったが...
            if obj_name == 'polisher':
                axis = 'x'
                cum_axis = 1 if axis == 'x' else 2
                num_rot = 1
            elif obj_name == 'sheetsander':
                ########################
                # direction = Z version
                #########################
                # axis = 'z'
                # cum_axis = 2
                # num_rot = 2
                ##########################
                ## direction = X version
                ##########################
                axis = 'x'
                cum_axis = 0 if axis == 'x' else 2
                num_rot =  -1

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

                    # 二値製品画像によるマスク処理。上下左右はいい感じにトリミングしている。
                    # rs_each_with_mask = cv2.bitwise_or(cv2.bitwise_not(rs_cl_img), rs_each_jet)[128:-128, 60:-60]
                    # rs_each_with_mask = cv2.bitwise_or(cv2.bitwise_not(rs_cl_img), rs_each_jet)[95:-95, 50:-50] # 344
                    rs_each_with_mask = cv2.bitwise_or(cv2.bitwise_not(rs_cl_img), rs_each_jet)[283:-283, 148:-148] #1024
                    # rs_each_with_mask = cv2.bitwise_or(cv2.bitwise_not(rs_cl_img), rs_each_jet)



                    # 余白を追加
                    # mergin = 4000
                    # rs_each_with_mask = cv2.copyMakeBorder(rs_each_with_mask, mergin, mergin, mergin, mergin, cv2.BORDER_CONSTANT, value=[255, 255, 255])

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
            #         pil_image_save_from_numpy(hatch_image/255.0, f"./hatch_image.png")
            #         rs_each_jet[np.all(rs_each_jet == [0, 0, 225], axis=-1)] = hatch_image[np.all(rs_each_jet == [0, 0, 225], axis=-1)]

            #         # 二値製品画像によるマスク処理。上下左右はいい感じにトリミングしている。
            #         # rs_each_with_mask = cv2.bitwise_or(cv2.bitwise_not(rs_cl_img), rs_each_jet)[128:-128, 60:-60]
            #         # rs_each_with_mask = cv2.bitwise_or(cv2.bitwise_not(rs_cl_img), rs_each_jet)[95:-95, 50:-50] # 344
            #         rs_each_with_mask = cv2.bitwise_or(cv2.bitwise_not(rs_cl_img), rs_each_jet)[283:-283, 148:-148] #1024

            #         save_dir = os.path.normpath(raw_pred_image_dir+"/../../internal_stricture_heatmap2")
            #         create_folder(save_dir)
            #         save_name = f"{save_dir}/{part_name}_{axis}_{label}_.png"
            #         pil_image_save_from_numpy(rs_each_with_mask/255.0, save_name)

