
import numpy as np
from PIL import Image,ImageDraw
import cv2



def pil_image_save_from_numpy(data,save_name):
        pil_image_ = Image.fromarray((data*255).astype(np.uint8))
        # save_name = f"{cond_save_path}/oracle_obs_cast_z_axis{0}.png"
        pil_image_.save(save_name)


# def pil_image_load_to_numpy(data_path,resize=None):
#         pil_image_color = Image.open(data_path)
#         if resize is not None:
#                 pil_image_color = pil_image_color.resize((resize))
#         numpy_image =  np.asarray(pil_image_color)
#         return numpy_image/255.0

def pil_image_load_to_numpy(data_path,resize=None,channel_type=None):
        if channel_type is not None:
            pil_image_color = Image.open(data_path).convert(channel_type)
        elif channel_type is None:
            pil_image_color = Image.open(data_path)
        if resize is not None:
                pil_image_color = pil_image_color.resize((resize))
        numpy_image =  np.asarray(pil_image_color)
        return numpy_image/255.0


def numpy_to_pil(data):
        return Image.fromarray((data*255).astype(np.uint8))

def pil_to_cv2(image):
        ''' PIL型 -> OpenCV型 '''
        new_image = np.array(image, dtype=np.uint8)
        if new_image.ndim == 2:  # モノクロ
                pass
        elif new_image.shape[2] == 3:  # カラー
                new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        elif new_image.shape[2] == 4:  # 透過
                new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
        return new_image

def cv2_ro_pil(image):
        ''' OpenCV型 -> PIL型 '''
        new_image = image.copy()
        if new_image.ndim == 2:  # モノクロ
                pass
        elif new_image.shape[2] == 3:  # カラー
                new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        elif new_image.shape[2] == 4:  # 透過
                new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
        new_image = Image.fromarray(new_image)
        return new_image


def cv2_hsv_mask(image):
        #HSV化
        hsvval=cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

        #HSV値の取得
        h = hsvval[:, :, 0] # H range [0-179]
        s = hsvval[:, :, 1] # S range [0-255]
        v = hsvval[:, :, 2] # V range [0-255]

        #マスク画像の生成(赤色)
        maskimg = np.zeros(h.shape, dtype=np.uint8)
        # maskimg[((h < 20) | (h > 200)) & (s > 170)] = 255  #RED
        # maskimg[((h > 40) & (h < 120)) & (s > 128)] = 255  #GREEN
        # maskimg[((h > 8) & (h < 28+10)) & (s > 184-45) & (s < 184+45) & (v > 184-55) & (v < 184+55)] = 255  #GREEN

        # maskimg[((h > 175-10) & (h < 175+10)) & (s > 5) & (s < 90) & (v > 92-15) & (v < 92+15)] = 25  #GREEN

        # maskimg[((h > 0) & (h < 10)) & (s > 0) & (s < 10) & (v > 92) & (v < 101)] = 25  #GREEN


        # maskimg[((h > 50-10) & (h < 50+10)) & (s > 0) & (s < 100) & (v > 92-10) & (v < 101)] = 255  #GREEN

        maskimg[((h > 100-20) & (h < 100+20)) & (s > 25) & (s < 255) & (v > 200) & (v < 255)] = 255  #GREEN

        return maskimg

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

        # 全てのチャンネルで条件を満たしているピクセルを白、それ以外のピクセルを黒とする
        mask = np.zeros_like(image, dtype=np.uint8)  # 黒のマスクを初期化
        mask[mask_r & mask_g & mask_b] = [1., 1., 1.]  # 全てのチャンネルで条件を満たすピクセルに1を設定
        return mask


def color_mask(image, mask_config):
        lower_bound = mask_config["target_mask_lb"]
        upper_bound = mask_config["target_mask_ub"]

        # RGB画像の各チャンネルを個別に比較
        mask_r = (image[:,:,0] >= lower_bound[0]) & (image[:,:,0] <= upper_bound[0])
        mask_g = (image[:,:,1] >= lower_bound[1]) & (image[:,:,1] <= upper_bound[1])
        mask_b = (image[:,:,2] >= lower_bound[2]) & (image[:,:,2] <= upper_bound[2])
        image[mask_r & mask_g & mask_b] = mask_config["target_mask"] 
        return image
