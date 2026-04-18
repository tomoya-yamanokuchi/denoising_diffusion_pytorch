import argparse
import glob
import logging
import os
import numpy as np
import cv2
from tqdm import tqdm


def main(args):

    img_size = int(args.img_size)
    origin_img_size = 512
    pid = int(args.process_id)
    product_name = args.product_name
    axis = args.axis

    logging.basicConfig(
            filename=f'logs/{product_name}_{pid}.log',
            level=logging.INFO,
            format='%(asctime)s : %(levelname)s : %(filename)s - %(message)s')
    logging.info(args)

    tgt_folder_name = os.path.join("./dataset/real_kaden", product_name, f"{img_size}x{img_size}", axis)
    os.makedirs(tgt_folder_name, exist_ok=True)

    num_batch = 1000
    # source_dir_name = os.path.join(args.dataset_dir, product_name, axis)
    source_dir_name = os.path.join(args.dataset_dir)
    file_list = glob.glob(f'{source_dir_name}/*')[pid*num_batch:(pid+1)*num_batch]

    tgt_img_size = (img_size, img_size)
    _tmp_mask = np.zeros((origin_img_size, origin_img_size, 3), dtype=np.uint8)

    for each_file in tqdm(file_list):
        img = cv2.imread(each_file)
        package = 255 * np.ones_like(img, dtype=np.uint8)
        p_mask = np.all(img != np.array([255, 255, 255]), axis=-1)
        package[p_mask] = [229, 229, 229]
        rs_package = cv2.resize(package, tgt_img_size, interpolation=cv2.INTER_AREA)

        # red (motor)
        # r_mask = 255 * np.all(img == np.array([51, 51, 229]), axis=-1).astype(np.uint8)
        r_mask = 255 * np.all(img == np.array([255*0.2, 255*0.2, 255*0.9]).astype(np.uint8), axis=-1).astype(np.uint8)
        rs_r_mask = cv2.resize(r_mask, tgt_img_size, interpolation=cv2.INTER_AREA)
        _th_r = np.median(np.unique(rs_r_mask))
        rs_package[rs_r_mask > _th_r] = [0, 0, 255]

        # green (pcb)
        # y_mask = 255 * np.all(img == np.array([51, 204, 204]), axis=-1).astype(np.uint8)
        y_mask = 255 * np.all(img == np.array([255*0.2, 255*0.8, 255*0.8]).astype(np.uint8), axis=-1).astype(np.uint8)
        rs_y_mask = cv2.resize(y_mask, tgt_img_size, interpolation=cv2.INTER_AREA)
        _th_y = np.median(np.unique(rs_y_mask))
        rs_package[rs_y_mask > _th_y] = [0, 255, 0]

        # blue (battery)
        b_mask = 255 * np.all(img == np.array([255*0.8, 255*0.8, 255*0.2]), axis=-1).astype(np.uint8)
        rs_b_mask = cv2.resize(b_mask, tgt_img_size, interpolation=cv2.INTER_AREA)
        _th_b = np.median(np.unique(rs_b_mask))
        rs_package[rs_b_mask > _th_b] = [255, 0, 0]

        cv2.imwrite(tgt_folder_name+f'/{each_file.split("/")[-1]}', rs_package)

    return True


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--process_id', required=True)
        parser.add_argument('--dataset_dir', required=True)
        parser.add_argument('--product_name', required=True)
        parser.add_argument('--axis', required=True)
        parser.add_argument('--img_size', required=True)
        args = parser.parse_args()
        main(args)
    except Exception as e:
        logging.exception(e)
