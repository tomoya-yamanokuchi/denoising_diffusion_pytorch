# -*- coding: utf-8 -*-

import os
import yaml
import json
import numpy as np
import torch
from natsort import natsorted
from PIL import Image
import glob
import cv2
import h5py
import pickle5 as pickle

def get_path(d,target_ext = ".pcd"):

    """[summary]
    get file name of arbitrary directory

    Args:
        d (str): [description]. directory name
        target_ext (str, optional): [description]. Defaults to ".pcd".

    Returns:
        path_with_ext (str): [description].file name includes path
        f_name (str): [description].file name
    """
    path_with_ext = []
    f_name = []
    for filename in os.listdir(d):
        _, ext = os.path.splitext(filename.lower())
        if ext == target_ext:
            path_with_ext.append(os.path.join(d, filename))
            f_name.append(_)
    path_with_ext = natsorted(path_with_ext)
    f_name = natsorted(f_name)

    return path_with_ext,f_name

def get_folder_name(root_path):

    files = os.listdir(root_path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(root_path, f))]
    files_dir = natsorted(files_dir)

    return files_dir


def create_folder(directory):

    """[summary]
    create folder if it does not exists
    Args:
        directory ([type]): [description]
    """

    #create save folder
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("create_directory | {}{}{}".format(cstyle.GREEN,directory,cstyle.END))

def load_yaml(file_path):
    try:
        with open(file_path) as file:
            print("load")
            print(cstyle.GREEN+file_path+cstyle.END)
            data = yaml.safe_load(file.read())
    except:
        print("cant load .yaml file")
        print(file_path)

    return data

def save_yaml(data,save_path):
    with open(save_path, 'w') as file:
        yaml.dump(data, file)



def save_json(data,save_path):
    with open(save_path, mode="w") as f:
        json.dump(data, f, indent=8,cls=MyEncoder)


def load_json(load_path):
    with open(load_path) as f:
        data = json.load(f)
    return data

# GIFアニメーションを作成
def create_gif(in_dir, out_filename ,speed):

    path_list = natsorted(glob.glob(os.path.join(*[in_dir, '*']))) # ファイルパスをソートしてリストする
    # path_list.reverse()
    imgs = []                                                   # 画像をappendするための空配列を定義

    # ファイルのフルパスからファイル名と拡張子を抽出
    for i in range(len(path_list)):
        img = Image.open(path_list[i])                          # 画像ファイルを1つずつ開く
        imgs.append(img)                                        # 画像をappendで配列に格納していく

    # appendした画像配列をGIFにする。durationで持続時間、loopでループ数を指定可能。
    imgs[0].save(out_filename,
                 save_all=True, append_images=imgs[1:], optimize=False, duration=speed)

def create_mp4(folderName,out_filename,speed):

    #画像ファイルの一覧を取得
    picList = os.listdir(folderName)
    picList = natsorted(picList)
    img_array = []

    for i in range(len(picList)):
        img = cv2.imread(folderName+picList[i])
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    name = out_filename
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('m','p','4','v'), speed, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

class cstyle():
    BLACK     = '\033[30m'
    RED       = '\033[31m'
    GREEN     = '\033[32m'
    YELLOW    = '\033[33m'
    BLUE      = '\033[34m'
    PURPLE    = '\033[35m'
    CYAN      = '\033[36m'
    WHITE     = '\033[37m'
    END       = '\033[0m'
    BOLD      = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE   = '\033[07m'

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return (obj.numpy()).tolist()
        else:
            return super(MyEncoder, self).default(obj)

class matplot_utils():
    def __init__(self):
        self.marker_dic = {
            ".":"point marker",
            "o":"circle marker",
            "v":"triangle_down marker",
            "^":"triangle_up marker",
            "<":"triangle_left marker",
            ">":"triangle_right marker",
            "1":"tri_down marker",
            "2":"tri_up marker",
            "3":"tri_left marker",
            "4":"tri_right marker",
            "s":"square marker",
            "p":"pentagon marker",
            "*":"star marker",
            "h":"hexagon1 marker",
            "H":"hexagon2 marker",
            "+":"plus marker",
            "x":"x marker",
            "D":"diamond marker",
            "d":"thin_diamond marker",
            "|":"vline marker",
            "_":"hline marker"
            }
    def get_marker_array(self):
        return list(self.marker_dic.keys())


class hdf5_utils():
    def __init__(self):
        pass

    def save(self,dataset,save_path):
        # save_path = os.path.join(self.results_folder, f"diffusion_dataset_{i}.hdf5")
        hf = h5py.File(save_path, 'w')
        dict_group = hf.create_group('dict_data')
        for k, v in dataset.items():
            dict_group[k] = v
        hf.close()

        # self.load()

    def load(self,load_path):
        dict_new = {}
        # load_path = os.path.join(self.results_folder, "diffusion_dataset.hdf5")
        file = h5py.File(load_path, 'r')
        dict_group_load = file['dict_data']
        dict_group_keys = dict_group_load.keys()
        for k in dict_group_keys:
            dict_new[k]= dict_group_load[k][:]

        return dict_new

class pickle_utils():
    def __init__(self):
        pass


    def save(self,dataset,save_path):
        with open(save_path,'wb') as f:
            pickle.dump(dataset, f)

    def load(self,load_path):
        with open(load_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
