import os
from natsort import natsorted
import shutil


def get_folder_name(root_path):

    files = os.listdir(root_path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(root_path, f))]

    return files_dir

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
if __name__ == '__main__':

    dataset_dir = "./datasets/flower-image/train/"
    data_dir    = "./datasets/flower-image/train_concat/"
    folder_list = get_folder_name(dataset_dir)

    for i in range(len(folder_list)):
        file_path,file_name = get_path(dataset_dir+folder_list[i],".jpeg")
        for j in range(len(file_path)):
            save_name = folder_list[i].replace(" ","_")
            shutil.move(file_path[j], data_dir+save_name+"_"+file_name[j]+".jpg")