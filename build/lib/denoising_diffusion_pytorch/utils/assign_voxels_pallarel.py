import os
import numpy as np
from tqdm import tqdm
import copy
import pyvista as pv
from PIL import Image,ImageDraw
import ray






@ray.remote
def assign_voxels(k, cubes, grid_centers):
    cube_copy = cubes.translate(grid_centers[k])  # 立方体を中心座標に移動
    return {str(k):cube_copy}