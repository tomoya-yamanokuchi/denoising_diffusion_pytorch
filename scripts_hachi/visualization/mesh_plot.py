

import torch
from torch import nn

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.normalization import LimitsNormalizer
from denoising_diffusion_pytorch.arrays import to_torch,to_device,to_np



from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils



print(torch.__version__)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':


        import numpy as np
        import os
        from PIL import Image,ImageDraw
        import pyvista as pv


        dataset_path    =  "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/geom_eval/Boxy_99"
        mesh_source     = f"{dataset_path}/blend/"





        ## load inner boxes and merge
        path, f_name = get_path(mesh_source,".stl")
        mesh1 = pv.read(path[0])
        mesh2 = pv.read(path[1])
        mesh3 = pv.read(path[2])
        mesh4 = pv.read(path[3])
        merged = mesh2.merge(mesh3)
        merged = merged.merge(mesh4)



        # 表示
        plotter = pv.Plotter()



        # plotter.add_mesh(sgrid,scalars=colors, rgb=True,opacity = 0.5,show_edges=True)
        plotter.add_mesh(mesh1, color =[0.1,0.1,0.1], opacity =0.1)
        plotter.add_mesh(merged, color =[0.8,0.8,0.1], opacity =0.8)

        arrow_x = pv.Arrow(
        start=(0, 0, 0), direction=(1, 0, 0), scale=0.08)
        arrow_y = pv.Arrow(
        start=(0, 0, 0), direction=(0, 1, 0), scale=0.08)
        arrow_z = pv.Arrow(
        start=(0, 0, 0), direction=(0, 0, 1), scale=0.08)
        plotter.set_background('white')
        # plotter.add_camera_orientation_widget()
        plotter.add_mesh(arrow_x, color="r")
        plotter.add_mesh(arrow_y, color='g')
        plotter.add_mesh(arrow_z, color='b')
        plotter.show()