

import os
import numpy as np
from tqdm import tqdm
import pyvista as pv
from PIL import Image, ImageDraw

from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder






if __name__ == '__main__':

    dataset_path    =  "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/geom"
    save_path       = os.path.normpath(f"{dataset_path}/../voxel_images")
    create_folder(save_path)

    files_dir = get_folder_name(dataset_path)

    for i in tqdm(range(len(files_dir))):
    # for i in tqdm(range(0,5000)):
    # for i in tqdm(range(5000,10000)):
    # for i in tqdm(range(10000,15000)):
    # for i in tqdm(range(15000,len(files_dir))):

        mesh_source = f"{dataset_path}/{files_dir[i]}/blend/"

        ## load inner boxes and merge
        path, f_name = get_path(mesh_source,".stl")
        # mesh1 = pv.read(path[0])
        mesh2 = pv.read(path[1])
        mesh3 = pv.read(path[2])
        mesh4 = pv.read(path[3])
        merged = mesh2.merge(mesh3)
        merged = merged.merge(mesh4)


        ## voxelize mesh and get sliced image
        s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),
                            "side_length":16}

        box_array_handler   = pv_box_array(grid_config=s_grid_config)
        _                   = box_array_handler.cast_mesh_to_box_array(mesh=merged)
        box_arrays_data     = box_array_handler.get_box_array_data()


        nearby_cells = box_arrays_data.boxes
        colors       = box_arrays_data.colors
        centers      = box_arrays_data.grid_centers
        grid_2dim    = box_arrays_data.grid_2dim_size
        grid_3dim    = box_arrays_data.grid_3dim_size


        ## save image
        imgs_z = box_array_handler.get_box_color_to_2d_image(box_color=colors,permute="z")
        pil_image = Image.fromarray((imgs_z*255).astype(np.uint8))
        save_name = f"{save_path}/{files_dir[i]}_cast_z_axis{0}.png"
        pil_image.save(save_name)
