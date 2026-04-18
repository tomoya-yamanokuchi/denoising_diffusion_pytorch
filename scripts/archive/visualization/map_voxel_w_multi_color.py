

import torch
from torch import nn



from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array_multi_type_obj
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils,load_yaml
# from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils,load_yaml


print(torch.__version__)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"




def mesh_data(data_source_dir):

    eval_folders = get_folder_name(data_source_dir)[:3]
    # eval_folders = get_folder_name(data_source_dir)[1:2]
    # import ipdb;ipdb.set_trace()

    eval_data = {}
    for i in range(len(eval_folders)):

        dataset_path                =  f"{data_source_dir}"+f"/{eval_folders[i]}"
        mesh_config_path            =  f"{dataset_path}/generated_configs_w_multi_color.yaml"
        mesh_config                 = load_yaml(mesh_config_path)

        mesh_components = {}
        for  idx, val in enumerate(mesh_config["inner_box"]):
            if  "Component" in val:
                mesh_path   = f"{dataset_path}/blend/Boxy_0_cut0_{val}.stl"
                mesh        = pv.read(mesh_path)
                data        = { val:{"mesh" :mesh,
                                    "color" :mesh_config["inner_box"][val]['color']}
                                }
                print(f"load: mesh_path | {mesh_path}, color | {data[val]['color']}")
                mesh_components.update(data)
            else:
                pass
        eval_data.update({f"{eval_folders[i]}":mesh_components})

    return eval_data






if __name__ == '__main__':


        import numpy as np
        import os
        from PIL import Image,ImageDraw
        import pyvista as pv

        eval_data_dir               =  "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/geom_eval/"
        # eval_data_dir               =  "/home/haxhi/dataset/nedo_dismantling_dataset/dataset_1/geom_eval/"
        eval_data_dir               ='/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/dataset_2_12900k/geom_test/'
        eval_dataset                =   mesh_data(eval_data_dir)
        mesh_components             =   eval_dataset["Boxy_2"]

        # import ipdb;ipdb.set_trace()
        # mesh_components["Component1"]["color"] = [1,1,1]
        # mesh_components["Component2"]["color"] = [1,1,1]
        # mesh_components.pop("Component1")
        # mesh_components.pop("Component2")


        ## voxelize mesh and get sliced image
        s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),
                            "side_length":16}


        box_array_handler   = pv_box_array_multi_type_obj(grid_config=s_grid_config)
        _                   = box_array_handler.cast_mesh_to_box_array(mesh_components)
        box_arrays_data     = box_array_handler.get_box_array_data()


        nearby_cells = box_arrays_data.boxes
        colors       = box_arrays_data.colors
        centers      = box_arrays_data.grid_centers
        grid_2dim    = box_arrays_data.grid_2dim_size
        grid_3dim    = box_arrays_data.grid_3dim_size
        batch_image_map = box_array_handler.batch_image_map


        # 表示
        plotter = pv.Plotter()

        for idx,elements in enumerate(nearby_cells):
            # if np.all(colors[int(elements)] != np.asarray([0,0,0])):
            if np.all(colors[int(elements)] <= np.asarray([0.9,0.9,0.9])):
            # if np.all(colors[int(elements)] == np.asarray([0.8,0.8,0.2])):
                plotter.add_mesh(nearby_cells[elements], color = colors[int(elements)] ,opacity = 0.9 , show_edges=True)
            else:
                plotter.add_mesh(nearby_cells[elements],style='wireframe' ,opacity =1e-5 , show_edges=True, edge_opacity= 0.01)

            # plotter.add_mesh(nearby_cells[elements], color = colors[int(elements)] ,opacity = 0.1 , show_edges=True)

        plotter.add_points(centers, render_points_as_spheres=True, color = [0,0,0], opacity = 0.5, )


        # plotter.add_mesh(sgrid,scalars=colors, rgb=True,opacity = 0.5,show_edges=True)
        # plotter.add_mesh(merged,color =[0.1,0.8,0.8], opacity =0.4)

        for idx,val in enumerate(mesh_components):
            plotter.add_mesh(mesh_components[val]["mesh"],color =mesh_components[val]["color"], opacity =0.9)


        arrow_x = pv.Arrow(
        start=(0, 0, 0), direction=(1, 0, 0), scale=0.05)
        arrow_y = pv.Arrow(
        start=(0, 0, 0), direction=(0, 1, 0), scale=0.05)
        arrow_z = pv.Arrow(
        start=(0, 0, 0), direction=(0, 0, 1), scale=0.05)
        plotter.set_background('white')
        # plotter.add_camera_orientation_widget()
        plotter.add_mesh(arrow_x, color="r")
        plotter.add_mesh(arrow_y, color='g')
        plotter.add_mesh(arrow_z, color='b')
        plotter.show()