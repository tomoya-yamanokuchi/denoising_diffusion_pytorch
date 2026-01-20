
import random
import numpy as np
import copy
import torch
from tqdm import tqdm


from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array_multi_type_obj
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils,load_yaml ,save_yaml




print(torch.__version__)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
        class Parser(parser):
            dataset: str = 'Image_diffusion_2D'
            config: str  = 'config.vae'
        grid_config = Parser().parse_args('grid_config')

        import ipdb; ipdb.set_trace()

        import numpy as np
        import os
        from PIL import Image,ImageDraw
        import pyvista as pv


        # dataset_path        =  "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/geom_eval/Boxy_99"
        # mesh_config         = load_yaml(dataset_path+"/generated_configs.yaml")

        # dataset_path    =  "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/geom"
        # dataset_path    =  "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_2/geom"
        # dataset_path    =  "/home/haxhi/workspace/nedo-dismantling-PyBlender/geom_test"
        # dataset_path    =  "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_4_12900k/geom/"
        # dataset_path   = "/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/geom_test_1"
        # dataset_path   = "/home/dev/dataset/denoising_diffusion_pytorch/dataset/dataset_6_13900k/geom_test_1"
        dataset_path   = "/home/dev/workspace/nedo-dismantling-PyBlender/data"

        save_path       = os.path.normpath(f"{dataset_path}/../voxel_images_w_multi_color_v1_")
        create_folder(save_path)


        color_list          = [[0.8,0.8,0.2],[0.2,0.8,0.8],[0.9,0.2,0.2]]
        # component_colors    = {"Component1":color_list[np.random.choice(2)],
        #                         "Component2":color_list[np.random.choice(2)],
        #                         "Component3":color_list[np.random.choice(2)],}


        files_dir = get_folder_name(dataset_path)

        for i in tqdm(range(len(files_dir))):
            mesh_config_path = f"{dataset_path}/{files_dir[i]}/generated_configs.yaml"
            mesh_config      = load_yaml(mesh_config_path)

            # component_colors    = {"Component1":color_list[np.random.choice(2)],
            #                         "Component2":color_list[np.random.choice(2)],
            #                         "Component3":color_list[np.random.choice(2)],}

            # component_colors    = {"Component1":color_list[1],
            #                         "Component2":color_list[2],
            #                         "Component3":color_list[0],}


            component_colors    = {"Component1":color_list[0],
                                    "Component2":color_list[1],
                                    "Component3":color_list[2],}
            # import ipdb;ipdb.set_trace()
            ## load inner boxes
            mesh_components = {}
            for idx, val in enumerate(mesh_config["inner_box"]):
                if  "Component" in val:
                    # tmp_data = f'blend/Boxy_*_cut0_{val}.stl
                    target      = 'Boxy_'  # '' より後ろ（時刻）を抽出したい
                    id          = files_dir[i].find(target)
                    tag         = files_dir[i][id+len(target):]
                    # mesh_path   =  f"{dataset_path}/{files_dir[i]}/blend/Boxy_{tag}_cut0_{val}.stl"
                    mesh_path   =  f"{dataset_path}/{files_dir[i]}/blend/Boxy_{0}_cut0_{val}.stl"
                    mesh        = pv.read(mesh_path)
                    data        = {val:{"mesh":mesh,
                                    "color":component_colors[val]}}
                    print(f"load: mesh_path | {mesh_path}, color | {data[val]['color']}")
                    mesh_components.update(data)
                    mesh_config["inner_box"][val].update({"color":copy.copy(component_colors[val])})
                else:
                    pass

            save_yaml(data=mesh_config,save_path=dataset_path+f"/{files_dir[i]}"+"/generated_configs_w_multi_color.yaml")

            ## voxelize mesh and get sliced image
            # s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),
            #                     "side_length":16}

            s_grid_config = grid_config.s_grid_config



            box_array_handler   = pv_box_array_multi_type_obj(grid_config=s_grid_config)
            _                   = box_array_handler.cast_mesh_to_box_array(mesh_components)
            box_arrays_data     = box_array_handler.get_box_array_data()


            nearby_cells = box_arrays_data.boxes
            colors       = box_arrays_data.colors
            centers      = box_arrays_data.grid_centers
            grid_2dim    = box_arrays_data.grid_2dim_size
            grid_3dim    = box_arrays_data.grid_3dim_size
            batch_image_map = box_array_handler.batch_image_map

            ## save image
            imgs_z = box_array_handler.get_box_color_to_2d_image(box_color=colors,permute="z")
            pil_image = Image.fromarray((imgs_z*255).astype(np.uint8))
            save_name = f"{save_path}/{files_dir[i]}_cast_z_axis{0}.png"
            pil_image.save(save_name)


        # array_2d_reshaped = box_array_handler.get_box_color_to_2d_image(box_color=colors,permute="z")
        # imgs_z = array_2d_reshaped
        # pil_image = Image.fromarray((imgs_z*255).astype(np.uint8))
        # pil_image.save(f"./cast_z_axis{0}_tmp.png")



        import ipdb;ipdb.set_trace()
        exit()


        # 表示
        plotter = pv.Plotter()

        for idx,elements in enumerate(nearby_cells):
            # if np.all(colors[int(elements)] != np.asarray([0,0,0])):
            if np.all(colors[int(elements)] <= np.asarray([0.9,0.9,0.9])):
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
