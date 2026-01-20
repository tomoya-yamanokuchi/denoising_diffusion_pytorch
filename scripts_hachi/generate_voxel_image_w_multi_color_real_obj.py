
import random
import numpy as np
import copy
import torch
from tqdm import tqdm
import ray
import time
import pickle
import yaml

import numpy as np
import os
from PIL import Image,ImageDraw
import imageio
import pyvista as pv


from denoising_diffusion_pytorch.utils.setup import Parser as parser
from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array_multi_type_obj
from denoising_diffusion_pytorch.utils.os_utils import get_path ,get_folder_name,create_folder,pickle_utils,load_yaml ,save_yaml
from denoising_diffusion_pytorch.utils.benchmark_model_utils import get_benchmark_model



# print(torch.__version__)
# print(torch.cuda.is_available())
# device = "cuda" if torch.cuda.is_available() else "cpu"
# ray.init(num_cpus=64)

def get_model(dataset_path, model_config):

        mesh_components = {}
        for idx, val in enumerate(model_config["parts_name"]):
                print(val)
                mesh_path   =  os.path.normpath(dataset_path+model_config["parts_name"][val]["path"])
                mesh        = pv.read(mesh_path)
                mesh_color  = model_config["parts_name"][val]["color"]
                data        = {val:{    "mesh":mesh,
                                        "color":mesh_color}}
                print(f"load: mesh_path | {mesh_path}, color | {data[val]['color']}")
                mesh_components.update(data)
        return mesh_components






if __name__ == '__main__':



        # dim = 16
        # dim = 25
        ### dim = 36
        dim = 49
        # dim = 64


        # s_grid_config = grid_config.s_grid_config
        # s_grid_config = {"bounds":(-0.3,0.3,-0.3,0.3,-0.3,0.3), 'side_length':25}
        s_grid_config = {"bounds":(-0.3,0.3,-0.3,0.3,-0.3,0.3), 'side_length':dim}
        # s_grid_config = {"bounds":(-0.4,0.4,-0.4,0.4,-0.4,0.4), 'side_length':dim}


        file_path = f'./my_dict{dim}.pkl'

        # box_array_handler_tmp   = pv_box_array_multi_type_obj(grid_config=s_grid_config)
        # aa = box_array_handler_tmp._create_box_array()
        # with open(file_path, 'wb') as f:
        #         pickle.dump(aa, f)

        if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                        load_nearby_cells = pickle.load(f)

        save_tag       = "SheetSander_kwon"
        # save_tag       = "Polisher"
        # save_tag       = "PowerCutter"

        # dataset_path   = f"/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/real_models/{save_tag}/random_generation_v1/samples/"
        # dataset_path   = "/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/real_models/{save_tag}/random_generation_v1/samples/"
        # dataset_path   = "/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/real_models/{save_tag}/random_generation_v1/samples/"

        dataset_path   = f"/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/real_models/{save_tag}/random_generation_v2/samples/"
        
        # with open(os.path.normpath(os.path.join(dataset_path,"../../generated_configs.yaml")), encoding='utf-8')as f:
        with open(os.path.normpath(os.path.join(dataset_path,"../../generated_configs_for_voxel_image.yaml")), encoding='utf-8')as f:
                model_config= yaml.safe_load(f)

        imgs_slice_save_path = os.path.normpath(dataset_path+"../"+f"cast_gif_{dim}")
        create_folder(imgs_slice_save_path)
        folder_names = get_folder_name(dataset_path)

        # for i in tqdm(range(0,10)):
        # for i in tqdm(range(0, 1000)):
        # for i in tqdm(range(1000, 2000)):
        # for i in tqdm(range(2000, 3000)):
        # for i in tqdm(range(3000, 4000)):
        # for i in tqdm(range(4000, 5000)):
        # for i in tqdm(range(5000, 6000)):
        # for i in tqdm(range(6000, 7000)):
        # for i in tqdm(range(7000, 8000)):
        # for i in tqdm(range(8000, 9000)):
        # for i in tqdm(range(9000, 10000)):
        # for i in tqdm(range(10000, 11000)):
        # for i in tqdm(range(11000, 12000)):
        # for i in tqdm(range(12000, 13000)):
        # for i in tqdm(range(13000, 14000)):
        # for i in tqdm(range(14000, 15000)):
        # for i in tqdm(range(15000, 16000)):
        # for i in tqdm(range(16000, 17000)):
        for i in tqdm(range(17000, 18000)):
        # for i in tqdm(range(18000, 19000)):
        # for i in tqdm(range(19000, 20000)):
        # for i in tqdm(range(19999, 20000)):
                data_dir = os.path.normpath(dataset_path+"/"+folder_names[i])+"/"
                mesh_components =  get_benchmark_model(dataset_path=data_dir,model_config=model_config)

                # current_time = time.time()
                box_array_handler   = pv_box_array_multi_type_obj(grid_config=s_grid_config,pre_near_by_cells=load_nearby_cells)
                _                   = box_array_handler.cast_mesh_to_box_array(mesh_components)
                box_arrays_data     = box_array_handler.get_box_array_data()
                # print(f"cal_time = {time.time()-current_time}")

                nearby_cells = box_arrays_data.boxes
                colors       = box_arrays_data.colors
                centers      = box_arrays_data.grid_centers
                grid_2dim    = box_arrays_data.grid_2dim_size
                grid_3dim    = box_arrays_data.grid_3dim_size
                batch_image_map = box_array_handler.batch_image_map

                ## save image
                imgs_z = box_array_handler.get_box_color_to_2d_image(box_color=colors,permute="z")
                # imgs_z_slice = box_array_handler.get_2d_image_to_mini_batch_image(imgs_z,permute = "z")
                # imageio.mimsave(imgs_slice_save_path+f"/{save_tag}_cast_z_axis_{i}_{dim}.gif", (imgs_z_slice*255).astype(np.uint8), fps=10)


                pil_image = Image.fromarray((imgs_z*255).astype(np.uint8))
                save_name = f"{data_dir}/cast_z_axis_{dim}.png"
                pil_image.save(save_name)

                # imgs_z = box_array_handler.get_box_color_to_2d_image(box_color=colors,permute="x")
                # pil_image = Image.fromarray((imgs_z*255).astype(np.uint8))
                # save_name = f"{data_dir}/cast_x_axis_{dim}.png"
                # pil_image.save(save_name)

                # imgs_z = box_array_handler.get_box_color_to_2d_image(box_color=colors,permute="y")
                # pil_image = Image.fromarray((imgs_z*255).astype(np.uint8))
                # save_name = f"{data_dir}/cast_y_axis_{dim}.png"
                # pil_image.save(save_name)




        import ipdb;ipdb.set_trace()
        exit()
        # import ipdb;ipdb.set_trace()

        points  = centers
        colors  = box_arrays_data.colors
        alpha_values = np.zeros(centers.shape[0])+0.99
        alpha_values[np.all(colors==np.asarray([1,1,1]),axis=1)]=0.0
        alpha_values[np.all(colors==np.asarray([0.9,0.9,0.9]),axis=1)]=0.5
        # alpha_values[np.all(colors==np.asarray([0.8,0.8,0.2]),axis=1)]=0.6
        # alpha_values[np.all(colors!=np.asarray([0.9,0.9,0.9]))and np.all(colors!=np.asarray([1,1,1]),axis=1)]=1.0


        cloud = pv.PolyData(points)
        cloud["rgba"] = np.column_stack([colors*255, alpha_values * 255]).astype(np.uint8)
        plotter = pv.Plotter()
        plotter.add_mesh(cloud, scalars="rgba", rgb=True, point_size=20)
        plotter.add_points(points[np.where(np.all(colors!=np.asarray([1,1,1]),axis=1))],color = [0,0,0], point_size=30, opacity = 0.08, )
        # plotter.add_mesh(cloud, scalars="rgba", point_size=20)
        # plotter.add_mesh(sgrid,scalars=colors, rgb=True,opacity = 0.5,show_edges=True)

        # plotter.add_mesh(cloud)


        arrow_x = pv.Arrow(
        start=(0, 0, 0), direction=(1, 0, 0), scale=0.08)
        arrow_y = pv.Arrow(
        start=(0, 0, 0), direction=(0, 1, 0), scale=0.08)
        arrow_z = pv.Arrow(
        start=(0, 0, 0), direction=(0, 0, 1), scale=0.08)
        plotter.set_background('white')
        plotter.add_mesh(arrow_x, color="r")
        plotter.add_mesh(arrow_y, color='g')
        plotter.add_mesh(arrow_z, color='b')

        plotter.show()

        import ipdb;ipdb.set_trace()

        # plotter = pv.Plotter()

        # for idx,elements in tqdm(enumerate(nearby_cells),total=len(nearby_cells)):
        #     # if np.all(colors[int(elements)] != np.asarray([0,0,0])):
        #     if np.all(colors[int(elements)] <= np.asarray([0.9,0.9,0.9])):
        #         plotter.add_mesh(nearby_cells[elements], color = colors[int(elements)] ,opacity = 0.9 , show_edges=True)
        #     else:
        #         plotter.add_mesh(nearby_cells[elements],style='wireframe' ,opacity =1e-5 , show_edges=True, edge_opacity= 0.01)

        #     # plotter.add_mesh(nearby_cells[elements], color = colors[int(elements)] ,opacity = 0.1 , show_edges=True)

        # # plotter.add_points(centers, render_points_as_spheres=True, color = [0,0,0], opacity = 0.5, )


        # # plotter.add_mesh(sgrid,scalars=colors, rgb=True,opacity = 0.5,show_edges=True)
        # # plotter.add_mesh(merged,color =[0.1,0.8,0.8], opacity =0.4)
        # # for idx,val in enumerate(mesh_components):
        # #     plotter.add_mesh(mesh_components[val]["mesh"],color =mesh_components[val]["color"], opacity =0.9)


        # arrow_x = pv.Arrow(
        # start=(0, 0, 0), direction=(1, 0, 0), scale=0.08)
        # arrow_y = pv.Arrow(
        # start=(0, 0, 0), direction=(0, 1, 0), scale=0.08)
        # arrow_z = pv.Arrow(
        # start=(0, 0, 0), direction=(0, 0, 1), scale=0.08)
        # plotter.set_background('white')
        # # plotter.add_camera_orientation_widget()
        # plotter.add_mesh(arrow_x, color="r")
        # plotter.add_mesh(arrow_y, color='g')
        # plotter.add_mesh(arrow_z, color='b')
        # plotter.show()

        # exit()
        #         # import ipdb;ipdb.set_trace()
        # bb = pv.Box(bounds=(-.048, .048, -.048, .048, -.048, .048), level=0, quads=True)
        # b2 = pv.Box(bounds=(-.046, .046, -.046, .046, -.046, .046), level=0, quads=True)

        # # 表示
        # plotter = pv.Plotter()

        # for idx, val in enumerate(model_config["parts_name"]):
        #         print(val)
        #         mesh_path   =  os.path.normpath(dataset_path+model_config["parts_name"][val]["path"])
        #         mesh        = pv.read(mesh_path)
        #         mesh_color  = model_config["parts_name"][val]["color"]
        #         plotter.add_mesh(mesh,color=mesh_color, opacity = 0.8)
        # #         data        = {val:{    "mesh":mesh,
        # #                                 "color":mesh_color}}
        # #         print(f"load: mesh_path | {mesh_path}, color | {data[val]['color']}")
        # #         mesh_components.update(data)
        # # # plotter.add_mesh(data["Body"]['mesh'], color =[0.1,0.1,0.1], opacity =0.1)
        # # plotter.add_mesh(bb, color =[0.1,0.1,0.1], opacity =0.1, show_edges=True)
        # # plotter.add_mesh(b2, color =[0.1,0.1,0.1], opacity =0.1, show_edges=True)
        
        # # plotter.add_mesh(data["Component1"]['mesh'], color =data["Component1"]['color'], opacity =0.8)
        # # plotter.add_mesh(data["Component2"]['mesh'], color =data["Component2"]['color'], opacity =0.8)
        # # plotter.add_mesh(data["Component3"]['mesh'], color =data["Component3"]['color'], opacity =0.8)

        # arrow_x = pv.Arrow(
        # start=(0, 0, 0), direction=(1, 0, 0), scale=0.05)
        # arrow_y = pv.Arrow(
        # start=(0, 0, 0), direction=(0, 1, 0), scale=0.05)
        # arrow_z = pv.Arrow(
        # start=(0, 0, 0), direction=(0, 0, 1), scale=0.05)
        # plotter.set_background('white')
        # # plotter.add_camera_orientation_widget()
        # plotter.add_mesh(arrow_x, color="r")
        # plotter.add_mesh(arrow_y, color='g')
        # plotter.add_mesh(arrow_z, color='b')
        # # plotter.save_graphic(f"./{save_tag}.svg")
        # plotter.show()
        # # plotter.show()