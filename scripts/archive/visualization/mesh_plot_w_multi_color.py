

import torch
from torch import nn




from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array
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

        val         = "Body"
        mesh_path   = f"{dataset_path}/blend/Boxy_0_cut0_{val}.stl"
        mesh        = pv.read(mesh_path)
        data        = { val:{"mesh" :mesh,
                            "color" :[0.2,0.2,0.2]}
                        }
        mesh_components.update(data)

        eval_data.update({f"{eval_folders[i]}":mesh_components})

    return eval_data






if __name__ == '__main__':


        import numpy as np
        import os
        from PIL import Image,ImageDraw
        import pyvista as pv

        # eval_data_dir               =  "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/geom_eval/"
        # eval_data_dir               =  "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_tmp/geom/"
        # eval_data_dir               =  "/home/haxhi/dataset/nedo_dismantling_dataset/dataset_1/geom_eval/"
        # eval_data_dir               =  "/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test_3"
        eval_data_dir               =  "/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/geom_test_1"
        # eval_data_dir               =  "/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/dataset_6_13900k/geom_test_1"

        # eval_data_dir ="/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test_3/"
        # eval_data_dir ="/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/geom_test_1"
        eval_data_dir ="/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/dataset_6_13900k/geom_test_1/"
        
        
        # 'Object_1':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_4_12900k/geom_test_3/Boxy_0",
        # 'Object_6':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_5_12900k/geom_test_1/Boxy_2",
        # 'Object_8':"/home/user/dataset/denoising_diffusion_pytorch/dataset/dataset_6_13900k/geom_test_1/Boxy_1",
        save_tag = "6_1_(Object_8)_Boxy_1_init_v3"

        # save_tag = "5_1_Boxy_1_init_v2"
        eval_dataset                =   mesh_data(eval_data_dir)
        data                        =   eval_dataset["Boxy_1"]

        # import ipdb;ipdb.set_trace()
        bb = pv.Box(bounds=(-.048, .048, -.048, .048, -.048, .048), level=0, quads=True)
        b2 = pv.Box(bounds=(-.046, .046, -.046, .046, -.046, .046), level=0, quads=True)
        
        # 表示
        plotter = pv.Plotter()
        # plotter.add_mesh(data["Body"]['mesh'], color =[0.1,0.1,0.1], opacity =0.1)
        plotter.add_mesh(bb, color =[164/255,163/255,159/255], opacity =0.08, show_edges=True)
        plotter.add_mesh(b2, color =[164/255,163/255,159/255], opacity =0.08, show_edges=True)
        
        plotter.add_mesh(data["Component1"]['mesh'], color =data["Component1"]['color'], opacity =0.8)
        plotter.add_mesh(data["Component2"]['mesh'], color =data["Component2"]['color'], opacity =0.8)
        plotter.add_mesh(data["Component3"]['mesh'], color =data["Component3"]['color'], opacity =0.8)

        arrow_x = pv.Arrow(
        start=(0, 0, 0), direction=(1, 0, 0), scale=0.05)
        arrow_y = pv.Arrow(
        start=(0, 0, 0), direction=(0, 1, 0), scale=0.05)
        arrow_z = pv.Arrow(
        start=(0, 0, 0), direction=(0, 0, 1), scale=0.05)
        plotter.set_background('white')
        # plotter.camera.position = (0.2, 0.2, 0.15) # polisher #latest
        
        # plotter.add_camera_orientation_widget()
        # plotter.add_mesh(arrow_x, color="r")
        # plotter.add_mesh(arrow_y, color='g')
        # plotter.add_mesh(arrow_z, color='b')
        plotter.save_graphic(f"./{save_tag}.svg")
        plotter.show()
        # plotter.show(screenshot='./airplane.png')