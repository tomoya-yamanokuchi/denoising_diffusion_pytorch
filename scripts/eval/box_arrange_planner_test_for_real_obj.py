
import os
import yaml


import numpy as np
from tqdm import tqdm
import pyvista as pv
import copy


from scipy.spatial.transform import Rotation
import trimesh

from pyvistaqt import BackgroundPlotter

from denoising_diffusion_pytorch.utils.benchmark_model_utils import get_benchmark_model
from denoising_diffusion_pytorch.utils.os_utils import create_folder,save_yaml



def pv_mesh_to_trimesh(data):
    vertices = data.points  # 頂点座標
    faces = data.faces.reshape(-1, 4)[:, 1:]  # 三角形インデックスに変換
    tm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return tm_mesh

def get_random_transformation_matrix(translation,rotation):
    rot         = Rotation.from_euler('xyz', rotation, degrees=True)
    rot_matrix  = rot.as_matrix()
    btm         = np.zeros((1,3))
    tmp_matrix  = np.vstack((rot_matrix,btm))
    trans       = np.asarray([[translation[0]],[translation[1]],[translation[2]],[1]])
    homo_mat    = np.hstack((tmp_matrix,trans))
    return homo_mat


def get_scaled_mesh(mesh,scale):
    origin = mesh.center
    mesh2  = mesh.translate(np.asarray(origin)*-1.0)
    mesh2  = mesh2.scale(scale, inplace=False)
    mesh3  = mesh2.translate(np.asarray(origin))
    return mesh3

def get_rotated_mesh(mesh, rotation):
    origin = mesh.center
    mesh2  = mesh.translate(np.asarray(origin)*-1.0)
    homo_matrix = get_random_transformation_matrix([0.0,0.0,0.0],rotation=rotation)
    mesh3 = mesh2.transform(homo_matrix)
    mesh4 = mesh3.translate(np.asarray(origin))
    return mesh4


def get_random_replacement(outer_shape, inner_shape, random_replacement_config):

    collision_flag = 1
    while collision_flag == 1:
        inner_shape_orig = inner_shape.copy()

        trans_x = np.random.uniform(-0.03, 0.03)
        trans_y = np.random.uniform(-0.03, 0.03)
        trans_z = np.random.uniform(-0.03, 0.03)
        rot_x   = np.random.uniform(-20, 20)
        rot_y   = np.random.uniform(-20, 20)
        rot_z   = np.random.uniform(-10, 10)
        scale   = np.random.uniform(0.5, 1.1)

        translation     = np.asarray([trans_x,trans_y,trans_z])
        rotation        = np.asarray([rot_x,rot_y,rot_z])


        translation     = np.random.uniform(random_replacement_config["translation"]["min"],random_replacement_config["translation"]["max"])
        rotation        = np.random.uniform(random_replacement_config["rotation"]["min"],random_replacement_config["rotation"]["max"])
        scale           = np.random.uniform(random_replacement_config["scale"]["min"],random_replacement_config["scale"]["max"])

        translated_mesh = inner_shape_orig.translate(translation)
        rotated_mesh    = get_rotated_mesh(translated_mesh, rotation)
        # mesh_b          = get_scaled_mesh(rotated_mesh,scale=[scale,scale,scale])
        # mesh_b          = get_scaled_mesh(rotated_mesh,scale=[scale[0],scale[1],scale[2]])
        mesh_b          = get_scaled_mesh(rotated_mesh,scale=[scale[0],scale[0],scale[0]])
        collision_points, ncol = outer_shape.collision(mesh_b,cell_tolerance = 1)

        # mesh_b_t      = pv_mesh_to_trimesh(mesh_b)
        # outer_shape_t = pv_mesh_to_trimesh(outer_shape)
        # inside = outer_shape_t.contains(mesh_b_t.vertices)

        print(f"translation: {translation}, rotation: {rotation}, scale: {scale}")

        del inner_shape_orig


        # if collision_points["ContactCells"].shape[0]==0 and any(inside):
        if collision_points["ContactCells"].shape[0]==0:
            collision_flag = 0
        else:
            collision_flag = 1
    return mesh_b


if __name__ == '__main__':




    # dataset_path   = "/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/real_models/SheetSander/"
    # dataset_path   = "/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/real_models/Polisher/"
    dataset_path   = "/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/real_models/PowerCutter/"


    with open(os.path.join(dataset_path,"generated_configs.yaml"), encoding='utf-8')as f:
        model_config= yaml.safe_load(f)

    save_root_folder = dataset_path+"/random_generation_v2/samples/"
    create_folder(save_root_folder)

    eval_data =  get_benchmark_model(dataset_path=dataset_path,model_config=model_config, for_data_gen = True)
    eval_data.pop("model_name")
    model_config = eval_data
    model_config_original = copy.deepcopy(eval_data)

    # ## sheet sander
    # replacement_config = {"internal_parts":{"Battery":{"translation"    :{  "min":[ 0.0,0.0, 0.0],
    #                                                                         "max":[ 0.0,0.0, 0.0]},
    #                                                     "rotation"      :{  "min":[0.0,0.0,0.0],
    #                                                                         "max":[0.0,0.0,0.0]},
    #                                                     "scale"         :{  "min":[0.5,0.5,0.5],
    #                                                                         "max":[1.1,1.1,1.1]}},
    #                                         "Motor1":{"translation"     :{  "min":[-0.03,-0.03,-0.03],
    #                                                                         "max":[ 0.03, 0.03, 0.03]},
    #                                                     "rotation"      :{  "min":[-20.0,-20.0,-20.0],
    #                                                                         "max":[ 20.0, 20.0, 20.0]},
    #                                                     "scale"         :{  "min":[0.5,0.5,0.5],
    #                                                                         "max":[1.1,1.1,1.1]}},
    #                                         "PCB1":{"translation"       :{  "min":[-0.03,-0.03,-0.03],
    #                                                                         "max":[ 0.03, 0.03, 0.03]},
    #                                                     "rotation"      :{  "min":[-20.0,-20.0,-20.0],
    #                                                                         "max":[ 20.0, 20.0, 20.0]},
    #                                                     "scale"         :{  "min":[0.5,0.5,0.5],
    #                                                                         "max":[1.1,1.1,1.1]}},
    #                                         }}


    # ## Polisher
    # replacement_config = {"internal_parts":{"Battery":{"translation"    :{  "min":[ 0.0,0.0,-0.01],
    #                                                                         "max":[ 0.0,0.03, 0.0]},
    #                                                     "rotation"      :{  "min":[0.0,0.0,0.0],
    #                                                                         "max":[0.0,0.0,0.0]},
    #                                                     "scale"         :{  "min":[0.6,0.6,0.6],
    #                                                                         "max":[1.1,1.1,1.1]}},
    #                                         "Motor1":{"translation"     :{  "min":[-0.01,-0.02,-0.02],
    #                                                                         "max":[ 0.01, 0.02, 0.02]},
    #                                                     "rotation"      :{  "min":[-20.0,-20.0,-20.0],
    #                                                                         "max":[ 20.0, 20.0, 20.0]},
    #                                                     "scale"         :{  "min":[0.6,0.6,0.6],
    #                                                                         "max":[1.1,1.1,1.1]}},
    #                                         "PCB1":{"translation"       :{  "min":[-0.01,-0.02,-0.02],
    #                                                                         "max":[ 0.01, 0.04, 0.04]},
    #                                                     "rotation"      :{  "min":[-10.0,-20.0,-20.0],
    #                                                                         "max":[ 10.0, 20.0, 20.0]},
    #                                                     "scale"         :{  "min":[0.6,0.6,0.6],
    #                                                                         "max":[1.1,1.1,1.1]}},
    #                                         }}


    ## Power cutter
    replacement_config = {"internal_parts":{"Battery":{"translation"    :{  "min":[ -0.01,-0.03,-0.01],
                                                                            "max":[  0.01, 0.03, 0.0]},
                                                        "rotation"      :{  "min":[0.0,0.0,0.0],
                                                                            "max":[0.0,0.0,0.0]},
                                                        "scale"         :{  "min":[0.6,0.6,0.6],
                                                                            "max":[1.1,1.1,1.1]}},
                                            "Motor1":{"translation"     :{  "min":[-0.10,-0.02,-0.10],
                                                                            "max":[ 0.01, 0.02, 0.02]},
                                                        "rotation"      :{  "min":[-20.0,-20.0,-20.0],
                                                                            "max":[ 20.0, 20.0, 20.0]},
                                                        "scale"         :{  "min":[0.6,0.6,0.6],
                                                                            "max":[1.1,1.1,1.1]}},
                                            "PCB1":{"translation"       :{  "min":[-0.03,-0.02,-0.02],
                                                                            "max":[ 0.03, 0.02, 0.02]},
                                                        "rotation"      :{  "min":[-10.0,-20.0,-20.0],
                                                                            "max":[ 10.0, 20.0, 20.0]},
                                                        "scale"         :{  "min":[0.6,0.6,0.6],
                                                                            "max":[1.1,1.1,1.1]}},
                                            }}
    save_yaml(replacement_config,save_path=save_root_folder+"/../replacement_config.yaml")

    # import ipdb;ipdb.set_trace()
    start_idx, stop_idx = 0, 50
    # start_idx, stop_idx = 0, 1000
    # start_idx, stop_idx = 1000, 2000
    # start_idx, stop_idx = 2000, 3000
    # start_idx, stop_idx = 3000, 4000
    # start_idx, stop_idx = 4000, 5000
    # start_idx, stop_idx = 5000, 6000
    # start_idx, stop_idx = 6000, 7000
    # start_idx, stop_idx = 7000, 8000
    # start_idx, stop_idx = 8000, 9000
    # start_idx, stop_idx = 9000, 10000
    # start_idx, stop_idx = 10000, 11000
    # start_idx, stop_idx = 11000, 12000
    # start_idx, stop_idx = 12000, 13000
    # start_idx, stop_idx = 13000, 14000
    # start_idx, stop_idx = 14000, 15000
    # start_idx, stop_idx = 15000, 16000
    # start_idx, stop_idx = 16000, 17000
    # start_idx, stop_idx = 17000, 18000
    # start_idx, stop_idx = 18000, 19000
    # start_idx, stop_idx = 19000, 20000
    # start_idx, stop_idx = 4844, 4848


    ####################
    # init plot setting
    ####################
    viewer_title = f'Cut_Visualization_{start_idx}_{stop_idx}'
    plotter = BackgroundPlotter(show = True,
            window_size=(1080, 1080), title=viewer_title)
    plotter.open_movie("./real_model_arrange_test2"+".mp4")


    arrow_x = pv.Arrow(
    start=(0, 0, 0), direction=(1, 0, 0), scale=0.1)
    arrow_y = pv.Arrow(
    start=(0, 0, 0), direction=(0, 1, 0), scale=0.1)
    arrow_z = pv.Arrow(
    start=(0, 0, 0), direction=(0, 0, 1), scale=0.1)
    plotter.add_mesh(arrow_x, color="r")
    plotter.add_mesh(arrow_y, color='g')
    plotter.add_mesh(arrow_z, color='b')

    for idx, val in enumerate(model_config):
        for idx, vals in enumerate(model_config[val]):
            mesh       = model_config[val][vals]["mesh"]
            color      = model_config[val][vals]["color"]
            if val == "outer_parts":
                plotter.add_mesh(mesh, opacity=0.5, color = color,show_edges=False,)
            else:
                plotter.add_mesh(mesh, opacity=0.8, color = color,show_edges=False,)


    # for i in tqdm(range(10)):
    for i in tqdm(range(start_idx, stop_idx)):
        replaced_parts_motor     = get_random_replacement( model_config_original["outer_parts"]["Body1"]["mesh"].copy(),model_config_original["internal_parts"]["Motor1"]["mesh"].copy(),random_replacement_config=replacement_config["internal_parts"]["Motor1"] )
        replaced_parts_pcb       = get_random_replacement( model_config_original["outer_parts"]["Body1"]["mesh"].copy(),model_config_original["internal_parts"]["PCB1"]["mesh"].copy(),random_replacement_config=replacement_config["internal_parts"]["PCB1"] )
        replaced_parts_battery   = get_random_replacement( model_config_original["outer_parts"]["Body1"]["mesh"].copy(),model_config_original["internal_parts"]["Battery"]["mesh"].copy(),random_replacement_config=replacement_config["internal_parts"]["Battery"] )


        model_config["internal_parts"]["Motor1"]["mesh"].points =  replaced_parts_motor.points
        model_config["internal_parts"]["Motor1"]["mesh"].faces =  replaced_parts_motor.faces

        model_config["internal_parts"]["PCB1"]["mesh"].points =  replaced_parts_pcb.points
        model_config["internal_parts"]["PCB1"]["mesh"].faces =  replaced_parts_pcb.faces

        model_config["internal_parts"]["Battery"]["mesh"].points =  replaced_parts_battery.points
        model_config["internal_parts"]["Battery"]["mesh"].faces =  replaced_parts_battery.faces

        # import ipdb;ipdb.set_trace()
        save_folder = os.path.join(save_root_folder,f"samples_{i}","models")
        create_folder(save_folder)

        for idx,val in enumerate(model_config["internal_parts"]):
            save_name = os.path.join(save_folder,f"./{val}.stl")
            model_config["internal_parts"][val]["mesh"].save(save_name)

        for idx,val in enumerate(model_config["outer_parts"]):
            save_name = os.path.join(save_folder,f"./{val}.stl")
            model_config["outer_parts"][val]["mesh"].save(save_name)
        plotter.render()
        plotter.app.processEvents()
        plotter.write_frame()

    exit()



    plotter.add_mesh(inner_box_1, opacity=0.8, color = [0.2,0.8,0,8],show_edges=True,)
    plotter.add_mesh(inner_box_2, opacity=0.8, color = [0.8,0.2,0,2],show_edges=True,)
    plotter.add_mesh(inner_box_3, opacity=0.8, color = [0.8,0.8,0,2],show_edges=True,)


    plotter.set_background('white')
    plotter.render()
    plotter.write_frame()
    plotter.app.processEvents()

    for j in range(100):

        # 直方体をランダムに配置
        # containers = random_placement((min_box_size, max_box_size), num_boxes, container_size)
        containers = random_placement_const_mode_1(box_arrange_config, container_size)


        for i, container in enumerate(containers):
            print(f"Box {i+1}: position={container[0]}, size={container[1]}")
            box_size = container[1]/2.0
            box_pos  = container[0]

            outer_bounds = [-box_size[0], box_size[0], -box_size[1], box_size[1], -box_size[2], box_size[2]]
            inner_cube_ = pv.Box(outer_bounds)
            inner_cube  = inner_cube_.translate(box_pos)
            # inner_box_list.append(inner_cube)
            inner_box_list[i].points = inner_cube.points
            inner_box_list[i].faces = inner_cube.faces
        plotter.render()
        plotter.app.processEvents()
        plotter.write_frame()
        import time;time.sleep(.05)
        # plotter.show()

