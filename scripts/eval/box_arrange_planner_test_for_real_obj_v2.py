
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


        axes = ['x', 'y', 'z']
        trans_min = np.array([random_replacement_config["translation"][axis][0] for axis in axes])
        trans_max = np.array([random_replacement_config["translation"][axis][1] for axis in axes])

        rot_min = np.array([random_replacement_config["rotation"][axis][0] for axis in axes])
        rot_max = np.array([random_replacement_config["rotation"][axis][1] for axis in axes])


        translation     = np.random.uniform(trans_min,trans_max)
        rotation        = np.random.uniform(rot_min,rot_max)
        scale           = np.random.uniform([1,1,1],[1,1,1])


        # import ipdb;ipdb.set_trace()



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



def get_translated_mesh(mesh, translation: dict, rotation: dict):
    # import ipdb;ipdb.set_trace()
    _trans = []
    for k, v in translation.items():
        if len(v) == 1:
            _trans.append(v[0])
        elif len(v) == 2:
            _trans.append(np.random.uniform(v[0], v[1]))

    _mesh = mesh.translate(np.asarray(_trans))

    _rotate = []
    for k, v in rotation.items():
        # if len(v) == 1:
        #     if v[0] != 0.0:
        #         if k == 'x':
        #             _mesh.rotate_x(v[0], point=np.asarray(_trans), inplace=True)
        #         elif k == 'y':
        #             _mesh.rotate_y(v[0], point=np.asarray(_trans), inplace=True)
        #         elif k == 'z':
        #             _mesh.rotate_z(v[0], point=np.asarray(_trans), inplace=True)
        # else:
        #     raise NotImplementedError()
        rot = np.random.uniform(v[0], v[1])
        if k == 'x':
            _mesh.rotate_x(rot, point=np.asarray(_trans), inplace=True)
            _rotate.append(rot)
        elif k == 'y':
            _mesh.rotate_y(rot, point=np.asarray(_trans), inplace=True)
            _rotate.append(rot)
        elif k == 'z':
            _mesh.rotate_z(rot, point=np.asarray(_trans), inplace=True)
            _rotate.append(rot)



    return _mesh,{"translation": np.asarray(_trans),
                  "rotation":np.asarray(_rotate)}




if __name__ == '__main__':




    dataset_path   = "/home/haxhi/dataset/denoising_diffusion_pytorch/dataset/real_models/SheetSander_kwon/"
    config_name    = "config/SheetSander_type5.yaml"

    with open(os.path.join(dataset_path, config_name), encoding='utf-8')as f:
        model_config_= yaml.safe_load(f)

    save_root_folder = dataset_path+"/random_generation_v2/samples/"
    create_folder(save_root_folder)

    eval_data =  get_benchmark_model(dataset_path=dataset_path,model_config=model_config_, for_data_gen = True)
    eval_data.pop("model_name")
    model_config = eval_data
    model_config_original = copy.deepcopy(eval_data)


    # import ipdb;ipdb.set_trace()
    # start_idx, stop_idx = 0, 50
    
    # start_idx, stop_idx = 0, 3000
    # start_idx, stop_idx = 3000, 6000
    # start_idx, stop_idx = 6000, 9000
    # start_idx, stop_idx = 9000, 12000
    # start_idx, stop_idx = 12000, 15000
    start_idx, stop_idx = 15000, 18000



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


    ####################
    # init plot setting
    ####################
    viewer_title = f'Cut_Visualization_{start_idx}_{stop_idx}'
    plotter = BackgroundPlotter(show = True,
            window_size=(1080, 1080), title=viewer_title)
    # plotter.open_movie("./real_model_arrange_test2"+".mp4")


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

        replaced_parts_motor   , coordinate_data  = get_translated_mesh( model_config_original["internal_parts"]["Motor"]["mesh"].copy()  ,model_config_original["internal_parts"]["Motor"]["translation"],   model_config_original["internal_parts"]["Motor"]["rotation"] )
        replaced_parts_pcb     , coordinate_data  = get_translated_mesh( model_config_original["internal_parts"]["PCB"]["mesh"].copy()    ,model_config_original["internal_parts"]["PCB"]["translation"],     model_config_original["internal_parts"]["PCB"]["rotation"] )
        replaced_parts_battery , coordinate_data  = get_translated_mesh( model_config_original["internal_parts"]["Battery"]["mesh"].copy(),model_config_original["internal_parts"]["Battery"]["translation"], model_config_original["internal_parts"]["Battery"]["rotation"] )

        model_config["internal_parts"]["Motor"]["mesh"].points =  replaced_parts_motor.points
        model_config["internal_parts"]["Motor"]["mesh"].faces =  replaced_parts_motor.faces

        model_config["internal_parts"]["PCB"]["mesh"].points =  replaced_parts_pcb.points
        model_config["internal_parts"]["PCB"]["mesh"].faces =  replaced_parts_pcb.faces

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
        # plotter.write_frame()

    exit()
