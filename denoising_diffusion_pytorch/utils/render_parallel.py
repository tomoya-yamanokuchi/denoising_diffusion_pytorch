
import os
import numpy as np
from tqdm import tqdm
import copy
import pyvista as pv
from PIL import Image,ImageDraw
import ray
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array

pv.global_theme.allow_empty_mesh = True


def get_random_transformation_matrix(translation,rotation):
    rot         = Rotation.from_euler('xyz', rotation, degrees=True)
    rot_matrix  = rot.as_matrix()
    btm         = np.zeros((1,3))
    tmp_matrix  = np.vstack((rot_matrix,btm))
    trans       = np.asarray([[translation[0]],[translation[1]],[translation[2]],[1]])
    homo_mat    = np.hstack((tmp_matrix,trans))
    return homo_mat

def get_rotated_mesh(mesh, rotation):
    origin = mesh.center
    mesh2  = mesh.translate(np.asarray(origin)*-1.0)
    homo_matrix = get_random_transformation_matrix([0.0,0.0,0.0],rotation=rotation)
    mesh3 = mesh2.transform(homo_matrix)
    mesh4 = mesh3.translate(np.asarray(origin))
    return mesh4

def get_cutting_plane(s_grid_config, action, action_table):

        action_axis = action_table[action]['axis']
        if action_axis == 'z':
            action_pos_candidate = np.linspace(s_grid_config["bounds"][0],s_grid_config["bounds"][1],s_grid_config["side_length"])
            action_pos = action_pos_candidate[action_table[action]['loc']]
            cutting_plane_translation = np.asarray([0,0,action_pos])
            cutting_plane_rotation    = np.asarray([0,0,0])
        elif action_axis == 'y':
            action_pos_candidate = np.linspace(s_grid_config["bounds"][0],s_grid_config["bounds"][1],s_grid_config["side_length"])
            action_pos = action_pos_candidate[action_table[action]['loc']]
            cutting_plane_translation = np.asarray([0,action_pos,0])
            cutting_plane_rotation    = np.asarray([90,0,0])
        elif action_axis  == 'x':
            action_pos_candidate = np.linspace(s_grid_config["bounds"][0],s_grid_config["bounds"][1],s_grid_config["side_length"])
            action_pos = action_pos_candidate[action_table[action]['loc']]
            cutting_plane_translation = np.asarray([action_pos,0,0])
            cutting_plane_rotation    = np.asarray([0,90,0])

        cutting_plane_ = pv.Box(bounds=(s_grid_config["bounds"][0]-0.01,
                                        s_grid_config["bounds"][1]+0.01,
                                        s_grid_config["bounds"][2]-0.01,
                                        s_grid_config["bounds"][3]+0.01,
                                        -0.00004,
                                        +0.00004,
                                ))
        cutting_plane_tranced = cutting_plane_.translate(cutting_plane_translation)
        cutting_plane = get_rotated_mesh(cutting_plane_tranced, cutting_plane_rotation)
        
        return cutting_plane

def get_cutting_plane_with_cost_map(k, s_grid_config,action_table, cost_map):

    cutting_plane_with_cost_map_one_step={}

    for i in range(len(action_table)):
        cutting_plane_with_cost_map_one_step[i] = {}
        axis_name   = action_table[i]["axis"]
        loc_idx     = action_table[i]["loc"]

        # cutting_plane_with_cost_map_one_step[i]["cost"] = 20.0
        ###########################################
        ## Set the cost for the volume containing the target internal component on each cut surface
        ###########################################
        # cutting_plane_with_cost_map_one_step[i]["cost"] = cutting_plane_with_cost_map_one_step[i]["cost"]+cost_map[k]["raw_cost"]['cost_b'][f"{axis_name}_axis"].mean(0)[loc_idx]+25.0 ## raw_cost ver
        cutting_plane_with_cost_map_one_step[i]["cost"] = cost_map[k]["raw_cost"]['cost_b'][f"{axis_name}_axis"].mean(0)[loc_idx]+25.0 ## raw_cost ver
        # cutting_plane_with_cost_map_one_step[i]["cost"] = np.clip(cost_map[k]["raw_cost"]['cost_b'][f"{axis_name}_axis"],0,1).mean(0)[loc_idx] ## clip ver

        ###########################################
        ## get all cutting plane candidates
        #############################################
        cutting_plane_with_cost_map_one_step[i]["mesh"] = get_cutting_plane(s_grid_config, i, action_table)

    ##################################################
    ### set the cutting cost around the cutting volume
    ##################################################
    slice_candidate = cost_map[k]["slice_candidate"]
    for axis_candidates in slice_candidate.values():
        for value in axis_candidates:
            # cutting_plane_with_cost_map_one_step[value]["cost"]=cutting_plane_with_cost_map_one_step[value]["cost"]-10.0  ##raw_cost ver
            cutting_plane_with_cost_map_one_step[value]["cost"]=cutting_plane_with_cost_map_one_step[value]["cost"]+10.0  ##raw_cost ver
            # cutting_plane_with_cost_map_one_step[value]["cost"]=cutting_plane_with_cost_map_one_step[value]["cost"]+0.2 #clip ver

    ###############################################
    ### set the cutting cost of the executed action to 0
    ##############################################
    cutting_pos  = cost_map[k]['slice_range'][-1]
    cutting_plane_with_cost_map_one_step[cutting_pos]["cost"]=0.0


    return cutting_plane_with_cost_map_one_step


def normalize_cost_and_assign_color(data: dict, cmap) -> dict:
    """
    cost を 0〜1 に正規化し、cmap に基づくカラー情報を 'color' キーとして追加する。

    Parameters:
        data (dict): 各キーに {'cost': float} を持つ辞書。
        cmap (matplotlib colormap): カラーマップ（例: plt.cm.viridis）

    Returns:
        dict: 各エントリに 'color' が追加された辞書。
    """
    # 1. コスト値抽出
    costs = np.array([entry["cost"] for entry in data.values()])
    
    # 2. 正規化
    min_cost = 0.0
    max_cost = costs.max()
    if max_cost - min_cost == 0:
        normalized_costs = np.zeros_like(costs)
    else:
        normalized_costs = (costs - min_cost) / (max_cost - min_cost)

    normalized_costs = 1.0-normalized_costs  # コストが高いほど色が濃くなるように反転

    # 3. カラーマップ適用して 'color' を追加
    updated_data = {}
    for i, (key, entry) in enumerate(data.items()):
        rgba = cmap(normalized_costs[i])
        updated_data[key] = entry.copy()  # 元の辞書を壊さないようにコピー
        updated_data[key]["color"] = rgba  # RGBAのまま追加（必要なら HEX に変換可）
        # もし HEX 色が欲しい場合はこちらも追加:
        # updated_data[key]["color_hex"] = mcolors.to_hex(rgba)

    return updated_data


@ray.remote
def one_step_voxel_render(k, s_grid_config,sample_images,save_path):
        # tmp_mesh = pv.read("/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/geom_eval/Boxy_99/blend/Boxy_0_cut0_Component3.stl")


        tmp_mesh = pv.Box(bounds=(s_grid_config["bounds"][0]+0.02,
                                  s_grid_config["bounds"][1]-0.02,
                                  s_grid_config["bounds"][2]+0.02,
                                  s_grid_config["bounds"][3]-0.02,
                                  s_grid_config["bounds"][4]+0.02,
                                  s_grid_config["bounds"][5]-0.02,
                                ))




        box_array_handler   = pv_box_array(grid_config=s_grid_config)
        _                   = box_array_handler.cast_mesh_to_box_array(mesh=copy.copy(tmp_mesh))
        box_arrays_data     = box_array_handler.get_box_array_data()
        nearby_cells        = box_arrays_data.boxes
        colors              = box_arrays_data.colors
        centers             = box_arrays_data.grid_centers
        grid_2dim           = box_arrays_data.grid_2dim_size
        grid_3dim           = box_arrays_data.grid_3dim_size
        batch_image_map     = box_array_handler.batch_image_map



        plotter = pv.Plotter(window_size=(512, 512),off_screen = True)
        step_image = sample_images[k]/255.0
        step_image = step_image.clip(0,1,step_image)


        updated_colors = box_array_handler.cast_2d_image_to_box_color(image=step_image,permute="z")


        # for idx,elements in tqdm(enumerate(nearby_cells)):
        #     if np.all(updated_colors[int(elements)] != np.asarray([1,1,1])):
        #         plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.9 , show_edges=True)
        #     else:
        #         # plotter.add_mesh(self.nearby_cells[elements],style='wireframe' ,opacity =1e-5 , show_edges=True, edge_opacity= 0.01)
        #         plotter.add_mesh(nearby_cells[elements],style='wireframe' ,opacity =1e-5 , show_edges=True,)

        for idx,elements in enumerate(nearby_cells):
            # if np.all(colors[int(elements)] != np.asarray([0,0,0])):
            if np.all(updated_colors[int(elements)] <= np.asarray([0.9,0.9,0.9])):
                plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.9 , show_edges=True)
                # pass
            else:
                plotter.add_mesh(nearby_cells[elements],style='wireframe' ,opacity =1e-5 , show_edges=True, edge_opacity= 0.01)




        plotter.add_points(centers, render_points_as_spheres=True, color = [0,0,0], opacity = 0.1, )

        arrow_x = pv.Arrow(
        start=(0, 0, 0), direction=(1, 0, 0), scale=0.08)
        arrow_y = pv.Arrow(
        start=(0, 0, 0), direction=(0, 1, 0), scale=0.08)
        arrow_z = pv.Arrow(
        start=(0, 0, 0), direction=(0, 0, 1), scale=0.08)
        # plotter.add_camera_orientation_widget()
        # plotter.add_mesh(arrow_x, color="r")

        plotter.add_mesh(arrow_x, color="r")
        plotter.add_mesh(arrow_y, color='g')
        plotter.add_mesh(arrow_z, color='b')



        data = plotter.save_graphic(save_path+f'/screenshot_{k}.eps')
        # imgs =plotter.screenshot(save_path+f'/screenshot_{i}.png')
        imgs =plotter.screenshot()
        pil_image = Image.fromarray(np.asarray(imgs))
        # rendered_imgs.append(pil_image)
        plotter.close()
        
        return pil_image

@ray.remote
def one_step_voxel_render_for_cutting_process(k, s_grid_config,sample_images,action,action_table,save_path):


        tmp_mesh = pv.Box(bounds=(s_grid_config["bounds"][0],
                                  s_grid_config["bounds"][1],
                                  s_grid_config["bounds"][2],
                                  s_grid_config["bounds"][3],
                                  s_grid_config["bounds"][4],
                                  s_grid_config["bounds"][5],
                                ))

        action_axis = action_table[action[k]]['axis']
        if action_axis == 'z':
            action_pos_candidate = np.linspace(s_grid_config["bounds"][0],s_grid_config["bounds"][1],s_grid_config["side_length"])
            action_pos = action_pos_candidate[action_table[action[k]]['loc']]
            cutting_plane_translation = np.asarray([0,0,action_pos])
            cutting_plane_rotation    = np.asarray([0,0,0])
        elif action_axis == 'y':
            action_pos_candidate = np.linspace(s_grid_config["bounds"][0],s_grid_config["bounds"][1],s_grid_config["side_length"])
            action_pos = action_pos_candidate[action_table[action[k]]['loc']]
            cutting_plane_translation = np.asarray([0,action_pos,0])
            cutting_plane_rotation    = np.asarray([90,0,0])
        elif action_axis  == 'x':
            action_pos_candidate = np.linspace(s_grid_config["bounds"][0],s_grid_config["bounds"][1],s_grid_config["side_length"])
            action_pos = action_pos_candidate[action_table[action[k]]['loc']]
            cutting_plane_translation = np.asarray([action_pos,0,0])
            cutting_plane_rotation    = np.asarray([0,90,0])

        cutting_plane_ = pv.Box(bounds=(s_grid_config["bounds"][0]-0.01,
                                  s_grid_config["bounds"][1]+0.01,
                                  s_grid_config["bounds"][2]-0.01,
                                  s_grid_config["bounds"][3]+0.01,
                                  -0.0001,
                                  0.0001,
                                ))
        cutting_plane_tranced = cutting_plane_.translate(cutting_plane_translation)
        cutting_plane = get_rotated_mesh(cutting_plane_tranced, cutting_plane_rotation)


        box_array_handler   = pv_box_array(grid_config=s_grid_config)
        _                   = box_array_handler.cast_mesh_to_box_array(mesh=copy.copy(tmp_mesh))
        box_arrays_data     = box_array_handler.get_box_array_data()
        nearby_cells        = box_arrays_data.boxes
        colors              = box_arrays_data.colors
        centers             = box_arrays_data.grid_centers
        grid_2dim           = box_arrays_data.grid_2dim_size
        grid_3dim           = box_arrays_data.grid_3dim_size
        batch_image_map     = box_array_handler.batch_image_map



        plotter = pv.Plotter(window_size=(800, 800),off_screen = True)
        step_image = sample_images[k]/255.0
        step_image = step_image.clip(0,1,step_image)



        updated_colors = box_array_handler.cast_2d_image_to_box_color(image=step_image,permute="z")

        for idx,elements in enumerate(nearby_cells):
            # if np.all(updated_colors[int(elements)] == np.asarray([0.1,0.1,0.1])):
            # if np.all(updated_colors[int(elements)] >= np.asarray([0.1,0.7,0.7])) & np.all((updated_colors[int(elements)] <  np.asarray([0.3,0.9,0.9]))):
            #     plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.5 , show_edges=True)

            # elif np.all(updated_colors[int(elements)] >= np.asarray([0.7,0.7,0.1])) & np.all((updated_colors[int(elements)] <  np.asarray([0.9,0.9,0.3]))):
            #     plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.5 , show_edges=True)

            # elif np.all(updated_colors[int(elements)] >= np.asarray([0.7,0.1,0.1])) & np.all((updated_colors[int(elements)] <  np.asarray([0.9,0.3,0.3]))):
            #     plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.5 , show_edges=True)

            # # plotter.add_mesh(nearby_cells[elements],style='wireframe' ,opacity =1e-5 , show_edges=True, edge_opacity= 0.01)
            # elif np.all(updated_colors[int(elements)] >= np.asarray([0.0,0.0,0.0])) & np.all((updated_colors[int(elements)] <  np.asarray([.3,.3,.3]))):
            #     plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.1 , show_edges=True)

            ##################################################################################################################
            ### normal visualization mode
            ###################################################################################################################
            if np.all(updated_colors[int(elements)] >= np.asarray([0.0,0.0,0.0])) & np.all((updated_colors[int(elements)] <  np.asarray([.5,.5,.5]))): # black
                # plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.1 , show_edges=True)
                pass
            else:
                if np.all(updated_colors[int(elements)] >= np.asarray([0.5,0.5,0.5])) & np.all((updated_colors[int(elements)] <  np.asarray([1.3,1.3,1.3]))): # white
                    opacity = 0.1 # default 0.1 emsamble 0.02
                    # opacity = 1.0
                    # plotter.add_mesh(nearby_cells[elements],style='wireframe' ,opacity =opacity , show_edges=True, edge_opacity= 0.01)
                    plotter.add_mesh(nearby_cells[elements],style='wireframe',opacity =0.001 , show_edges=True, edge_opacity= 0.01, color=[0.8,0.8,0.8])
                    plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = opacity , show_edges=True)
                else:
                    opacity = 0.9
                    plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = opacity , show_edges=True)
            plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 1e-10 , show_edges=True)


            # ##################################################################################################################
            # ### ensemble image visualization mode
            # ###################################################################################################################
            # if np.all(updated_colors[int(elements)] >= np.asarray([0.0,0.0,0.0])) & np.all((updated_colors[int(elements)] <  np.asarray([.5,.5,.5]))): # black
            #     # plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.1 , show_edges=True)
            #     pass
            # else:
            #     if np.all(updated_colors[int(elements)] >= np.asarray([0.95,0.95,0.95])) & np.all((updated_colors[int(elements)] <  np.asarray([1.3,1.3,1.3]))): # white
            #         opacity = 0.01
            #         # opacity = 1.0
            #         # plotter.add_mesh(nearby_cells[elements],style='wireframe' ,opacity =opacity , show_edges=True, edge_opacity= 0.01)
            #         plotter.add_mesh(nearby_cells[elements],style='wireframe',opacity =0.001 , show_edges=True, edge_opacity= 0.01, color=[0.8,0.8,0.8])
            #         plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = opacity , show_edges=True)
            #     else:
            #         opacity = 0.9
            #         plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = opacity , show_edges=True)
            # # plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 1e-10 , show_edges=True)



            # ##################################################################################################################
            # ### cost map visualization mode
            # ###################################################################################################################
            # # if np.all(updated_colors[int(elements)] >= np.asarray([0.0,0.0,0.0])) & np.all((updated_colors[int(elements)] <  np.asarray([.1,.1,.1]))): # black
            # #     pass
            # # else:
            # #     if np.all(updated_colors[int(elements)] >= np.asarray([0.99,0.99,0.99])) & np.all((updated_colors[int(elements)] <  np.asarray([1.3,1.3,1.3]))): # white
            # #         opacity = 0.1
            # #         plotter.add_mesh(nearby_cells[elements],style='wireframe',opacity =0.001 , show_edges=True, edge_opacity= 0.01, color=[0.8,0.8,0.8])
            # #         plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = opacity , show_edges=True)
            # #     else:
            # #         opacity = 0.1
            # #         plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = opacity , show_edges=True)
            # opacity = 0.5
            # plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = opacity , show_edges=True)


        plotter.add_points(centers, render_points_as_spheres=True, color = [0,0,0], opacity = 1e-10)

        if (k)%2== 0:
            plotter.add_mesh(cutting_plane, color = (226/255.0, 220/255.0, 222/255.0),opacity = 0.8 ,show_edges = False, diffuse=1.0)
            # plotter.add_mesh(cutting_plane, color = (240/255, 245/255, 37/255.0),opacity = 0.8 ,show_edges = False)
            # plotter.add_mesh(cutting_plane, color = (240/255, 230/255, 144/255.0),opacity = 0.8 ,show_edges = False)
            # plotter.add_mesh(cutting_plane, color = (255/255, 182/256, 193/255),opacity = 0.8 ,show_edges = False) #default 0.8
        else:
            plotter.add_mesh(cutting_plane, color = (0.7, 0.7, 0.0),opacity = 0.0 ,show_edges = False)
        # plotter.add_mesh(cutting_plane, color = (0.7, 0.7, 0.0),opacity = 1.0 ,show_edges = False)





        arrow_x = pv.Arrow(
        start=(0, 0, 0), direction=(1, 0, 0), scale=0.08)
        arrow_y = pv.Arrow(
        start=(0, 0, 0), direction=(0, 1, 0), scale=0.08)
        arrow_z = pv.Arrow(
        start=(0, 0, 0), direction=(0, 0, 1), scale=0.08)
        # plotter.add_camera_orientation_widget()
        # plotter.add_mesh(arrow_x, color="r")

        # plotter.add_mesh(arrow_x, color="r")
        # plotter.add_mesh(arrow_y, color='g')
        # plotter.add_mesh(arrow_z, color='b')
        # plotter.set_background('black', top='white')
        cube = pv.Cube(center=(s_grid_config["bounds"][0], s_grid_config["bounds"][0], s_grid_config["bounds"][0]))
        plotter.set_focus(cube.center)
        parallel_projection = True
        if parallel_projection is True:
            plotter.camera.parallel_projection = True
            plotter.camera.parallel_scale = 0.1
        # plotter.camera.position = (.2, .2, .2) # 64 ver default
        # plotter.camera.position = (0.25, 0.25, 0.25) # 64 ver cutting process
        plotter.camera.position = (0.1+0.2, 0.35+0.2, 0.1+0.2) # for parallel projection
        plotter.camera.up = (0.0, 0.0, 1.0)
        data = plotter.save_graphic(save_path+f'/screenshot_{k}.eps')
        # imgs =plotter.screenshot(save_path+f'/screenshot_{i}.png')
        imgs =plotter.screenshot()
        pil_image = Image.fromarray(np.asarray(imgs))
        # rendered_imgs.append(pil_image)
        plotter.close()

        # os.remove(save_path+f'/screenshot_{k}.eps')

        return pil_image
    
    
@ray.remote
def one_step_voxel_render_for_cutting_process_w_cost_map(k, s_grid_config,sample_images,action,action_table,cost_map,save_path):

        tmp_mesh = pv.Box(bounds=(s_grid_config["bounds"][0],
                                  s_grid_config["bounds"][1],
                                  s_grid_config["bounds"][2],
                                  s_grid_config["bounds"][3],
                                  s_grid_config["bounds"][4],
                                  s_grid_config["bounds"][5],
                                ))

        action_axis = action_table[action[k]]['axis']
        if action_axis == 'z':
            action_pos_candidate = np.linspace(s_grid_config["bounds"][0],s_grid_config["bounds"][1],s_grid_config["side_length"])
            action_pos = action_pos_candidate[action_table[action[k]]['loc']]
            cutting_plane_translation = np.asarray([0,0,action_pos])
            cutting_plane_rotation    = np.asarray([0,0,0])
        elif action_axis == 'y':
            action_pos_candidate = np.linspace(s_grid_config["bounds"][0],s_grid_config["bounds"][1],s_grid_config["side_length"])
            action_pos = action_pos_candidate[action_table[action[k]]['loc']]
            cutting_plane_translation = np.asarray([0,action_pos,0])
            cutting_plane_rotation    = np.asarray([90,0,0])
        elif action_axis  == 'x':
            action_pos_candidate = np.linspace(s_grid_config["bounds"][0],s_grid_config["bounds"][1],s_grid_config["side_length"])
            action_pos = action_pos_candidate[action_table[action[k]]['loc']]
            cutting_plane_translation = np.asarray([action_pos,0,0])
            cutting_plane_rotation    = np.asarray([0,90,0])

        cutting_plane_ = pv.Box(bounds=(s_grid_config["bounds"][0]-0.01,
                                  s_grid_config["bounds"][1]+0.01,
                                  s_grid_config["bounds"][2]-0.01,
                                  s_grid_config["bounds"][3]+0.01,
                                  -0.0001,
                                  0.0001,
                                ))
        cutting_plane_tranced = cutting_plane_.translate(cutting_plane_translation)
        cutting_plane = get_rotated_mesh(cutting_plane_tranced, cutting_plane_rotation)



        box_array_handler   = pv_box_array(grid_config=s_grid_config)
        _                   = box_array_handler.cast_mesh_to_box_array(mesh=copy.copy(tmp_mesh))
        box_arrays_data     = box_array_handler.get_box_array_data()
        nearby_cells        = box_arrays_data.boxes
        colors              = box_arrays_data.colors
        centers             = box_arrays_data.grid_centers
        grid_2dim           = box_arrays_data.grid_2dim_size
        grid_3dim           = box_arrays_data.grid_3dim_size
        batch_image_map     = box_array_handler.batch_image_map



        plotter = pv.Plotter(window_size=(800, 800),off_screen = True)
        step_image = sample_images[k]/255.0
        step_image = step_image.clip(0,1,step_image)



        updated_colors = box_array_handler.cast_2d_image_to_box_color(image=step_image,permute="z")

        for idx,elements in enumerate(nearby_cells):
            # if np.all(updated_colors[int(elements)] == np.asarray([0.1,0.1,0.1])):
            # if np.all(updated_colors[int(elements)] >= np.asarray([0.1,0.7,0.7])) & np.all((updated_colors[int(elements)] <  np.asarray([0.3,0.9,0.9]))):
            #     plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.5 , show_edges=True)

            # elif np.all(updated_colors[int(elements)] >= np.asarray([0.7,0.7,0.1])) & np.all((updated_colors[int(elements)] <  np.asarray([0.9,0.9,0.3]))):
            #     plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.5 , show_edges=True)

            # elif np.all(updated_colors[int(elements)] >= np.asarray([0.7,0.1,0.1])) & np.all((updated_colors[int(elements)] <  np.asarray([0.9,0.3,0.3]))):
            #     plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.5 , show_edges=True)

            # # plotter.add_mesh(nearby_cells[elements],style='wireframe' ,opacity =1e-5 , show_edges=True, edge_opacity= 0.01)
            # elif np.all(updated_colors[int(elements)] >= np.asarray([0.0,0.0,0.0])) & np.all((updated_colors[int(elements)] <  np.asarray([.3,.3,.3]))):
            #     plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.1 , show_edges=True)

            ##################################################################################################################
            ### normal visualization mode
            ###################################################################################################################
            # if np.all(updated_colors[int(elements)] >= np.asarray([0.0,0.0,0.0])) & np.all((updated_colors[int(elements)] <  np.asarray([.5,.5,.5]))): # black
            #     # plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.1 , show_edges=True)
            #     pass
            # else:
            #     if np.all(updated_colors[int(elements)] >= np.asarray([0.5,0.5,0.5])) & np.all((updated_colors[int(elements)] <  np.asarray([1.3,1.3,1.3]))): # white
            #         opacity = 0.1
            #         # opacity = 1.0
            #         # plotter.add_mesh(nearby_cells[elements],style='wireframe' ,opacity =opacity , show_edges=True, edge_opacity= 0.01)
            #         plotter.add_mesh(nearby_cells[elements],style='wireframe',opacity =0.001 , show_edges=True, edge_opacity= 0.01, color=[0.8,0.8,0.8])
            #         plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = opacity , show_edges=True)
            #     else:
            #         opacity = 0.9
            #         plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = opacity , show_edges=True)
            # plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 1e-10 , show_edges=True)


            ##################################################################################################################
            ### ensemble image visualization mode
            ###################################################################################################################
            if np.all(updated_colors[int(elements)] >= np.asarray([0.0,0.0,0.0])) & np.all((updated_colors[int(elements)] <  np.asarray([.5,.5,.5]))): # black
                # plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.1 , show_edges=True)
                pass
            else:
                if np.all(updated_colors[int(elements)] >= np.asarray([0.95,0.95,0.95])) & np.all((updated_colors[int(elements)] <  np.asarray([1.3,1.3,1.3]))): # white
                    opacity = 0.01
                    # opacity = 1.0
                    # plotter.add_mesh(nearby_cells[elements],style='wireframe' ,opacity =opacity , show_edges=True, edge_opacity= 0.01)
                    plotter.add_mesh(nearby_cells[elements],style='wireframe',opacity =0.001 , show_edges=True, edge_opacity= 0.01, color=[0.8,0.8,0.8])
                    plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = opacity , show_edges=True)
                else:
                    opacity = 0.9
                    plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = opacity , show_edges=True)
            # plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 1e-10 , show_edges=True)
            # pass


            # ##################################################################################################################
            # ### cost map visualization mode
            # ###################################################################################################################
            # # if np.all(updated_colors[int(elements)] >= np.asarray([0.0,0.0,0.0])) & np.all((updated_colors[int(elements)] <  np.asarray([.1,.1,.1]))): # black
            # #     pass
            # # else:
            # #     if np.all(updated_colors[int(elements)] >= np.asarray([0.99,0.99,0.99])) & np.all((updated_colors[int(elements)] <  np.asarray([1.3,1.3,1.3]))): # white
            # #         opacity = 0.1
            # #         plotter.add_mesh(nearby_cells[elements],style='wireframe',opacity =0.001 , show_edges=True, edge_opacity= 0.01, color=[0.8,0.8,0.8])
            # #         plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = opacity , show_edges=True)
            # #     else:
            # #         opacity = 0.1
            # #         plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = opacity , show_edges=True)
            # opacity = 0.5
            # plotter.add_mesh(nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = opacity , show_edges=True)


        plotter.add_points(centers, render_points_as_spheres=True, color = [0,0,0], opacity = 1e-10)


        # if (k)%2== 0:
        #     # plotter.add_mesh(cutting_plane, color = (0.7, 0.7, 0.0),opacity = 0.8 ,show_edges = False)
        #     plotter.add_mesh(cutting_plane, color = (255/255, 182/256, 193/255),opacity = 0.8 ,show_edges = False) #default 0.8
        # else:
        #     plotter.add_mesh(cutting_plane, color = (0.7, 0.7, 0.0),opacity = 0.0 ,show_edges = False)
        # # plotter.add_mesh(cutting_plane, color = (0.7, 0.7, 0.0),opacity = 1.0 ,show_edges = False)

        if k>=3 and k%2==0 and k<=action.shape[0]-2: # 最初と最後の2ステップはコストマップを表示しない
            aa = int(k/2-2)
            cutting_plane_with_cost_map = normalize_cost_and_assign_color(get_cutting_plane_with_cost_map(aa,s_grid_config,action_table, cost_map),plt.cm.plasma)
            # cutting_plane_with_cost_map = normalize_cost_and_assign_color(get_cutting_plane_with_cost_map(aa,s_grid_config,action_table, cost_map),plt.cm.cool)

            all_slice_candidates = []
            for axis_candidates in cost_map[aa]["slice_candidate"].values():
                all_slice_candidates.extend(axis_candidates)


            parity = action[k] % 2  # 偶数なら 0、奇数なら 1 最終的に決定される切断面の位置actionを必ず表示するために可変化
            for key in cutting_plane_with_cost_map:
                color = cutting_plane_with_cost_map[key]["color"][:3]
                # plotter.add_mesh(
                #     cutting_plane_with_cost_map[key]["mesh"],
                #     color=color,
                #     opacity=0.3,
                #     show_edges=True,
                #     line_width=2
                # )
                if key % 2 == parity and key != action[k] and key not in all_slice_candidates and cutting_plane_with_cost_map[key]["cost"] >= 46.0:
                    plotter.add_mesh(
                        cutting_plane_with_cost_map[key]["mesh"],
                        color=color,
                        opacity=0.45,
                        show_edges=True,
                        line_width=2
                    )
                elif key % 2 == parity and key != action[k]:
                        plotter.add_mesh(
                        cutting_plane_with_cost_map[key]["mesh"],
                        color=color,
                        opacity=0.2,
                        show_edges=True,
                        line_width=2
                    )
                elif key == action[k]:
                    plotter.add_mesh(
                        cutting_plane_with_cost_map[key]["mesh"],
                        color=color,
                        opacity=0.5,
                        show_edges=True,
                        line_width=5
                    )
                # # else key % 2 == parity and key != action[k] and key in all_slice_candidates:\
                # else:
                #     plotter.add_mesh(
                #         cutting_plane_with_cost_map[key]["mesh"],
                #         color=color,
                #         opacity=0.2,
                #         show_edges=True,
                #         line_width=2
                #     )
                        

                # plotter.add_mesh(cutting_plane_with_cost_map[action[k]]["mesh"], color = (255/255, 182/256, 193/255),opacity = 0.8 ,show_edges = False) #default 0.8



        arrow_x = pv.Arrow(
        start=(0, 0, 0), direction=(1, 0, 0), scale=0.08)
        arrow_y = pv.Arrow(
        start=(0, 0, 0), direction=(0, 1, 0), scale=0.08)
        arrow_z = pv.Arrow(
        start=(0, 0, 0), direction=(0, 0, 1), scale=0.08)
        # plotter.add_camera_orientation_widget()
        # plotter.add_mesh(arrow_x, color="r")

        # plotter.add_mesh(arrow_x, color="r")
        # plotter.add_mesh(arrow_y, color='g')
        # plotter.add_mesh(arrow_z, color='b')
        # plotter.set_background('black', top='white')
        cube = pv.Cube(center=(s_grid_config["bounds"][0], s_grid_config["bounds"][0], s_grid_config["bounds"][0]))
        plotter.set_focus(cube.center)
        parallel_projection = True
        if parallel_projection is True:
            plotter.camera.parallel_projection = True
            plotter.camera.parallel_scale = 0.1
        # plotter.camera.position = (.2, .2, .2) # 64 ver default
        # plotter.camera.position = (0.25, 0.25, 0.25) # 64 ver cutting process
        plotter.camera.position = (0.1+0.2, 0.35+0.2, 0.1+0.2) # for parallel projection
        plotter.camera.up = (0.0, 0.0, 1.0)
        data = plotter.save_graphic(save_path+f'/screenshot_{k}.eps')
        # imgs =plotter.screenshot(save_path+f'/screenshot_{i}.png')
        imgs =plotter.screenshot()
        pil_image = Image.fromarray(np.asarray(imgs))
        # rendered_imgs.append(pil_image)
        plotter.close()

        # os.remove(save_path+f'/screenshot_{k}.eps')

        return pil_image



@ray.remote
def one_step_pcd_render_for_cutting_process(k, s_grid_config,sample_images,action,action_table,save_path):


        # tmp_mesh = pv.Box(bounds=(s_grid_config["bounds"][0]+0.02,
        #                           s_grid_config["bounds"][1]-0.02,
        #                           s_grid_config["bounds"][2]+0.02,
        #                           s_grid_config["bounds"][3]-0.02,
        #                           s_grid_config["bounds"][4]+0.02,
        #                           s_grid_config["bounds"][5]-0.02,
        #                         ))

        tmp_mesh = pv.Box(bounds=(s_grid_config["bounds"][0],
                                  s_grid_config["bounds"][1],
                                  s_grid_config["bounds"][2],
                                  s_grid_config["bounds"][3],
                                  s_grid_config["bounds"][4],
                                  s_grid_config["bounds"][5],
                                ))


        action_axis = action_table[action[k]]['axis']
        if action_axis == 'z':
            action_pos_candidate = np.linspace(s_grid_config["bounds"][0],s_grid_config["bounds"][1],s_grid_config["side_length"])
            action_pos = action_pos_candidate[action_table[action[k]]['loc']]
            cutting_plane_translation = np.asarray([0,0,action_pos])
            cutting_plane_rotation    = np.asarray([0,0,0])
        elif action_axis == 'y':
            action_pos_candidate = np.linspace(s_grid_config["bounds"][0],s_grid_config["bounds"][1],s_grid_config["side_length"])
            action_pos = action_pos_candidate[action_table[action[k]]['loc']]
            cutting_plane_translation = np.asarray([0,action_pos,0])
            cutting_plane_rotation    = np.asarray([90,0,0])
        elif action_axis  == 'x':
            action_pos_candidate = np.linspace(s_grid_config["bounds"][0],s_grid_config["bounds"][1],s_grid_config["side_length"])
            action_pos = action_pos_candidate[action_table[action[k]]['loc']]
            cutting_plane_translation = np.asarray([action_pos,0,0])
            cutting_plane_rotation    = np.asarray([0,90,0])

        cutting_plane_ = pv.Box(bounds=(s_grid_config["bounds"][0]-0.2,
                                  s_grid_config["bounds"][1]+0.2,
                                  s_grid_config["bounds"][2]-0.2,
                                  s_grid_config["bounds"][3]+0.2,
                                  -0.001,
                                  0.001,
                                ))
        cutting_plane_tranced = cutting_plane_.translate(cutting_plane_translation)
        cutting_plane = get_rotated_mesh(cutting_plane_tranced, cutting_plane_rotation)



        box_array_handler   = pv_box_array(grid_config=s_grid_config)
        _                   = box_array_handler.cast_mesh_to_box_array(mesh=copy.copy(tmp_mesh))
        box_arrays_data     = box_array_handler.get_box_array_data()
        nearby_cells        = box_arrays_data.boxes
        colors              = box_arrays_data.colors
        centers             = box_arrays_data.grid_centers
        grid_2dim           = box_arrays_data.grid_2dim_size
        grid_3dim           = box_arrays_data.grid_3dim_size
        batch_image_map     = box_array_handler.batch_image_map



        plotter = pv.Plotter(window_size=(512, 512),off_screen = True)
        step_image = sample_images[k]/255.0
        step_image = step_image.clip(0,1,step_image)



        updated_colors = box_array_handler.cast_2d_image_to_box_color(image=step_image,permute="z")

        points  = centers
        colors  = updated_colors
        alpha_values = np.zeros(centers.shape[0])+0.99
        outer_shape_transparency = 0.5
        null_shape_transparency = 0.08
        # null_shape_transparency = 0.0
        # 範囲内にある色の行をチェックし、alpha_valuesを設定
        alpha_values[np.all((colors >= np.asarray([-0.1, -0.1, -0.1])) & (colors < np.asarray([0.3, 0.3, 0.3])), axis=1)] = 0.0 # Black
        alpha_values[np.all((colors >= np.asarray([0.6, 0.6, 0.6])) & (colors <= np.asarray([0.9, 0.9, 0.9])), axis=1)] = outer_shape_transparency # outer shape
        alpha_values[np.all((colors > np.asarray([0.9, 0.9, 0.9])) & (colors < np.asarray([1.3, 1.3, 1.3])), axis=1)] = null_shape_transparency  # white

        colors[alpha_values==outer_shape_transparency] = np.asarray([0.7,0.7,0.7])
        # colors[alpha_values==null_shape_transparency] = np.asarray([0.85,0.85,0.85])
        colors[alpha_values==null_shape_transparency] = np.asarray([0.9,0.9,0.9])



        cloud = pv.PolyData(points)
        cloud["rgba"] = np.column_stack([colors*255, alpha_values * 255]).astype(np.uint8)

        # plotter.add_mesh(cutting_plane, color = "palevioletred",opacity = 1.0 ,show_edges = True)
        # plotter.add_mesh(cutting_plane, color = "lightpink",opacity = 0.5 ,show_edges = False)
        # plotter.add_mesh(cutting_plane, color = "lightpink",opacity = 0.1 ,show_edges = True)

        plotter.add_mesh(cloud, scalars="rgba", rgb=True, point_size=5)
        # plotter.add_mesh(cloud, scalars="rgba", rgb=True, point_size=10)

        plotter.add_points(points[np.where(alpha_values==null_shape_transparency)], color = [0.8,0.8,0.8], point_size = 0.8, opacity = 0.1, render_points_as_spheres=True)
        # plotter.add_points(points, color = [0.8,0.8,0.8], point_size = 0.8, opacity = 0.3) # 全体ボクセルの重心店をポイントクラウドで表示
        # plotter.add_mesh(cloud.scale([1,1,0.5], inplace=False), scalars="rgba", rgb=True, point_size=10)

        arrow_x = pv.Arrow(
        start=(0, 0, 0), direction=(1, 0, 0), scale=0.08)
        arrow_y = pv.Arrow(
        start=(0, 0, 0), direction=(0, 1, 0), scale=0.08)
        arrow_z = pv.Arrow(
        start=(0, 0, 0), direction=(0, 0, 1), scale=0.08)

        # plotter.add_mesh(arrow_x, color="r")
        # plotter.add_mesh(arrow_y, color='g')
        # plotter.add_mesh(arrow_z, color='b')

        # plotter.set_background('black', top='white')
        # plotter.set_background('gray')

        # plotter.camera.position = (.2, .2, .2) # 16 ver
        # plotter.camera.position = (1.2, 1.2, 1.2) # 64 ver

        # view_offset_val = -0.1 # render_0
        view_offset_val = 0.15 # render_1

        # plotter.camera.position = (-1.0, 0.7, 0.5) # polisher default
        # plotter.camera.position = (-1.+0.1, 0.7-0.1, 0.5-0.1)
        # plotter.camera.position = (-1.+-view_offset_val, 0.7+view_offset_val, 0.5+view_offset_val) # polisher
        plotter.camera.position = (-1.+-view_offset_val, 1.0+view_offset_val, 1.0+view_offset_val) # polisher
        

        # pl.camera.focal_point = (0.2, 0.3, 0.3)
        plotter.camera.up = (0.0, 0.0, 1.0)

        data = plotter.save_graphic(save_path+f'/screenshot_{k}.eps')
        # imgs =plotter.screenshot(save_path+f'/screenshot_{i}.png')
        imgs = plotter.screenshot()
        pil_image = Image.fromarray(np.asarray(imgs))
        # rendered_imgs.append(pil_image)
        plotter.close()

        # os.remove(save_path+f'/screenshot_{k}.eps')

        return pil_image

