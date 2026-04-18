
import numpy as np
import json
from tqdm import tqdm
import copy
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from multiprocessing import Pool
from PIL import Image,ImageDraw
import ray

from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array
from denoising_diffusion_pytorch.utils.os_utils import save_json ,get_folder_name,create_folder,pickle_utils
from denoising_diffusion_pytorch.utils.render_parallel import one_step_voxel_render,one_step_voxel_render_for_cutting_process,one_step_pcd_render_for_cutting_process,one_step_voxel_render_for_cutting_process_w_cost_map

class pv_voxel_render():

    def __init__(self, s_grid_config, tmp_mesh):
        self.box_array_handler   = pv_box_array(grid_config=s_grid_config)
        _                   = self.box_array_handler.cast_mesh_to_box_array(mesh=copy.copy(tmp_mesh))
        box_arrays_data     = self.box_array_handler.get_box_array_data()

        self.nearby_cells   = box_arrays_data.boxes
        self.colors              = box_arrays_data.colors
        self.centers        = box_arrays_data.grid_centers
        grid_2dim           = box_arrays_data.grid_2dim_size
        grid_3dim           = box_arrays_data.grid_3dim_size
        batch_image_map     = self.box_array_handler.batch_image_map


    def render_initializer(self,save_path,init_color):
        ####################
        # init plot setting
        ####################
        plotter = BackgroundPlotter(show = False,
                window_size=(1080, 1080), title='Cut Visualization')
        plotter.open_gif("hosdfsdfsge.gif")


        # plotter.add_points(self.centers, render_points_as_spheres=True, color = [0,0,0], opacity = 0.5, )
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

        # plotter.render()
        # plotter.write_frame()
        # plotter.app.processEvents()

        return plotter





    def render_voxel_denoising(self,save_path,sample_images):

        plotter = self.render_initializer(10,[2,4,4])

        # plotter = BackgroundPlotter(show = True,
        #         window_size=(1080, 1080), title='Cut Visualization')




        for i in tqdm(range(sample_images.shape[0])):
            step_image = sample_images[i]/255.0
            updated_colors = self.box_array_handler.cast_2d_image_to_box_color(image=step_image,permute="z")


            plotter.clear()

            for idx,elements in tqdm(enumerate(self.nearby_cells)):
            #     if np.all(updated_colors[int(elements)] != np.asarray([1,1,1])):
            #         plotter.add_mesh(self.nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.9 , show_edges=True)
            #     else:
            #         # plotter.add_mesh(self.nearby_cells[elements],style='wireframe' ,opacity =1e-5 , show_edges=True, edge_opacity= 0.01)
            #         plotter.add_mesh(self.nearby_cells[elements],style='wireframe' ,opacity =1e-5 , show_edges=True,)
                plotter.add_mesh(self.nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.1 , show_edges=True)


            plotter.add_points(self.centers, render_points_as_spheres=True, color = [0,0,0], opacity = 0.1, )
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

            plotter.render()
            plotter.write_frame()
            plotter.app.processEvents()


        plotter.show()




    def render_voxel_denoisings(self,save_path,sample_images):


        # plotter = BackgroundPlotter(show = True,
        #         window_size=(1080, 1080), title='Cut Visualization')

        rendered_imgs = []
        for i in tqdm(range(sample_images.shape[0])):

            plotter = pv.Plotter(window_size=(512, 512),off_screen = True)
            # plotter.set_background('white')
            
            step_image = sample_images[i]/255.0
            step_image = step_image.clip(0,1,step_image)
            

            updated_colors = self.box_array_handler.cast_2d_image_to_box_color(image=step_image,permute="z")


            for idx,elements in tqdm(enumerate(self.nearby_cells)):
                if np.all(updated_colors[int(elements)] != np.asarray([1,1,1])):
                    plotter.add_mesh(self.nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.9 , show_edges=True)
                else:
                    # plotter.add_mesh(self.nearby_cells[elements],style='wireframe' ,opacity =1e-5 , show_edges=True, edge_opacity= 0.01)
                    plotter.add_mesh(self.nearby_cells[elements],style='wireframe' ,opacity =1e-5 , show_edges=True,)
                
                # plotter.add_mesh(self.nearby_cells[elements], color = updated_colors[int(elements)] ,opacity = 0.1 , show_edges=True)
                # plotter.add_mesh(self.nearby_cells[elements], color = [0.1,0.1,0,1] ,opacity = 1 , show_edges=True)
                # plotter.add_mesh(self.nearby_cells[elements], color = self.colors[int(elements)]  ,opacity = 1 , show_edges=True)
                
                # plotter.add_mesh(self.nearby_cells[elements])
                


            plotter.add_points(self.centers, render_points_as_spheres=True, color = [0,0,0], opacity = 0.1, )
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

            # plotter.camera_position = (0.5, 0.5, .0)
            # plotter.camera.up       = (0, 0.0, 1.0)


            data = plotter.save_graphic(save_path+f'/screenshot_{i}.eps')
            # imgs =plotter.screenshot(save_path+f'/screenshot_{i}.png')
            imgs =plotter.screenshot()
            pil_image = Image.fromarray(np.asarray(imgs))
            rendered_imgs.append(pil_image)
            plotter.close()
            # plotter.deep_clean()
            # plotter.show()
            # del plotter

        rendered_imgs[0].save(save_path+"/../"+f"voxel_diffusion.gif", save_all=True, append_images=rendered_imgs[1:], optimize=False, duration=50, loop=0)





class pv_voxel_render_parallel():
    def __init__(self):
        hoge = 1

    def render_voxel_denoising_v3(self, save_path, s_grind_config, sample_images):

        sample_images_obj = ray.put(sample_images)
        s_grid_config_obj = ray.put(s_grind_config)
        save_path_obj       = ray.put(save_path)

        # length = 10
        length  =sample_images.shape[0]

        result_obj = [one_step_voxel_render.remote(
            k,
            s_grid_config_obj,
            sample_images_obj,
            save_path_obj) for k in range(length)]

        # result_obj = [one_step_voxel_render.remote(
        #     k,
        #     s_grid_config_obj,
        #     sample_images_obj,
        #     save_path) for k in range(10)]

        denoising_process_3d = ray.get(result_obj)
        denoising_process_3d[0].save(save_path+"/../"+f"voxel_diffusion.gif", save_all=True, append_images=denoising_process_3d[1:], optimize=False, duration=50, loop=0)


    def render_cutting_process_v3(self, save_path, s_grind_config, action, action_table, sample_images, save_tag):

        sample_images_obj = ray.put(sample_images)
        s_grid_config_obj = ray.put(s_grind_config)
        action_obj        = ray.put(action)
        action_table_obj  = ray.put(action_table)
        save_path_obj     = ray.put(save_path)

        # length = 10
        length  =sample_images.shape[0]

        result_obj = [one_step_voxel_render_for_cutting_process.remote(
            k,
            s_grid_config_obj,
            sample_images_obj,
            action_obj,
            action_table_obj,
            save_path_obj) for k in range(length)]


        cutting_process_3d = ray.get(result_obj)
        cutting_process_3d[0].save(save_path+"/../"+f"cutting_process_{save_tag}.gif", save_all=True, append_images=cutting_process_3d[1:], optimize=False, duration=500, loop=0)



    def render_cutting_process_v4(self, save_path, s_grind_config, action, action_table, cost_map_logs, sample_images, save_tag):

        sample_images_obj = ray.put(sample_images)
        s_grid_config_obj = ray.put(s_grind_config)
        action_obj        = ray.put(action)
        action_table_obj  = ray.put(action_table)
        cost_map_logs_obj = ray.put(cost_map_logs)
        save_path_obj     = ray.put(save_path)

        # length = 10
        length  =sample_images.shape[0]

        result_obj = [one_step_voxel_render_for_cutting_process_w_cost_map.remote(
            k,
            s_grid_config_obj,
            sample_images_obj,
            action_obj,
            action_table_obj,
            cost_map_logs_obj,
            save_path_obj) for k in range(length)]


        cutting_process_3d = ray.get(result_obj)
        cutting_process_3d[0].save(save_path+"/../"+f"cutting_process_{save_tag}.gif", save_all=True, append_images=cutting_process_3d[1:], optimize=False, duration=500, loop=0)





    def render_cutting_process_as_pcl_v1(self, save_path, s_grind_config, action, action_table, sample_images, save_tag):


        sample_images_obj = ray.put(sample_images)
        s_grid_config_obj = ray.put(s_grind_config)
        action_obj        = ray.put(action)
        action_table_obj  = ray.put(action_table)
        save_path_obj     = ray.put(save_path)

        length  = sample_images.shape[0]

        result_obj = [one_step_pcd_render_for_cutting_process.remote(
            k,
            s_grid_config_obj,
            sample_images_obj,
            action_obj,
            action_table_obj,
            save_path_obj) for k in range(length)]


        cutting_process_3d = ray.get(result_obj)
        cutting_process_3d[0].save(save_path+"/../"+f"cutting_process_{save_tag}.gif", save_all=True, append_images=cutting_process_3d[1:], optimize=False, duration=500, loop=0)









if __name__ == '__main__':


    save_folder     = f"./results/voxel_image_64"+f"/eval_{13}_mid/"

    save_folders = get_folder_name(save_folder)


    # plotter = BackgroundPlotter(show = False,
    #         window_size=(1080, 1080), title='Cut Visualization')

    image_loss = []
    slice_tag  = []
    for i in range(len(save_folders)):
        if save_folders[i].startswith("batch"):
            load_data = pickle_utils().load(load_path=save_folder+"/"+save_folders[i]+f"/denoising_process_{i}.pickle")
            print(f"image_loss:{load_data['loss']}")
            sampled_images = load_data["denoising_process"]
            gt_mesh        = load_data["gt_mesh"]
            s_grind_config = load_data["s_grid_config"]
            renderer = pv_voxel_render(s_grid_config=s_grind_config,tmp_mesh=gt_mesh)
            renderer = pv_voxel_render(s_grid_config=s_grind_config,tmp_mesh=gt_mesh)
            renderer.render_voxel_denoising(save_path="./hjge",sample_images=sampled_images)
            # image_loss.append(load_data["loss"])
            # slice_tag = load_data["slice_tag"]


    image_loss_np = np.asarray(image_loss)

    data = {"image_loss":image_loss_np.tolist(),
            "image_loss_mean":image_loss_np.mean(),
            "image_loss_std" :image_loss_np.std(),
            "image_loss_var" :image_loss_np.var(),
            "slice_tag"      : slice_tag
            }


    save_name = save_folder+"/post_processed_data.json"
    # save_yaml(data=data,save_path=save_name)
    save_json(data=data,save_path=save_name) 