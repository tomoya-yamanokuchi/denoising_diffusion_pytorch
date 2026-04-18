import numpy as np
import pyvista as pv

from box_arrange_planner_test import random_placement_const_mode_1
from pyvistaqt import BackgroundPlotter






if __name__ == '__main__':



        container_size =  np.array([0.1, 0.05, 0.07])-0.005 # 直方体を配置する容器のサイズ

        # 直方体をランダムに配置
        # containers = random_placement((min_box_size, max_box_size), num_boxes, container_size)


        # # dataset_4_12900k geom_test_1
        # box_arrange_config = {
        #         "Box_1": {
        #         "position": { "min": [0.0, -0.02, 0.0], "max": [0.0, -0.005, 0.0] },
        #         "size": { "min": [0.08, 0.0, 0.05], "max": [0.1, 0.01, 0.1] }
        #         },
        #         "Box_2": {
        #         "position": { "min": [0.01, 0.005, 0.0], "max": [0.03, 0.02, 0.0] },
        #         "size": { "min": [0.03, 0.01, 0.02], "max": [0.04, 0.02, 0.05] }
        #         },
        #         "Box_3": {
        #         "position": { "min": [-0.01, 0.0, 0.0], "max": [-0.03, 0.02, 0.0] },
        #         "size": { "min": [0.01, 0.01, 0.01], "max": [0.03, 0.03, 0.03] }
        #         }
        #         }
        # color_1=[0.8,0.2,0.2]
        # color_2=[0.8,0.8,0.2]
        # color_3=[0.2,0.8,0.8]

        # # # dataset_4_12900k geom_test_2
        # box_arrange_config = {
        #                         "Box_1": {
        #                         "position": { "min": [0.0, -0.02, 0.0], "max": [0.0, -0.005, 0.0] },
        #                         "size": { "min": [0.08, 0.0, 0.05], "max": [0.1, 0.01, 0.1] }
        #                         },
        #                         "Box_2": {
        #                         "position": { "min": [0.01, 0.005, 0.0], "max": [0.03, 0.02, 0.0] },
        #                         "size": { "min": [0.03, 0.01, 0.02], "max": [0.04, 0.02, 0.05] }
        #                         },
        #                         "Box_3": {
        #                         "position": { "min": [-0.01, 0.0, 0.0], "max": [-0.03, 0.02, 0.0] },
        #                         "size": { "min": [0.01, 0.01, 0.01], "max": [0.03, 0.03, 0.03] }
        #                         }
        #                         }
        # # geom_test_2
        # color_1=[0.9,0.2,0.2]
        # color_2=[0.2,0.8,0.8]
        # color_3=[0.8,0.8,0.2]


        # # # # dataset_4_12900k geom_test_3
        # box_arrange_config ={
        #                         "Box_1": {
        #                         "position": { "min": [0.0, -0.02, 0.0], "max": [0.0, -0.005, 0.0] },
        #                         "size": { "min": [0.08, 0.0, 0.05], "max": [0.1, 0.01, 0.1] }
        #                         },
        #                         "Box_2": {
        #                         "position": { "min": [0.01, 0.005, 0.0], "max": [0.03, 0.02, 0.0] },
        #                         "size": { "min": [0.03, 0.01, 0.02], "max": [0.04, 0.02, 0.05] }
        #                         },
        #                         "Box_3": {
        #                         "position": { "min": [-0.01, 0.0, 0.0], "max": [-0.03, 0.02, 0.0] },
        #                         "size": { "min": [0.01, 0.01, 0.01], "max": [0.03, 0.03, 0.03] }
        #                         }
        #                         }
        # # # geom_test_3
        # color_1=[0.2,0.8,0.8]
        # color_2=[0.9,0.2,0.2]
        # color_3=[0.8,0.8,0.2]


        # dataset_5_12900k geom_test_1
        box_arrange_config = {
                                "Box_1": {
                                "position": { "min": [0.0, -0.02, 0.0], "max": [0.0, -0.005, 0.0] },
                                "size": { "min": [0.08, 0.0, 0.05], "max": [0.1, 0.01, 0.1] }
                                },
                                "Box_2": {
                                "position": { "min": [-0.01, 0.0, 0.0], "max": [-0.03, 0.02, 0.0] },
                                "size": { "min": [0.03, 0.01, 0.02], "max": [0.04, 0.02, 0.05] }
                                },
                                "Box_3": {
                                "position": { "min": [0.01, 0.005, 0.0], "max": [0.03, 0.02, 0.0] },
                                "size": { "min": [0.01, 0.01, 0.01], "max": [0.03, 0.03, 0.03] }
                                }
                                }
        # geom_test_1
        color_1=[0.9,0.2,0.2]
        color_2=[0.8,0.8,0.2]
        color_3=[0.2,0.8,0.8]
        
        # # dataset_6_13900k geom_test_1=
        # box_arrange_config = {
        #                         "Box_1": {
        #                         "position": { "min": [0.0, -0.04, 0.0], "max": [0.0, -0.0, 0.0] },
        #                         "size":     { "min": [0.08, 0.0, 0.05], "max": [0.1, 0.015, 0.1] }
        #                         },
        #                         "Box_2": {
        #                         "position": { "min": [-0.01, 0.0, 0.0], "max": [-0.03, 0.02, 0.0] },
        #                         "size":     { "min": [0.03, 0.01, 0.02], "max": [0.04, 0.02, 0.05] }
        #                         },
        #                         "Box_3": {
        #                         "position": { "min": [0.01, 0.005, 0.0], "max": [0.03, 0.02, 0.0] },
        #                         "size":     { "min": [0.01, 0.01, 0.01], "max": [0.03, 0.03, 0.03] }
        #                         }
        #                         }
        # # geom_test_1
        # color_1=[0.8,0.8,0.2]
        # color_2=[0.2,0.8,0.8]
        # color_3=[0.9,0.2,0.2]



        # import ipdb;ipdb.set_trace()


        # containers = random_placement(box_arrange_config, container_size)
        # import ipdb;ipdb.set_trace()


        ####################
        # init plot setting
        ####################
        plotter = BackgroundPlotter(show = True,
                window_size=(1080, 1080), title='Cut Visualization')
        # plotter.open_gif("./dataset_4_12900k_geom_test_3"+".gif")
        plotter.open_movie("./dataset_5_12900k_geom_test_1"+".mp4")


        outer_bounds    = np.asarray([-container_size[0], container_size[0], -container_size[1], container_size[1], -container_size[2], container_size[2]])/2.0
        outer_cubes     = pv.Box(outer_bounds)
        #     plotter.add_mesh(outer_cubes, opacity=0.1, show_edges=True, color = [0.1,0.1,0.1])

        bb = pv.Box(bounds=(-.048, .048, -.048, .048, -.048, .048), level=0, quads=True)
        b2 = pv.Box(bounds=(-.046, .046, -.046, .046, -.046, .046), level=0, quads=True)
        plotter.add_mesh(bb, color =[164/255,163/255,159/255], opacity =0.08, show_edges=True)
        plotter.add_mesh(b2, color =[164/255,163/255,159/255], opacity =0.08, show_edges=True)


        arrow_x = pv.Arrow(
        start=(0, 0, 0), direction=(1, 0, 0), scale=0.01)
        arrow_y = pv.Arrow(
        start=(0, 0, 0), direction=(0, 1, 0), scale=0.01)
        arrow_z = pv.Arrow(
        start=(0, 0, 0), direction=(0, 0, 1), scale=0.01)
        # plotter.add_camera_orientation_widget()
        # plotter.add_mesh(arrow_x, color="r")
        # plotter.add_mesh(arrow_y, color='g')
        # plotter.add_mesh(arrow_z, color='b')


        inner_box_1 = pv.Box(outer_bounds*0.3)
        inner_box_2 = pv.Box(outer_bounds*0.3)
        inner_box_3 = pv.Box(outer_bounds*0.3)

        inner_box_list =[inner_box_1,inner_box_2,inner_box_3]


        #     plotter.add_mesh(inner_box_1, opacity=0.8, color = [0.9,0.2,0,2],show_edges=True,)
        #     plotter.add_mesh(inner_box_2, opacity=0.8, color = [0.8,0.8,0,2],show_edges=True,)
        #     plotter.add_mesh(inner_box_3, opacity=0.8, color = [0.2,0.8,0,8],show_edges=True,)

        plotter.add_mesh(inner_box_list[0], opacity=0.8, color = color_1,show_edges=True,)
        plotter.add_mesh(inner_box_list[1], opacity=0.8, color = color_2,show_edges=True,)
        plotter.add_mesh(inner_box_list[2], opacity=0.8, color = color_3,show_edges=True,)


        plotter.set_background('white')
        plotter.render()
        plotter.write_frame()
        plotter.app.processEvents()

        for j in range(50):

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

