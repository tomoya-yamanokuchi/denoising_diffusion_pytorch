

import torch
from torch import nn



import numpy as np
import os
from PIL import Image,ImageDraw
import pyvista as pv

# 円のエッジを作る関数（XY平面に resolution 分割の円）
def make_circle_edge(z, radius=1.0, resolution=32):
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    points = np.c_[radius * np.cos(theta), radius * np.sin(theta), np.full(resolution, z)]

    # Create a polyline (line loop)
    lines = []
    for i in range(resolution):
        lines.extend([2, i, (i + 1) % resolution])  # 2-point line segments, looped
    poly = pv.PolyData()
    poly.points = points
    poly.lines = lines
    return poly



if __name__ == '__main__':





        # import ipdb;ipdb.set_trace()
        bb = pv.Box(bounds=(-.048, .048, -.048, .048, -.048, .048), level=0, quads=True)
        b2 = pv.Box(bounds=(-.046, .046, -.046, .046, -.046, .046), level=0, quads=True)

        # 表示
        plotter = pv.Plotter()
        # plotter.add_mesh(data["Body"]['mesh'], color =[0.1,0.1,0.1], opacity =0.1)
        plotter.add_mesh(bb, color =[0.1,0.1,0.1], opacity =0.1, show_edges=True, edge_color="black", line_width=10, edge_opacity=1.0, )
        edges = b2.extract_all_edges()
        plotter.add_mesh(bb.extract_all_edges(), color="black", line_width=3, opacity=1.0, )
        # plotter.add_mesh(data["Body"]['mesh'], color =[0.1,0.1,0.1], opacity =0.1, show_edges=True)
        # plotter.add_mesh(b2, color =[0.1,0.1,0.1], opacity =0.1, show_edges=True)

        c_1 = pv.Box(bounds=(-.01, .01, -.01, .01, -.022, .035), level=0, quads=True)
        plotter.add_mesh(c_1.translate([-0.025,0.02,-0.01]), color =[255/255,127/255,80/255], opacity =0.8, show_edges=False)

        c_2 = pv.Cylinder(center=[0.0, 0.0, 0], direction=[0, 0, 1], radius=0.02, height=0.08)
        plotter.add_mesh(c_2.translate([0.015,-0.025,0]), color =[30/255,144/255,255/255], opacity =0.8, show_edges=False)

        c_3 = pv.Box(bounds=(-.005, .005, -.02, .02, -.01, .01), level=0, quads=True)
        plotter.add_mesh(c_3.translate([0.02,0.025,0]), color =[46/255,139/255,87/255], opacity =0.9, show_edges=False)

        height= 0.08
        radius  = 0.02
        resolution = 1000
        edge_top = make_circle_edge(z=+height/2, radius=radius, resolution=resolution)
        edge_bottom = make_circle_edge(z=-height/2, radius=radius, resolution=resolution)
        # plotter.add_mesh(edge_top.translate([0.015,-0.025,0]), color="black", line_width=3)
        # plotter.add_mesh(edge_bottom.translate([0.015,-0.025,0]), color="black", line_width=3)

        arrow_x = pv.Arrow(
        start=(0, 0, 0), direction=(1, 0, 0), scale=0.05)
        arrow_y = pv.Arrow(
        start=(0, 0, 0), direction=(0, 1, 0), scale=0.05)
        arrow_z = pv.Arrow(
        start=(0, 0, 0), direction=(0, 0, 1), scale=0.05)
        plotter.set_background('white')
        plotter.camera.parallel_projection = True
        # plotter.camera.position = (0.2, 0.2, 0.15) # polisher #latest
        # plotter.add_camera_orientation_widget()
        # plotter.add_mesh(arrow_x, color="r")
        # plotter.add_mesh(arrow_y, color='g')
        # plotter.add_mesh(arrow_z, color='b')

        # plotter.camera.position = (0.3, 0.056, 0.06) # 64 ver cutting process
        # plotter.camera.up = (0.0, 0.0, 1.0)
        
        # plotter.camera_position = [
        #     (1, 1, 1),       # カメラ位置
        #     (0, 0, 0),       # 注視点（オブジェクトの中心）
        #     (0, 0, 1)        # 上方向ベクトル（通常Z軸）
        # ]

        # 斜投影っぽくする（カメラを斜め上から前にずらす）
        plotter.camera.position = (0.3, 0.1, 0.04)  # XZ 方向に傾ける（Yは固定）
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 0.0, 1.0)        # Y軸を上にする（正面保持）

        plotter.save_graphic(f"./sample_model.svg")
        # plotter.show()
        # plotter.show(screenshot='./airplane.png')