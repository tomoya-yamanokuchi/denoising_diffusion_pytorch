import numpy as np
import pyvista as pv
import copy


from denoising_diffusion_pytorch.utils.os_utils import get_path



# class pv_box_array():

#     def __init__(self, grid_config):

#         self.grid_bounds   = s_grid_config["bounds"]
#         self.grid_side_len = s_grid_config["side_length"]

#         ## create uniform voxel
#         density =(self.grid_bounds[1]-self.grid_bounds[0])/self.grid_side_len
#         mesh = pv.Box(bounds = (self.grid_bounds[0],self.grid_bounds[1],
#                                 self.grid_bounds[2],self.grid_bounds[3],
#                                 self.grid_bounds[4],self.grid_bounds[5]),level=6)

#         self.grid = pv.voxelize(mesh,density=density)
#         self.grid_centers = self.grid.cell_centers().points
#         self.box_array = self._create_box_array()

#     def _create_box_array(self):

#         ## 立方体の一辺の長さ
#         side_length = np.abs(self.grid_centers[1][0]-self.grid_centers[0][0])/2.0
#         bounds = [-1, 1, -1, 1, -1, 1]
#         cubes = pv.Box(bounds)
#         cubes =  cubes.scale([side_length, side_length, side_length], inplace=False)


#         nearby_cells = {}
#         # 立方体を配置
#         for i in range(centers.shape[0]):
#             cube_copy = cubes.translate(centers[i])  # 立方体を中心座標に移動
#             nearby_cells.update({str(i):cube_copy}) # コピーされた立方体を追加

#         self.box_array = nearby_cells

#         return nearby_cells


#     def cast_mesh_to_box_array(self,mesh):
#         ## 立方体の一辺の長さ
#         side_length = np.abs(self.grid_centers[1][0]-self.grid_centers[0][0])/2.0

#         ugrid = pv.voxelize(mesh,density=side_length)
#         ugrid_cell_center = ugrid.cell_centers().points

#         colors = np.zeros((self.grid.GetNumberOfCells(), 3))+ [0, 0, 0]
#         idxs = self.grid.find_closest_cell(ugrid_cell_center)
#         colors[idxs] = [0.8,0.8,0.1]



#     def get_box_array(self):
#         return self.box_array



# 立方体の数
num_cubes = 4096
# num_cubes = 32768


s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),
                "side_length":16}

bounds = s_grid_config["bounds"]
side_length = s_grid_config["side_length"]


# xx, yy, zz = np.meshgrid(   np.linspace(bounds[0], bounds[1], side_length),
#                             np.linspace(bounds[2], bounds[3], side_length),
#                             np.linspace(bounds[4], bounds[5], side_length))
# centers= np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))






density=(bounds[1]-bounds[0])/s_grid_config["side_length"]
mesh = pv.Box(bounds = (bounds[0],bounds[1],bounds[2],bounds[3],bounds[4],bounds[5]),level=6)
sgrid = pv.voxelize(mesh,density=density)
centers = sgrid.cell_centers().points


# 立方体の一辺の長さ
# side_length = (centers[1][2]-centers[0][2])/2.0
side_length = np.abs(centers[1][0]-centers[0][0])/2.0


# 立方体の頂点座標を計算
cube_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
]) * side_length

# 立方体のセルを定義
cube_cells = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7]
])


bounds = [-1, 1, -1, 1, -1, 1]
cubes = pv.Box(bounds)
cubes =  cubes.scale([side_length, side_length, side_length], inplace=False)

import ipdb;ipdb.set_trace()


nearby_cells = {}
# 立方体を配置
for i in range(centers.shape[0]):
    # cube_copy = copy.deepcopy(cubes)  # 各立方体のコピーを作成
    cube_copy = cubes.translate(centers[i])  # コピーされた立方体を中心座標に移動
    # cubes += cube_copy  # コピーされた立方体を追加
    nearby_cells.update({str(i):cube_copy})
    # print(f"{cube_copy.center}")
import ipdb;ipdb.set_trace()







mesh_source = "/home/haxhi/workspace/nedo-dismantling-PyBlender/data/Boxy/blend/"
path, f_name = get_path(mesh_source,".stl")

mesh1 = pv.read(path[0])
mesh2 = pv.read(path[1])
mesh3 = pv.read(path[2])
mesh4 = pv.read(path[3])

merged_ = mesh2.merge(mesh3)
merged = merged_.merge(mesh4)

ugrid = pv.voxelize(merged,density=side_length)
ugrid_cell_center = ugrid.cell_centers().points

colors = np.zeros((sgrid.GetNumberOfCells(), 3))+ [0, 0, 0]
idxs = sgrid.find_closest_cell(ugrid_cell_center)
colors[idxs] = [0.8,0.8,0.1]
import ipdb;ipdb.set_trace()



# 表示
plotter = pv.Plotter()


for idx,elements in enumerate(nearby_cells):

    if np.all(colors[int(elements)] != np.asarray([0,0,0])):
        plotter.add_mesh(nearby_cells[elements], color = colors[int(elements)] ,opacity = 0.9 , show_edges=True)
    else:
        plotter.add_mesh(nearby_cells[elements],style='wireframe' ,opacity = 0.1 , show_edges=True, edge_opacity= 0.01)

# plotter.add_mesh(nearby_cells["1"], opacity = 0.4 , show_edges=True)



plotter.add_points(centers)


# plotter.add_mesh(sgrid,scalars=colors, rgb=True,opacity = 0.5,show_edges=True)
plotter.add_mesh(merged,color =[0.1,0.8,0.8], opacity =0.4)

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