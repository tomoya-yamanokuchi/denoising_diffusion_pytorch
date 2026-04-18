



import numpy as np
import pyvista as pv

from pyvista import CellType

from denoising_diffusion_pytorch.utils.os_utils import get_path





# from pyvista import examples
# import pyvista
# import numpy as np

# mesh = examples.download_foot_bones()
# density = mesh.length / 100
# x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
# x = np.arange(x_min, x_max, density)
# y = np.arange(y_min, y_max, density)
# z = np.arange(z_min, z_max, density)
# x, y, z = np.meshgrid(x, y, z)

# # Create unstructured grid from the structured grid
# grid = pyvista.StructuredGrid(x, y, z)
# ugrid = pyvista.UnstructuredGrid(grid)

# # get part of the mesh within the mesh's bounding surface.
# selection = ugrid.select_enclosed_points(mesh.extract_surface(),
#                                          tolerance=0.0,
#                                          check_surface=False)
# mask = selection.point_arrays['SelectedPoints'].view(np.bool)
# mask = mask.reshape(x.shape)

# pyvista.plot(grid.points, scalars=mask)




# mask = mask.reshape(x.shape)
# mask = pyvista.wrap(mask).threshold(1)

# p = pyvista.Plotter(notebook=False)
# p.add_mesh(mask)
# p.show()








# # UnstructuredGridの準備（適当なデータを仮定）
# points = np.random.rand(100, 3)  # 仮の点データ
# cells = np.array([[4, 0, 1, 2, 3]])  # 仮のセルデータ

# ugrid = pv.UnstructuredGrid()
# ugrid.points = points
# ugrid.cells = {'hexahedron': cells}





# cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])
# cell_type = np.array([CellType.HEXAHEDRON, CellType.HEXAHEDRON], np.int8)
# cell1 = np.array([[0, 0, 0],
#                   [1, 0, 0],
#                   [1, 1, 0],
#                   [0, 1, 0],
#                   [0, 0, 1],
#                   [1, 0, 1],
#                   [1, 1, 1],
#                   [0, 1, 1]], dtype=np.float32)
# cell2 = np.array([[0, 0, 2],
#                   [1, 0, 2],
#                   [1, 1, 2],
#                   [0, 1, 2],
#                   [0, 0, 3],
#                   [1, 0, 3],
#                   [1, 1, 3],
#                   [0, 1, 3]], dtype=np.float32)
# points = np.vstack((cell1, cell2))

# ugrid = pv.UnstructuredGrid(cells, cell_type, points)


import numpy as np
import pyvista as pv

# 16x16x16のグリッドマップを作成
nx, ny, nz = 16, 16, 16
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
z = np.linspace(0, 1, nz)
xx, yy, zz = np.meshgrid(x, y, z)
grid = pv.StructuredGrid(xx, yy, zz)

# z軸でスライスして2次元マップを作成
slice_index = 8  # スライスするz軸のインデックス
slice_grid = grid.slice(normal=[0, 0, 1], origin=[0, 0, z[slice_index]])

# 2次元マップを表示
slice_grid.plot(show_scalar_bar=False)
grid.plot(opacity = 0.6)











mesh_source = "/home/haxhi/workspace/nedo-dismantling-PyBlender/data/Boxy/blend/"
path, f_name = get_path(mesh_source,".stl")

mesh1 = pv.read(path[0])
mesh2 = pv.read(path[1])
mesh3 = pv.read(path[2])
mesh4 = pv.read(path[3])


plotter = pv.Plotter()

# plotter.add_mesh(mesh1,opacity =0.8)

# plotter.add_mesh(mesh2,opacity =0.8)
# plotter.add_mesh(mesh3,opacity =0.8)
# plotter.add_mesh(mesh4,opacity =0.8)

merged_ = mesh2.merge(mesh3)
merged = merged_.merge(mesh4)

plotter.add_mesh(merged,color =[0.8,0.8,0.8], opacity =0.1)

ugrid = pv.voxelize(merged,density=0.001)
# UnstructuredGridの範囲を取得
bounds = ugrid.bounds

# StructuredGridの点を生成
nx, ny, nz = 10, 10, 10  # 仮のボクセルサイズ
xx, yy, zz = np.meshgrid(   np.linspace(bounds[0], bounds[1], nx),
                            np.linspace(bounds[2], bounds[3], ny),
                            np.linspace(bounds[4], bounds[5], nz))
structured_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
sgrid = pv.StructuredGrid(xx, yy, zz)



s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,-0.05),
                "side_length":16}
mesh = pv.Box(bounds = (-0.05,0.05,-0.05,0.05,-0.05,0.05),level=6)
sgrid = pv.voxelize(mesh,density=(s_grid_config["bounds"][1]-s_grid_config["bounds"][0])/s_grid_config["side_length"])



colors = np.zeros((sgrid.GetNumberOfCells(), 3))+ [0, 0, 0.5]
ugrid_cell_center = ugrid.cell_centers().points
sgrid_cell_center = sgrid.cell_centers().points





# for i in range(ugrid_cell_center.shape[0]):
#     idx,elements = np.where(np.isclose(ugrid_cell_center[i],sgrid_cell_center))
#     colors[idx]= [1,0,0]

idx = sgrid.find_closest_cell(ugrid_cell_center)
colors[idx] = [1.0,0.0,0.0]

colors[10] = [0.8,0.8,0.1]
colors[190] = [0.8,0.8,0.1]
colors[191] = [0.8,0.8,0.1]


cell_colors = np.zeros((sgrid.n_cells, 3))+ [0, 0, 0.5]
cat_cell_index = sgrid.find_containing_cell(ugrid_cell_center)

cell_colors[cat_cell_index] = [0.8,0.8,0.1]
cell_colors[190] = [0.8,0.8,0.1]

import ipdb;ipdb.set_trace()


# # セルごとに色を決定
# colors = []
# import ipdb;ipdb.set_trace()
# for cell in ugrid.cells:
#     import ipdb;ipdb.set_trace()
#     cell_center = np.mean([ugrid.points[idx] for idx in cell], axis=0)
#     import ipdb;ipdb.set_trace()

#     # ボクセルの中心がUnstructuredGridの点に含まれているかどうかを確認する
#     voxel_center_in_points = any(np.all(np.isclose(p, cell_center) for p in ugrid.points))
#     if voxel_center_in_points:
#         colors.append([255, 0, 0])  # 存在するボクセルの色を赤色に設定
#     else:
#         colors.append([255, 255, 255])  # 存在しないボクセルの色を白色に設定

# # StructuredGridに色を関連付けて表示

# sgrid.cell_arrays['Colors'] = np.array(colors)

plotter.add_mesh(sgrid, scalars=cell_colors, rgb=True, opacity=0.5)

# plotter.add_mesh(sgrid, cell_colors=cell_colors, opacity=0.5)

plotter.add_points(sgrid_cell_center ,show_edges=True, render_points_as_spheres= True, color= [0.,0.,0.5], point_size = 20, opacity=0.1)
plotter.add_mesh(ugrid,opacity = 0.8, color =[0.2,0.8,0.2])
plotter.add_points(ugrid_cell_center, color = [0.2,0.8,0.2])
plotter.show()