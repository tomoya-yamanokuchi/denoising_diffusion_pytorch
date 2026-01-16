import numpy as np
import pyvista as pv
from pyvista import examples


plotter = pv.Plotter()

# filename = examples.planefile
# mesh = pv.read(filename)
# voxel = pv.voxelize(mesh,density=2.0)

# UnstructuredGridの準備（適当なデータを仮定）
# ここではUnstructuredGridのx, y, z座標を仮に設定しています
# points = np.random.rand(100, 3)  # 仮の点データ
# cells = np.array([[4, 0, 1, 2, 3]])  # 仮のセルデータ
# ugrid = pv.UnstructuredGrid(points = points, cells = cells)


# # Load a surface to voxelize
# surface = examples.download_foot_bones()
# # Create a voxel model of the bounding surface
# ugrid = pv.voxelize(surface, density=surface.length / 100)












from pyvista import CellType


cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])
cell_type = np.array([CellType.HEXAHEDRON, CellType.HEXAHEDRON], np.int8)
cell1 = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 1],
                  [1, 1, 1],
                  [0, 1, 1]], dtype=np.float32)
cell2 = np.array([[0, 0, 2],
                  [1, 0, 2],
                  [1, 1, 2],
                  [0, 1, 2],
                  [0, 0, 3],
                  [1, 0, 3],
                  [1, 1, 3],
                  [0, 1, 3]], dtype=np.float32)
points = np.vstack((cell1, cell2))



ugrid = pv.UnstructuredGrid(cells, cell_type, points)

# UnstructuredGridの範囲を取得
bounds = np.asarray(ugrid.bounds)


import ipdb;ipdb.set_trace()



# StructuredGridを生成
nx, ny, nz = 10, 10, 10  # 仮のボクセルサイズ
xx, yy, zz = np.meshgrid(np.linspace(bounds[0], bounds[1], nx),
                          np.linspace(bounds[2], bounds[3], ny),
                          np.linspace(bounds[4], bounds[5], nz))

structured_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
sgrid = pv.StructuredGrid(xx, yy, zz)



# # UnstructuredGridの点がStructuredGridの点に近いものを探して、色を設定
# # セルごとに色を決定
# colors = []
# for cell in ugrid.cells:
#     cell_center = np.mean([ugrid.points[idx] for idx in cell], axis=0)
#     nearest_point_idx = np.argmin(np.linalg.norm(structured_points - cell_center, axis=1))
#     colors.append([255, 0, 0] if structured_points[nearest_point_idx] in points else [255, 255, 255])



# for i in range()
nearby_cells = {}
for i in range(structured_points.shape[0]):
    nearby_cells_candidate = ugrid.find_closest_point(structured_points[i], n=1)
    nearby_cells.update({i:nearby_cells_candidate})


# colors = np.ones_like(sgrid.points) * [0, 0, 255]  # 白色
colors = np.random.rand(sgrid.GetNumberOfCells(), 3)* [0, 0, 1]

import ipdb;ipdb.set_trace()


for i, cell in enumerate(nearby_cells):
    # import ipdb;ipdb.set_trace()
    # if ugrid.points[cell] in points:
    #     colors[i] = [255, 0, 0]  # 存在する点の色を赤色に設定


    if ugrid.points[nearby_cells[i]] in points:
        # import ipdb;ipdb.set_trace()
        colors[i] = [1, 0, 0]  # 存在する点の色を赤色に設定


# StructuredGridに色を関連付けて表示
plotter.add_mesh(sgrid, opacity = 0.8 , scalars=colors, rgb=True)
plotter.add_mesh(ugrid)

plotter.show()




