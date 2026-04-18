# import numpy as np
# import matplotlib.pyplot as plt

# # サイズ
# N = 32

# # 各軸の1Dコストマップ
# cost_x = np.linspace(0, 1, N)
# cost_y = np.linspace(1, 0, N)
# cost_z = np.abs(np.sin(np.linspace(0, np.pi, N)))

# # 3Dグリッドでブロードキャスト
# cost_x_3d = cost_x[:, None, None] * np.ones((N, N, N))
# cost_y_3d = cost_y[None, :, None] * np.ones((N, N, N))
# cost_z_3d = cost_z[None, None, :] * np.ones((N, N, N))

# # 各点で最大コストとその軸を取得
# stacked = np.stack([cost_x_3d, cost_y_3d, cost_z_3d], axis=-1)
# max_cost = np.max(stacked, axis=-1)
# max_dir = np.argmax(stacked, axis=-1)

# # RGB色を作成（軸ごとに色を割り当て）
# color_volume = np.zeros((*max_dir.shape, 3))

# color_volume[max_dir == 0, 0] = max_cost[max_dir == 0]  # x軸=赤
# color_volume[max_dir == 1, 1] = max_cost[max_dir == 1]  # y軸=緑
# color_volume[max_dir == 2, 2] = max_cost[max_dir == 2]  # z軸=青

# # 可視化（matplotlibのvoxelで各点を塗る）
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# # 表示する範囲（全体 or 一部）
# filled = np.ones((N, N, N), dtype=bool)
# ax.voxels(filled, facecolors=color_volume, edgecolors=None)

# ax.set_title("Max-Cost Axis per Voxel (RGB colored)")
# plt.tight_layout()
# plt.show()


import numpy as np
import pyvista as pv

N = 32
cost_x = np.linspace(0, 1, N)
cost_y = np.linspace(1, 0, N)
cost_z = np.abs(np.sin(np.linspace(0, np.pi, N)))

cost_x_3d = cost_x[:, None, None] * np.ones((N, N, N))
cost_y_3d = cost_y[None, :, None] * np.ones((N, N, N))
cost_z_3d = cost_z[None, None, :] * np.ones((N, N, N))

stacked = np.stack([cost_x_3d, cost_y_3d, cost_z_3d], axis=-1)
max_cost = np.max(stacked, axis=-1)
max_dir = np.argmax(stacked, axis=-1)

# RGB配列を作成
color_volume = np.zeros((*max_dir.shape, 3))
color_volume[max_dir == 0, 0] = max_cost[max_dir == 0]
color_volume[max_dir == 1, 1] = max_cost[max_dir == 1]
color_volume[max_dir == 2, 2] = max_cost[max_dir == 2]

# PyVista用に構造化グリッドを作成
grid = pv.UniformGrid()
grid.dimensions = np.array(color_volume.shape[:3]) + 1
grid.spacing = (1, 1, 1)
grid.origin = (0, 0, 0)

# RGBカラーを1Dにして格納
rgba_flat = (color_volume * 255).astype(np.uint8).reshape(-1, 3)
rgba_flat = np.concatenate([rgba_flat, np.full((rgba_flat.shape[0], 1), 255, dtype=np.uint8)], axis=1)
grid["rgba"] = rgba_flat

# 描画
plotter = pv.Plotter()
plotter.add_volume(grid, scalars="rgba", rgba=True)
plotter.show()
