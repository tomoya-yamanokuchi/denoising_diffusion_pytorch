
import pyvista as pv
import os
from pyvista import examples
import numpy as np


from denoising_diffusion_pytorch.utils.os_utils import get_path








if __name__ == '__main__':

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

    merged = mesh2.merge(mesh3)
    merged = merged.merge(mesh4)
    plotter.add_mesh(merged,opacity =0.1)

    voxel = pv.voxelize(merged,density=0.001)

    scalar_data =   np.random.rand(voxel.GetNumberOfPoints())# 仮のスカラーデータを生成
    plotter.add_mesh(voxel, scalars=scalar_data, cmap='viridis' ,opacity=0.8)
    
    
    # ボクセルのRGB値をランダムに生成
    colors = np.random.rand(voxel.GetNumberOfCells(), 3)

    plotter.add_mesh(voxel, scalars=colors, rgb=True)


    import ipdb;ipdb.set_trace()
    plotter.show()