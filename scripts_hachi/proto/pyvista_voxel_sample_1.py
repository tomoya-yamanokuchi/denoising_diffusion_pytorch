
import pyvista as pv
import os
from pyvista import examples

import numpy as np
# Load mesh and texture into PyVista
# mesh_path = os.path.join('mesh','rooster.obj')
# mesh = pv.read(mesh_path)
# tex_path = os.path.join('mesh','rooster01.jpg')
# tex = pv.read_texture(tex_path)


# mesh = examples.download_cow()

# # Initialize the plotter object with four sub plots
# pl = pv.Plotter(shape=(2, 2))
# # First subplot show the mesh with the texture
# pl.subplot(0, 0)
# pl.add_mesh(mesh,name='rooster')
# # pl.show()

# # Second subplot show the voxelized repsentation of the mesh with voxel size of 0.01. We remove the surface check as the mesh has small imperfections
# pl.subplot(0, 1)
# voxels = pv.voxelize(mesh, density=0.1, check_surface=False)
# # We add the voxels as a new mesh, add color and show their edges

# colors = np.random.uniform([0,0,0],[255,255,255],(voxels.n_points,3))

# voxels["colors"]  = colors

# pl.add_mesh(voxels, show_edges=True)
# import ipdb;ipdb.set_trace()


# # Third subplot shows the voxel representation using cones 
# pl.subplot(1,0)
# glyphs = voxels.glyph(factor=1e-3, geom=pv.Cone())
# pl.add_mesh(glyphs)

# # Forth subplot shows the voxels together with a contour showing the per voxel distance to the mesh
# pl.subplot(1,1)
# # Calculate the distance between the voxels and the mesh. Add the results as a new scalar to the voxels
# voxels.compute_implicit_distance(mesh, inplace=True)
# # Create a contour representing the calculated distance
# contours = voxels.contour(6, scalars="implicit_distance")
# # Add the voxels and the contour with different opacity to show both
# pl.add_mesh(voxels, opacity=0.25, scalars="implicit_distance")
# pl.add_mesh(contours, opacity=0.5, scalars="implicit_distance")


# # Link all four views so all cameras are moved at the same time
# pl.link_views()
# # Set camera start position
# pl.camera_position = 'xy'
# # Show everything
# import ipdb;ipdb.set_trace()
# pl.disable_depth_peeling()
# pl.show()








"""
.. _voxelize_surface_mesh_example:

Voxelize a Surface Mesh
~~~~~~~~~~~~~~~~~~~~~~~

Create a voxel model (like legos) of a closed surface or volumetric mesh.

This example also demonstrates how to compute an implicit distance from a
bounding :class:`pyvista.PolyData` surface.

"""
import numpy as np

import pyvista as pv

# sphinx_gallery_thumbnail_number = 2
from pyvista import examples

# Load a surface to voxelize
surface = examples.download_foot_bones()
surface

###############################################################################
cpos = [
    (7.656346967151718, -9.802071079151158, -11.021236183314311),
    (0.2224512272564101, -0.4594554282112895, 0.5549738359311297),
    (-0.6279216753504941, -0.7513057097368635, 0.20311105371647392),
]

surface.plot(cpos=cpos, opacity=0.75)


###############################################################################
# Create a voxel model of the bounding surface
voxels = pv.voxelize(surface, density=surface.length / 200)

# p = pv.Plotter()
# p.add_mesh(voxels, color=True, show_edges=True, opacity=0.5)
# p.add_mesh(surface, color="lightblue", opacity=0.5)
# p.show(cpos=cpos)


###############################################################################
# We could even add a scalar field to that new voxel model in case we
# wanted to create grids for modelling. In this case, let's add a scalar field
# for bone density noting:
voxels["density"] = np.full(voxels.n_cells, 3.65)  # g/cc
# voxels

###############################################################################
voxels.plot(scalars="density", cpos=cpos)


###############################################################################
# A constant scalar field is kind of boring, so let's get a little fancier by
# added a scalar field that varies by the distance from the bounding surface.
voxels.compute_implicit_distance(surface, inplace=True)
# voxels

###############################################################################
contours = voxels.contour(6, scalars="implicit_distance")

import ipdb;ipdb.set_trace()
p = pv.Plotter()
p.add_mesh(voxels, opacity=0.25, scalars="implicit_distance")
p.add_mesh(contours, opacity=0.5, scalars="implicit_distance")
p.show(cpos=cpos)

















# import numpy as np
# import pyvista as pv
# from matplotlib.colors import ListedColormap

# x =np.random.randint(0,100,5000)
# y = np.random.randint(0,100,5000)
# z = np.random.randint(0,100,5000)
# c = np.random.randint(0,1,5000)

# values = np.zeros((max(x) + 1, max(y) + 1, max(z) + 1))
# for i in range(len(x)):
#     values[x[i], y[i], z[i]] = c[i] + 1
# mapping = np.linspace(0, 1.8, 256)

# newcolors = np.zeros((256, 4))
# newcolors[mapping == 0] = np.array([0, 0, 0, 0.1]) #should be alpha=0 but 0.1 to see the problem 
# newcolors[mapping > 0.5] = np.array([0, 0, 1, 1])
# newcolors[mapping > 1.5] = np.array([1, 0, 0, 1])
# my_colormap = ListedColormap(newcolors)

# grid = pv.UniformGrid()
# grid.dimensions = values.shape
# grid.origin = (0, 0, 0)
# grid.spacing = (1, 1, 1)
# grid.point_data["values"] = values.flatten(order="F")


# color = np.asarray([[255,0,0]])
# colors  = color.repeat(grid.n_points,axis = 0)
# # import ipdb;ipdb.set_trace()

# colors = np.random.uniform([0,0,0],[255,255,255],(grid.n_points,3))
# grid["colors"]  = colors

# # voxels = pv.voxelize(grid, density=1.0, check_surface=False)
# pv = pv.Plotter()
# pv.add_mesh(grid, opacity=0.5, show_edges=True)
# # pv.add_mesh(grid,cmap= my_colormap, show_edges=True)

# pv.disable_depth_peeling()
# import ipdb;ipdb.set_trace()

# pv.show()

# # vol = grid.threshold(0.5)

# # # Smooth the surface
# # surf = vol.extract_geometry()
# # smooth = surf.smooth(n_iter=1000)
# # smooth.plot(show_edges=False, cmap=my_colormap)