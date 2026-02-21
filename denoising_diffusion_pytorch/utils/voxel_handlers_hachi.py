
import os
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw
import pickle
import _pickle as cpickle
import joblib
import gzip
from  tqdm import tqdm

import ray
from denoising_diffusion_pytorch.utils.os_utils import get_path,cstyle
from denoising_diffusion_pytorch.utils.assign_voxels_pallarel import assign_voxels

class box_array_data():

    """Represents a collection of 3D boxes arranged on a grid with associated color data.

    Attributes:
        boxes (dict): An dict of 3D boxes. {str(n) : <class 'pyvista.core.pointset.PolyData'>}.
        grid_centers (pyvista_ndarray): A 2D array of grid center coordinates, shape (n, 3).
        grid_2dim_size (tuple): A tuple representing the 2D grid size, calculated from the grid centers.
        grid_3dim_size (tuple): A tuple representing the 3D grid size, calculated from the grid centers.

    Methods:
        set_colors(colors): Sets the colors for each voxel.
        get_grid_centers(): Returns the grid centers.
        get_boxes(): Returns the voxels.
        get_colors(): Returns the voxel colors.
    """


    def __init__(self, boxes, grid_centers):
        """Initializes the BoxArrayData object. Initializes the grid sizes (2D and 3D) based on the number of grid centers.

        Args:
            boxes (dict): An dict of 3D boxes. {str(n) : <class 'pyvista.core.pointset.PolyData'>}.
            grid_centers (pyvista_ndarray): A 2D array of grid center coordinates, shape (n, 3).

        Raises:
            NotImplementedError: If not np.cbrt(self.grid_centers.shape[0]).is_integer() and np.sqrt(self.grid_centers.shape[0]).is_integer().
        """

        self.boxes = boxes
        self.grid_centers = grid_centers

        if np.cbrt(self.grid_centers.shape[0]).is_integer() and np.sqrt(self.grid_centers.shape[0]).is_integer():
            len_2d = int(np.sqrt(self.grid_centers.shape[0]))
            len_3d = int(np.cbrt(self.grid_centers.shape[0]))
            self.grid_2dim_size = (len_2d,len_2d)
            self.grid_3dim_size = (len_3d,len_3d,len_3d)
        else:
            NotImplementedError()


    def set_colors(self,colors):
        """Sets the colors for each voxel.

        Args:
            colors (np.ndarray): An array of colors, shape (n, 3), where each entry is a color.

        """
        self.colors = colors

    def get_grid_centers(self):
        return self.grid_centers

    def get_boxes(self):
        return self.boxes

    def get_colors(self):
        return self.colors


class pv_box_array():

    def __init__(self, grid_config):

        self.grid_bounds   = grid_config["bounds"]
        self.grid_side_len = grid_config["side_length"]

        ## create uniform voxel
        density =(self.grid_bounds[1]-self.grid_bounds[0])/self.grid_side_len
        mesh = pv.Box(bounds = (self.grid_bounds[0],self.grid_bounds[1],
                                self.grid_bounds[2],self.grid_bounds[3],
                                self.grid_bounds[4],self.grid_bounds[5]),level=6)

        self.grid = pv.voxelize(mesh,density=density)
        self.grid_centers = self.grid.cell_centers().points
        # self._box_colors = np.zeros((self.grid.GetNumberOfCells(), 3))+[0, 0, 0]
        self._box_colors = np.zeros((self.grid.GetNumberOfCells(), 3))+[1, 1, 1]


        box_array = self._create_box_array()
        self.box_array = box_array_data(boxes=box_array , grid_centers= self.grid_centers)
        self.box_array.set_colors(colors=self._box_colors)


        batch_img_len = int(self.box_array.grid_2dim_size[0]/self.box_array.grid_3dim_size[0])
        self.batch_image_map ={}
        k = 0
        for i in range(batch_img_len):
            for j in range(batch_img_len):
                data = {k:(i,j)}
                self.batch_image_map.update(data)
                k = k+1


    def _create_box_array(self):

        ## 立方体の一辺の長さ
        side_length = np.abs(self.grid_centers[1][0]-self.grid_centers[0][0])/2.1
        bounds = [-1, 1, -1, 1, -1, 1]
        cubes = pv.Box(bounds)
        cubes =  cubes.scale([side_length, side_length, side_length], inplace=False)


        nearby_cells = {}
        # 立方体を配置
        for i in range(self.grid_centers.shape[0]):
            cube_copy = cubes.translate(self.grid_centers[i])  # 立方体を中心座標に移動
            nearby_cells.update({str(i):cube_copy}) # コピーされた立方体を追加

        # self.box_array = nearby_cells

        return nearby_cells


    def cast_mesh_to_box_array(self,mesh):
        ## 立方体の一辺の長さ
        vicinity_box_length = np.abs(self.grid_centers[0]-self.grid_centers[1]).max()
        side_length = vicinity_box_length/2.0

        # import ipdb;ipdb.set_trace()
        ugrid = pv.voxelize(mesh,density=side_length)
        ugrid_cell_center = ugrid.cell_centers().points

        # colors = np.zeros((self.grid.GetNumberOfCells(), 3))+[1, 1, 1]
        idxs = self.grid.find_closest_cell(ugrid_cell_center)
        self._box_colors[idxs] = [0.8,0.8,0.1]

        self.box_array.set_colors(colors=self._box_colors)

        return self._box_colors


    def get_box_color_to_2d_image(self,box_color=None,permute = "x"):

        box_arrays_data =  self.get_box_array_data()
        grid_2dim    = box_arrays_data.grid_2dim_size[0]
        grid_3dim    = box_arrays_data.grid_3dim_size[0]
        batch_img_len = int(grid_2dim/grid_3dim)

        # box_color[0]    = np.asarray([0.9,0.2,0.2])
        # box_color[15]   = np.asarray([0.9,0.2,0.2])
        # box_color[240]  = np.asarray([0.9,0.2,0.2])
        # box_color[255]  = np.asarray([0.9,0.2,0.2])

        batch_2d_image_ = box_color.reshape(grid_3dim, grid_3dim, grid_3dim, 3)
        cast_image = np.empty((grid_2dim,grid_2dim,3))

        if permute == "z":
            batch_2d_image  = batch_2d_image_
        elif permute == "y":
            batch_2d_image  = batch_2d_image_.transpose(1,0,2,3)
        elif permute == "x":
            batch_2d_image  = batch_2d_image_.transpose(2,1,0,3)

        k = 0
        for j in range(batch_img_len):
            for i in range(batch_img_len):
                cast_image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim] = batch_2d_image[k]
                k = k+1

        return cast_image


    def cast_2d_image_to_box_color(self,image,permute):
        box_arrays_data =  self.get_box_array_data()
        grid_2dim    = box_arrays_data.grid_2dim_size[0]
        grid_3dim    = box_arrays_data.grid_3dim_size[0]
        batch_img_len = int(grid_2dim/grid_3dim)


        batch_2d_image_ = np.zeros((grid_3dim, grid_3dim, grid_3dim, 3))
        k = 0
        for j in range(batch_img_len):
            for i in range(batch_img_len):
                batch_2d_image_[k] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
                k = k+1


        if permute == "z":
            batch_2d_image  = batch_2d_image_
        elif permute == "y":
            batch_2d_image  = batch_2d_image_.transpose(1,0,2,3)
        elif permute == "x":
            batch_2d_image  = batch_2d_image_.transpose(2,1,0,3)


        self._box_colors = batch_2d_image.reshape(-1,3)

        self.box_array.set_colors(colors=self._box_colors)

        return self.get_box_array_data().colors



    def get_slice_image(self, image=None, slice_tag = [1]):

        box_arrays_data =  self.get_box_array_data()
        grid_2dim    = box_arrays_data.grid_2dim_size[0]
        grid_3dim    = box_arrays_data.grid_3dim_size[0]
        batch_img_len = int(grid_2dim/grid_3dim)

        cast_image = np.empty((grid_2dim,grid_2dim,3))

        k = 0
        for j in range(batch_img_len):
            for i in range(batch_img_len):
                if k in slice_tag:
                    cast_image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
                else:
                    cast_image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]*0.0
                k = k+1



        # batch_2d_tag = self.batch_image_map[slice_tag]
        # j = batch_2d_tag[0]
        # i = batch_2d_tag[1]

        # batch_2d_image_ = np.zeros((grid_3dim, grid_3dim, grid_3dim, 3))
        # k = 0
        # for j in range(batch_img_len):
        #     for i in range(batch_img_len):
        #         batch_2d_image_[k] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
        #         k = k+1

        # self._box_colors = batch_2d_image.reshape(-1,3)

        # self.box_array.set_colors(colors=self._box_colors)


        # return image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
        return cast_image


    def get_2d_image_to_mini_batch_image(self,image,permute):
        box_arrays_data =  self.get_box_array_data()
        grid_2dim    = box_arrays_data.grid_2dim_size[0]
        grid_3dim    = box_arrays_data.grid_3dim_size[0]
        batch_img_len = int(grid_2dim/grid_3dim)


        batch_2d_image_ = np.zeros((grid_3dim, grid_3dim, grid_3dim, 3))
        k = 0
        for j in range(batch_img_len):
            for i in range(batch_img_len):
                batch_2d_image_[k] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
                k = k+1


        if permute == "z":
            batch_2d_image  = batch_2d_image_
        elif permute == "y":
            batch_2d_image  = batch_2d_image_.transpose(1,0,2,3)
        elif permute == "x":
            batch_2d_image  = batch_2d_image_.transpose(2,1,0,3)

        return batch_2d_image


    def update_2d_image(self,image=None,batch_img =None, idx = (0,0)):
        box_arrays_data =  self.get_box_array_data()
        grid_2dim    = box_arrays_data.grid_2dim_size[0]
        grid_3dim    = box_arrays_data.grid_3dim_size[0]
        batch_img_len = int(grid_2dim/grid_3dim)
        j = idx[0]
        i = idx[1]
        image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim] = batch_img

        return image


    def get_box_array_data(self):
        return self.box_array



class pv_box_array_multi_type_obj():

    """A class that manages a 3D grid of voxelized boxes with color information and various image manipulation functionalities.

    This class allows the creation of a 3D grid of boxes, and supports the modification of box colors,
    generation of 2D images from the 3D grid, and other image manipulation operations.

    Attributes:
        grid_bounds (tuple): The 3D bounds for the grid, in the form (xmin, xmax, ymin, ymax, zmin, zmax).
        grid_side_len (int): The number of divisions along each axis for the grid.
        grid (pyvista.core.pointset.UnstructuredGrid): A pyvista object representing the voxelized grid.
        grid_centers (pyvista.core.pyvista_ndarray.pyvista_ndarray): The center points of each voxel in the grid.
        _box_colors (np.ndarray): A (n, 3) array representing the colors associated with each box.
        box_array (box_array_data): A custom class storing information about the grid's boxes.
        batch_image_map (dict): A mapping of 2D image batches to corresponding indices in the 3D grid.
        For example, self.batch_image_map = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (1, 3), 8: (2, 0), 9: (2, 1), 10: (2, 2), 11: (2, 3), 12: (3, 0), 13: (3, 1), 14: (3, 2), 15: (3, 3)}
    """

    def __init__(self, grid_config, pre_near_by_cells = None):
        """convert mesh to voxels with a grid configuration.

        Args:
            grid_config (dict): A dictionary containing the grid configuration with keys:
                                 - "bounds" (tuple): 3D bounds of the grid.
                                 - "side_length" (int): The number of divisions along each axis.
        """

        self.grid_bounds   = grid_config["bounds"]
        self.grid_side_len = grid_config["side_length"]

        ## create uniform voxel
        density =(self.grid_bounds[1]-self.grid_bounds[0])/self.grid_side_len
        mesh = pv.Box(bounds = (self.grid_bounds[0],self.grid_bounds[1],
                                self.grid_bounds[2],self.grid_bounds[3],
                                self.grid_bounds[4],self.grid_bounds[5]),level=6)

        # self.grid = pv.voxelize(mesh,density=density)
        self.grid = pv.voxelize(mesh,density=density,check_surface=False)
        self.grid_centers = self.grid.cell_centers().points
        # self._box_colors = np.zeros((self.grid.GetNumberOfCells(), 3))+[0, 0, 0]
        self._box_colors = np.zeros((self.grid.GetNumberOfCells(), 3))+[1, 1, 1]

        box_array = self._create_box_array(pre_nearby_cells=pre_near_by_cells)
        self.box_array = box_array_data(boxes=box_array , grid_centers= self.grid_centers)
        self.box_array.set_colors(colors=self._box_colors)


        batch_img_len = int(self.box_array.grid_2dim_size[0]/self.box_array.grid_3dim_size[0])
        self.batch_image_map ={}
        k = 0
        for i in range(batch_img_len):
            for j in range(batch_img_len):
                data = {k:(i,j)}
                self.batch_image_map.update(data)
                k = k+1



    def _create_box_array(self, pre_nearby_cells = None):
        """Creates the array of 3D boxes(voxels) placed at the grid centers.

        Returns:
            dict: A dictionary where keys are the box indices and values are the pyvista Box objects. i.e., rerun:{str(n) : <class 'pyvista.core.pointset.PolyData'>}
        """
        ## 立方体の一辺の長さ
        side_length = np.abs(self.grid_centers[1][0]-self.grid_centers[0][0])/2.1
        bounds = [-1, 1, -1, 1, -1, 1]
        cubes = pv.Box(bounds)
        cubes =  cubes.scale([side_length, side_length, side_length], inplace=False)

        if pre_nearby_cells is not None and str(self.grid_centers.shape[0]-1) in pre_nearby_cells:
            nearby_cells = pre_nearby_cells
        else:
            nearby_cells = {}
            # 立方体を配置
            for i in tqdm(range(self.grid_centers.shape[0])):
                cube_copy = cubes.translate(self.grid_centers[i])  # 立方体を中心座標に移動
                nearby_cells.update({str(i):cube_copy}) # コピーされた立方体を追加


        # # file_path = './my_dict.pkl.gz'
        # file_path = './my_dict.pkl'

        # if os.path.exists(file_path):
        #     with open(file_path, 'rb') as f:
        #         # load_nearby_cells = pickle.load(f)
        #         load_nearby_cells = cpickle.load(f)

        #     print("File is successfully loaded.")
        #     if str(self.grid_centers.shape[0]-1) in load_nearby_cells:
        #             nearby_cells = load_nearby_cells
        #     else:
        #         import ipdb;ipdb.set_trace()
        # else:
        #     # ファイルが存在しない場合
        #     print(f"{file_path} is not exist")

        #     nearby_cells = {}
        #     # 立方体を配置
        #     for i in tqdm(range(self.grid_centers.shape[0])):
        #         cube_copy = cubes.translate(self.grid_centers[i])  # 立方体を中心座標に移動
        #         nearby_cells.update({str(i):cube_copy}) # コピーされた立方体を追加

        #     # with open('./my_dict.pkl', 'wb') as f:
        #     #     pickle.dump(nearby_cells, f)

        #     with open(file_path, 'wb') as f:
        #         cpickle.dump(nearby_cells, f,  protocol=-1)


        #     # with open('./my_dict.joblib', 'wb') as f:
        #     #     joblib.dump(nearby_cells, f, compress = 3)

        #     # with open('./my_dict.pkl.gz', 'wb') as f:
        #     #     pickle.dump(nearby_cells, f)


        # try:
        #     print("try_load_nearby_cells")
        #     # with open('./my_dict.pkl', 'rb') as f:
        #     #     load_nearby_cells = pickle.load(f)

        #     # with open('./my_dict.joblib', 'rb') as f:
        #     #     load_nearby_cells = joblib.load(f)

        #     with open('./my_dict.pkl.gz', 'rb') as f:
        #         load_nearby_cells = pickle.load(f)

        #     print("try_load_nearby_cells")

        #     if str(self.grid_centers.shape[0]-1) in load_nearby_cells:
        #         nearby_cells = load_nearby_cells
        #         print("f")
        #     else:
        #         import ipdb;ipdb.set_trace()

        # except:
        #     nearby_cells = {}
        #     # 立方体を配置
        #     for i in tqdm(range(self.grid_centers.shape[0])):
        #         cube_copy = cubes.translate(self.grid_centers[i])  # 立方体を中心座標に移動
        #         nearby_cells.update({str(i):cube_copy}) # コピーされた立方体を追加

        #     # with open('./my_dict.pkl', 'wb') as f:
        #     #     pickle.dump(nearby_cells, f)

        #     # with open('./my_dict.joblib', 'wb') as f:
        #     #     joblib.dump(nearby_cells, f, compress = 3)


        #     with open('./my_dict.pkl.gz', 'wb') as f:
        #         pickle.dump(nearby_cells, f)

        #     print("hogege")

        # import ipdb;ipdb.set_trace()


        return nearby_cells



    def _create_box_arrayfsdafdsafsa(self):
        """Creates the array of 3D boxes(voxels) placed at the grid centers.

        Returns:
            dict: A dictionary where keys are the box indices and values are the pyvista Box objects. i.e., rerun:{str(n) : <class 'pyvista.core.pointset.PolyData'>}
        """
        ## 立方体の一辺の長さ
        side_length = np.abs(self.grid_centers[1][0]-self.grid_centers[0][0])/2.1
        bounds = [-1, 1, -1, 1, -1, 1]
        cubes = pv.Box(bounds)
        cubes =  cubes.scale([side_length, side_length, side_length], inplace=False)


        nearby_cells = {}
        cube_obj        = ray.put(cubes)
        length          = self.grid_centers.shape[0]
        grid_center_obg = self.grid_centers

        result_obj = [assign_voxels.remote(
            k,
            cube_obj,
            grid_center_obg) for k in range(length)]

        aa = ray.get(result_obj)

        for i in range(len(aa)):
                cube_copy = aa[i][str(i)]
                nearby_cells.update({str(i):cube_copy})

        # # 立方体を配置
        # for i in range(self.grid_centers.shape[0]):
        #     cube_copy = cubes.translate(self.grid_centers[i])  # 立方体を中心座標に移動
        #     nearby_cells.update({str(i):cube_copy}) # コピーされた立方体を追加

        # self.box_array = nearby_cells

        return nearby_cells



    def cast_mesh_to_box_array(self,mesh_components):
        """Maps mesh components to the voxel grid and update the voxel colors.

        Args:
            mesh_components (dict): A dictionary of mesh components with their corresponding colors. i.e., {'Component': {'mesh':pyvista.core.pointset.PolyData,
            "color":list}}

        Returns:
            np.ndarray: The updated box colors for the grid (n,3).
        """

        ## 立方体の一辺の長さ
        vicinity_box_length = np.abs(self.grid_centers[0]-self.grid_centers[1]).max()
        side_length = vicinity_box_length/2.0

        for idx, val in enumerate(mesh_components):
            # ugrid = pv.voxelize(mesh_components[val]["mesh"],density=side_length)
            ugrid = pv.voxelize(mesh_components[val]["mesh"],density=side_length,check_surface=False)
            ugrid_cell_center = ugrid.cell_centers().points

            # colors = np.zeros((self.grid.GetNumberOfCells(), 3))+[1, 1, 1]
            idxs = self.grid.find_closest_cell(ugrid_cell_center)
            if not len(idxs)==0:
                self._box_colors[idxs] = mesh_components[val]["color"]
            else:
                print(f"{cstyle.YELLOW}====== object name {val} is not allocated ======= {cstyle.END}")
                pass
        # ugrid = pv.voxelize(mesh,density=side_length)
        # ugrid_cell_center = ugrid.cell_centers().points

        # # colors = np.zeros((self.grid.GetNumberOfCells(), 3))+[1, 1, 1]
        # idxs = self.grid.find_closest_cell(ugrid_cell_center)
        # self._box_colors[idxs] = [0.8,0.8,0.1]

        self.box_array.set_colors(colors=self._box_colors)

        return self._box_colors


    def get_box_color_to_2d_image(self,box_color=None,permute = "x"):

        """Converts box colors into a 2D image representation.

        Args:
            box_color (np.ndarray, optional): A color array for the boxes.
            permute (str, optional): Permutation type for the image ('x', 'y', or 'z').

        Returns:
            np.ndarray: The generated 2D image (v*np.sqrt(v), v*np.sqrt(v), 3).
        """

        box_arrays_data =  self.get_box_array_data()
        grid_2dim    = box_arrays_data.grid_2dim_size[0]
        grid_3dim    = box_arrays_data.grid_3dim_size[0]
        batch_img_len = int(grid_2dim/grid_3dim)

        # box_color[0]    = np.asarray([0.9,0.2,0.2])
        # box_color[15]   = np.asarray([0.9,0.2,0.2])
        # box_color[240]  = np.asarray([0.9,0.2,0.2])
        # box_color[255]  = np.asarray([0.9,0.2,0.2])

        batch_2d_image_ = box_color.reshape(grid_3dim, grid_3dim, grid_3dim, 3)
        cast_image = np.empty((grid_2dim,grid_2dim,3))

        if permute == "z":
            batch_2d_image  = batch_2d_image_
        elif permute == "y":
            batch_2d_image  = batch_2d_image_.transpose(1,0,2,3)
        elif permute == "x":
            batch_2d_image  = batch_2d_image_.transpose(2,1,0,3)

        k = 0
        for j in range(batch_img_len):
            for i in range(batch_img_len):
                cast_image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim] = batch_2d_image[k]
                k = k+1

        return cast_image


    def cast_2d_image_to_box_color(self,image,permute):
        """Updates box colors from a 2D image.

        Args:
            image (np.ndarray): The input 2D image with color data (v*np.sqrt(v), v*np.sqrt(v), 3). v is grind dim.
            permute (str): The permutation type for the image ('x', 'y', or 'z').

        Returns:
            np.ndarray: The updated box colors (n,3). n is number of total voxel .
        """

        box_arrays_data =  self.get_box_array_data()
        grid_2dim    = box_arrays_data.grid_2dim_size[0]
        grid_3dim    = box_arrays_data.grid_3dim_size[0]
        batch_img_len = int(grid_2dim/grid_3dim)


        batch_2d_image_ = np.zeros((grid_3dim, grid_3dim, grid_3dim, 3))
        k = 0
        for j in range(batch_img_len):
            for i in range(batch_img_len):
                batch_2d_image_[k] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
                k = k+1


        if permute == "z":
            batch_2d_image  = batch_2d_image_
        elif permute == "y":
            batch_2d_image  = batch_2d_image_.transpose(1,0,2,3)
        elif permute == "x":
            batch_2d_image  = batch_2d_image_.transpose(2,1,0,3)


        self._box_colors = batch_2d_image.reshape(-1,3)

        self.box_array.set_colors(colors=self._box_colors)

        return self.get_box_array_data().colors



    def get_slice_image(self, image=None, slice_tag = [1]):
        """Generates a slice of the 2D image based on specific tags.

        Args:
            image (np.ndarray): The input 2D image to extract a slice from.
            slice_tag (list): A list of indices indicating which slices to keep.

        Returns:
            np.ndarray: The 2D slice of the image.
        """


        box_arrays_data =  self.get_box_array_data()
        grid_2dim    = box_arrays_data.grid_2dim_size[0]
        grid_3dim    = box_arrays_data.grid_3dim_size[0]
        batch_img_len = int(grid_2dim/grid_3dim)

        cast_image = np.empty((grid_2dim,grid_2dim,3))

        k = 0
        for j in range(batch_img_len):
            for i in range(batch_img_len):
                if k in slice_tag:
                    cast_image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
                else:
                    cast_image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]*0.0
                k = k+1



        # batch_2d_tag = self.batch_image_map[slice_tag]
        # j = batch_2d_tag[0]
        # i = batch_2d_tag[1]

        # batch_2d_image_ = np.zeros((grid_3dim, grid_3dim, grid_3dim, 3))
        # k = 0
        # for j in range(batch_img_len):
        #     for i in range(batch_img_len):
        #         batch_2d_image_[k] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
        #         k = k+1

        # self._box_colors = batch_2d_image.reshape(-1,3)

        # self.box_array.set_colors(colors=self._box_colors)


        # return image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
        return cast_image


    def get_2d_image_to_mini_batch_image(self,image,permute):
        """Converts a 2D image into a mini-batch image representation.

        Args:
            image (np.ndarray): The input 2D image. (v*np.sqrt(v), v*np.sqrt(v), 3). v is grind dim.
            permute (str): The permutation type for the image ('x', 'y', or 'z').

        Returns:
            np.ndarray: The mini-batch image. (v, v, v, 3). v is grind dim.

        """


        box_arrays_data =  self.get_box_array_data()
        grid_2dim    = box_arrays_data.grid_2dim_size[0]
        grid_3dim    = box_arrays_data.grid_3dim_size[0]
        batch_img_len = int(grid_2dim/grid_3dim)


        batch_2d_image_ = np.zeros((grid_3dim, grid_3dim, grid_3dim, 3))
        k = 0
        for j in range(batch_img_len):
            for i in range(batch_img_len):
                batch_2d_image_[k] = image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim]
                k = k+1


        if permute == "z":
            batch_2d_image  = batch_2d_image_
        elif permute == "y":
            batch_2d_image  = batch_2d_image_.transpose(1,0,2,3)
        elif permute == "x":
            batch_2d_image  = batch_2d_image_.transpose(2,1,0,3)

        return batch_2d_image


    def update_2d_image(self,image=None, batch_img =None, idx = (0,0)):
        """Updates a specific section of the 2D image with new data.

        Args:
            image (np.ndarray): The original 2D image. (v*np.sqrt(v), v*np.sqrt(v), 3). v is grind dim.
            batch_img (np.ndarray): The new batch image to replace part of the original image. (v, v, v, 3). v is grind dim.
            idx (tuple): The (row, col) index of the slice to update.

        Returns:
            np.ndarray: The updated 2D image. (v*np.sqrt(v), v*np.sqrt(v), 3). v is grind dim.
        """

        box_arrays_data =  self.get_box_array_data()
        grid_2dim    = box_arrays_data.grid_2dim_size[0]
        grid_3dim    = box_arrays_data.grid_3dim_size[0]
        batch_img_len = int(grid_2dim/grid_3dim)
        j = idx[0]
        i = idx[1]
        image[j*grid_3dim:(j+1)*grid_3dim,i*grid_3dim:(i+1)*grid_3dim] = batch_img

        return image


    def get_box_array_data(self):
        """Retrieves the box array data.

        Returns:
            box_array_data: A custom class storing information about the grid's boxes.
        """

        return self.box_array







































if __name__ == '__main__':


    mesh_source = "/home/haxhi/workspace/nedo-dismantling-PyBlender/dataset_1/geom_eval/Boxy_99/blend/"
    path, f_name = get_path(mesh_source,".stl")

    mesh1 = pv.read(path[0])
    mesh2 = pv.read(path[1])
    mesh3 = pv.read(path[2])
    mesh4 = pv.read(path[3])


    merged = mesh2.merge(mesh3)
    merged = merged.merge(mesh4)


    s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),
                        "side_length":16}

    box_array_handler   = pv_box_array(grid_config=s_grid_config)
    _                   = box_array_handler.cast_mesh_to_box_array(mesh=merged)
    box_arrays_data     = box_array_handler.get_box_array_data()


    nearby_cells = box_arrays_data.boxes
    colors       = box_arrays_data.colors
    centers      = box_arrays_data.grid_centers
    grid_2dim    = box_arrays_data.grid_2dim_size
    grid_3dim    = box_arrays_data.grid_3dim_size


    # colors[0]= np.asarray([0.9,0.2,0.2])
    # colors[63]= np.asarray([0.9,0.2,0.2])
    # colors[255]= np.asarray([0.9,0.2,0.2])
    # colors[-1]= np.asarray([0.9,0.2,0.2])
    # colors[256]= np.asarray([0.9,0.2,0.2])




    array_2d_reshaped = box_array_handler.get_box_color_to_2d_image(box_color=colors,permute="z")
    imgs_z = array_2d_reshaped
    pil_image = Image.fromarray((imgs_z*255).astype(np.uint8))
    pil_image.save(f"./cast_z_axis{0}.png")


    array_2d_reshaped = box_array_handler.get_box_color_to_2d_image(box_color=colors,permute="y")
    imgs_y = array_2d_reshaped
    pil_image = Image.fromarray((imgs_y*255).astype(np.uint8))
    pil_image.save(f"./cast_y_axis{0}.png")


    array_2d_reshaped = box_array_handler.get_box_color_to_2d_image(box_color=colors,permute="x")
    imgs_x = array_2d_reshaped
    pil_image = Image.fromarray((imgs_x*255).astype(np.uint8))
    pil_image.save(f"./cast_x_axis{0}.png")







    #===================================
    batch_images = np.random.rand(s_grid_config["side_length"],s_grid_config["side_length"],3)

    updated_images = box_array_handler.update_2d_image(image=imgs_z,batch_img=batch_images,idx=(0,0))
    updated_colors = box_array_handler.cast_2d_image_to_box_color(image=updated_images,permute="z")

    ## save as PIL images
    imgs = box_array_handler.get_box_color_to_2d_image(box_color=updated_colors,permute="z")
    pil_image = Image.fromarray((imgs*255).astype(np.uint8))
    pil_image.save(f"./sequence_{1}_z.png")



    img_2_0 = box_array_handler.get_box_color_to_2d_image(box_color=updated_colors,permute="x")
    img_2_1 = box_array_handler.update_2d_image(image=img_2_0,batch_img=batch_images,idx=(1,0))
    updated_colors = box_array_handler.cast_2d_image_to_box_color(image=img_2_1,permute="x")

    ## save as PIL images
    imgs = box_array_handler.get_box_color_to_2d_image(box_color=updated_colors,permute="z")
    pil_image = Image.fromarray((imgs*255).astype(np.uint8))
    pil_image.save(f"./sequence_{2}_z.png")

    ## save as PIL images
    imgs = box_array_handler.get_box_color_to_2d_image(box_color=updated_colors,permute="x")
    pil_image = Image.fromarray((imgs*255).astype(np.uint8))
    pil_image.save(f"./sequence_{2}_x.png")



    img_3_0 = box_array_handler.get_box_color_to_2d_image(box_color=updated_colors,permute="y")
    img_3_1 = box_array_handler.update_2d_image(image=img_3_0,batch_img=batch_images,idx=(2,2))
    updated_colors = box_array_handler.cast_2d_image_to_box_color(image=img_3_1,permute="y")

    ## save as PIL images
    imgs = box_array_handler.get_box_color_to_2d_image(box_color=updated_colors,permute="z")
    pil_image = Image.fromarray((imgs*255).astype(np.uint8))
    pil_image.save(f"./sequence_{3}_z.png")


    ## save as PIL images
    imgs = box_array_handler.get_box_color_to_2d_image(box_color=updated_colors,permute="y")
    pil_image = Image.fromarray((imgs*255).astype(np.uint8))
    pil_image.save(f"./sequence_{3}_y.png")





    colors = updated_colors


    import ipdb;ipdb.set_trace()



    # 表示
    plotter = pv.Plotter()

    for idx,elements in enumerate(nearby_cells):
        if np.all(colors[int(elements)] != np.asarray([1,1,1])):
            plotter.add_mesh(nearby_cells[elements], color = colors[int(elements)] ,opacity = 0.9 , show_edges=True)
        else:
            plotter.add_mesh(nearby_cells[elements],style='wireframe' ,opacity =1e-5 , show_edges=True, edge_opacity= 0.01)

        # plotter.add_mesh(nearby_cells[elements], color = colors[int(elements)] ,opacity = 0.1 , show_edges=True)

    plotter.add_points(centers, render_points_as_spheres=True, color = [0,0,0], opacity = 0.5, )


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
