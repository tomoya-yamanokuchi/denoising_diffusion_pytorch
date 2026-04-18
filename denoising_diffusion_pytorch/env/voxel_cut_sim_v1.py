#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image
# from denoising_diffusion_pytorch.utils.voxel_handlers_hachi import pv_box_array_multi_type_obj
from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array_multi_type_obj
# from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy
from denoising_diffusion_pytorch.utils.pil_utils import numpy_to_pil ,cv2_hsv_mask ,pil_to_cv2,color_range_mask
from .types import AxisImages, DismantlingObservation, DismantlingInfo, DismantlingStepResult

class index_map():
    """Maps 1D indices to 2D coordinates and vice versa.

    This class provides functionality to convert between a 1D index and its corresponding
    2D coordinates (row, column) in a grid. The grid is assumed to be square, with the
    side length specified in the grid_config dictionary.

    Attributes:
        to_2d_loc (dict): A dictionary mapping 1D indices to 2D (i, j) coordinates.
        to_1d_loc (dict): A dictionary mapping 2D (i, j) coordinates to 1D indices.
    """

    def __init__(self,grid_config):

        """
        Initializes the IndexMap class with the grid configuration.

        Args:
            grid_config (dict): A dictionary containing the configuration of the grid.
                It must contain the "side_length" key, which specifies the side length of
                the square grid. The value should be a positive integer.

        Raises:
            NotImplementedError: If the side length is not a perfect square.


        Examples:
            >>> s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),
                                "side_length":16}
        """

        image_length = np.sqrt(grid_config["side_length"])
        if image_length.is_integer():
            image_length= int(image_length)
        else:
            NotImplementedError()

        to_2d_loc ={}
        to_1d_loc ={}

        k = 0
        for i in range(image_length):
            for j in range(image_length):
                to_2d_loc.update({k:(i,j)})
                to_1d_loc.update({(i,j):k})
                k+=1

        self.to_2d_loc = to_2d_loc
        self.to_1d_loc = to_1d_loc

    def map_1d_to_2d_loc(self,data):

        """Maps a 1D index to a 2D coordinate.

        Args:
            data (int): A 1D index to be converted to a 2D coordinate.

        Returns:
            tuple: A 2D coordinate (i, j) corresponding to the 1D index.

        Raises:
            KeyError: If the 1D index is not found in the mapping.
        """


        return self.to_2d_loc[data]


    def map_2d_to_1d_loc(self,data):
        """Maps a 2D coordinate to a 1D index.

        Args:
            data (tuple): A tuple representing the 2D coordinate (i, j) to be converted
                to a 1D index.

        Returns:
            int: The 1D index corresponding to the 2D coordinate.

        Raises:
            KeyError: If the 2D coordinate is not found in the mapping.
        """

        return self.to_1d_loc[data]




class voxel_cut_handler():
    """Handles voxel operations including initialization, map to slice image, and updates voxel color.

    This class manages operations related to voxel data, such as slicing voxel grids into 2D images,
    updating voxel colors, and extracting specific image slices based on actions. It interacts with
    the `index_map` and `pv_box_array_multi_type_obj` to map 3D voxel data to 2D images and modify
    voxel properties.

    Attributes:
        index_map_fn (indexmap):
        voxel_hander (pv_box_array_multi_type_obj): A handler for managing voxel data.
        colors (np.ndarray): An array of voxel colors.
        init_imgs_z (np.ndarray): A 2D image representing voxel data sliced along the Z-axis.
        init_imgs_x (np.ndarray): A 2D image representing voxel data sliced along the X-axis.
        init_imgs_y (np.ndarray): A 2D image representing voxel data sliced along the Y-axis.
    """
    def __init__(self, grid_config, mesh_components,zero_initialize,pre_near_by_cells=None):
        """Initializes the VoxelCutHandler class.

        Args:
            grid_config (dict): A dictionary containing the configuration for the grid (e.g., "side_length").
            mesh_components (object): Mesh data that will be cast into voxel data.
            zero_initialize (bool): If True, initializes the voxel colors to zero (black). If False, initializes
                                        with original mesh colors.

        Raises:
            NotImplementedError: If the `zero_initialize` value is neither True nor False.
        """

        ## create  slice pos to 2d image pos map fun
        self.index_map_fn       = index_map(grid_config=grid_config)
        ## create vocel handler　# 1) ボクセルハンドラ生成（内部にボクセル格子の幾何を持つ）
        self.voxel_hander       = pv_box_array_multi_type_obj(grid_config=grid_config,pre_near_by_cells=pre_near_by_cells)
        _                       = self.voxel_hander.cast_mesh_to_box_array(mesh_components=mesh_components)
        nearby_cells            = self.voxel_hander.get_box_array_data().boxes

        ## get voxel coros
        if zero_initialize is True:
            self.colors             = self.voxel_hander.get_box_array_data().colors *0.0
        elif zero_initialize is False:
            self.colors             = self.voxel_hander.get_box_array_data().colors
        else:
            NotImplementedError()


        ## get each axis sliced image
        self.init_imgs_z = self.voxel_hander.get_box_color_to_2d_image(box_color=self.colors,permute="z")
        # pil_image_ = Image.fromarray((imgs_z*255).astype(np.uint8))
        # save_name = f"{cond_save_path}/oracle_obs_cast_z_axis{0}.png"
        # pil_image_.save(save_name)

        ## save image
        self.init_imgs_x = self.voxel_hander.get_box_color_to_2d_image(box_color=self.colors,permute="x")
        # pil_image_ = Image.fromarray((imgs_x*255).astype(np.uint8))
        # save_name = f"{cond_save_path}/oracle_obs_cast_x_axis{0}.png"
        # pil_image_.save(save_name)

        ## save image
        self.init_imgs_y = self.voxel_hander.get_box_color_to_2d_image(box_color=self.colors,permute="y")
        # pil_image_ = Image.fromarray((imgs_y*255).astype(np.uint8))
        # save_name = f"{cond_save_path}/oracle_obs_cast_y_axis{0}.png"
        # pil_image_.save(save_name)



    def get_obs(self,action):
        """Extracts a 2D image based on the specified action.

        Args:
            action (dict): A dictionary specifying the axis ('axis') and the location ('loc') to extract.

        Returns:
            np.ndarray: The extracted 2D image slice corresponding to the action. Image size follows grid_config["side_length"]

        Examples:
            >>> action={'axis': 'y', 'loc': 9}
        """
        imgs  = self.voxel_hander.get_box_color_to_2d_image(box_color=self.colors, permute=action["axis"])
        batch =  self.voxel_hander.get_2d_image_to_mini_batch_image(image=imgs,permute="z")
        extract_image = batch[action['loc']]

        return extract_image


    def update_color(self,mini_batch_image,config):
        """Updates the voxel colors using the provided mini-batch image and configuration.

        Args:
            mini_batch_image (np.ndarray): A mini-batch image that contains updated color information.
            config (dict): A configuration dictionary containing the axis ('axis') and the location ('loc')
                            to apply the update to.
        """

        ## get 2d image according to config axis
        imgs            = self.voxel_hander.get_box_color_to_2d_image(box_color=self.colors, permute=config["axis"])
        ## update image according to config and minibatch image
        update_imgs     = self.voxel_hander.update_2d_image(image=imgs, batch_img=mini_batch_image, idx=self.index_map_fn.map_1d_to_2d_loc(config["loc"]))
        ## update box color based on updated imags
        self.cast_2d_image_to_box_color(img=update_imgs,config=config)


    def cast_2d_image_to_box_color(self,img,config):
        """Applies a 2D image slice back to the voxel color data.

        Args:
            img (np.ndarray): The updated 2D image to apply to the voxel color data.
            config (dict): The configuration dictionary containing the axis ('axis') for the update.
        """
        updated_colors  = self.voxel_hander.cast_2d_image_to_box_color(image=img, permute=config["axis"])
        self.colors     = updated_colors


    def get_2d_image(self,axis):
        """Returns the 2D image slice along the specified axis.

        Args:
            axis (str): The axis ('x', 'y', or 'z') along which to slice the voxel data.

        Returns:
            np.ndarray: The 2D image slice along the specified axis.
        """
        imgs            = self.voxel_hander.get_box_color_to_2d_image(box_color=self.colors, permute=axis)
        return imgs




class dismantling_env():
    """Dismantling Environment for voxel-based cutting tasks.

    This environment simulates the process of slicing a 3D voxel model using cutting actions,
    calculating rewards based on cutting costs, and managing observations through sequential
    voxel slice updates.

    Attributes:
        grid_config (dict)                : Configuration of the voxel grid,  including bounds and side length.
        oracle_obs_model (VoxelCutHandler): Model for oracle observation of the voxel grid (non-modified).
        seq_obs_model (VoxelCutHandler)   : Model for sequential observation, updating after each action.
        action_table (dict)               : A table mapping action indices to actions (axis and location).
        observation_history (dict)        : A history of actions taken during the environment's operation.
        oracle_target_shape_vol (float)   : The initial volume of the target shape (used for reward calculation).
        image_dim (tuple)                 : The dimensions of the image slice from the voxel grid.
        mini_batch_image_dim (tuple)      : Dimensions for the mini-batch image.
    """

    def __init__(self, grid_config, mesh_components, pre_near_by_cells=None):
        """Initializes the dismantling environment.

        Args:
            grid_config (dict): A dictionary containing the grid configuration, including "bounds" and "side_length".
            mesh_components (object): The mesh data representing the 3D model to be dismantled.

        Sets up the observation models and action table for voxel cutting.

        Examples:
                >>> s_grid_config = {   "bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),
                                        "side_length":16}
        """

        self.grid_config            = grid_config
        self.oracle_obs_model       = voxel_cut_handler(grid_config=self.grid_config, mesh_components=mesh_components,zero_initialize=False,pre_near_by_cells=pre_near_by_cells)
        self.seq_obs_model          = voxel_cut_handler(grid_config=self.grid_config, mesh_components=mesh_components,zero_initialize=True,pre_near_by_cells=pre_near_by_cells)

        self.action_table           = self.get_action_table(grid_config=self.grid_config)
        self.observation_history    = {}


        oracle_slice_image_z            = self.oracle_obs_model.init_imgs_z
        self.oracle_target_shape_vol    = self.get_reward(oracle_slice_image_z)

        self.image_dim                  =  oracle_slice_image_z.shape
        self.mini_batch_image_dim       =  (self.grid_config["side_length"],self.grid_config["side_length"],self.image_dim[2])


    def get_action_table(self,grid_config):
        """_summary_
            define slice action index

        Args:
            grid_config (dict)
        Returns:
            action table (dict): {i:{"axis":data_order[val],"loc":j}})
            i    : Serial number of the action index
            axis : axis name
            loc  : slice index
            In the current configuration, Data_order is unified as [“Z”, “X”, “Y”].
        """


        """Creates an action table that maps action indices to slice operations. In the current configuration, Data_order is unified as [“z”, “x”, “y”].

        Args:
            grid_config (dict): Configuration dictionary for the voxel grid.

        Returns:
            dict: A table mapping action indices to action descriptions.
                Each action includes the axis (e.g., "z", "x", "y") and the slice location.

        Examples:
            >>> action table (dict): {i:{"axis":data_order[val],"loc":j}})
            >>> i    : Serial number of the action index
            >>> axis : axis name
            >>> loc  : slice index
        """

        image_length = grid_config["side_length"]
        action_table  = {}

        i   = 0
        # data_order = ["x","y","z"]
        data_order = ["z","x","y"]
        # data_order = ["z","y","x"]
        for val in range(len(data_order)):
            for j in range(image_length):
                action_table.update({i:{"axis":data_order[val],"loc":j}})
                i+=1

        return action_table


    def get_reward(self,mini_batch_image):

        """Calculates the cutting cost (reward) based on color matching.

        Args:
            mini_batch_image (np.ndarray): A 2D image after slicing the voxel grid along an axis. (self.grid_config["side_length"]*3, self.grid_config["side_length"]*3, 3)

        Returns:
            float: The sum of matching pixels for the target color (used as the reward).
        """

        # until vae_20250507.py
        target_mask_y = np.asarray([0.8,0.8,0.2])
        image_mask_config_y = {"target_mask":target_mask_y,
                            "target_mask_lb":target_mask_y-np.asarray([0.1,0.1,0.1]),
                            "target_mask_ub":target_mask_y+np.asarray([0.2,0.2,0.6])}

        target_mask_b = np.asarray([0.2,0.8,0.8])
        image_mask_config_b = {"target_mask":target_mask_b,
                            "target_mask_lb":target_mask_b-np.asarray([0.1,0.1,0.1]),
                            "target_mask_ub":target_mask_b+np.asarray([0.7,0.2,0.2])}


        target_mask_r = np.asarray([0.8,0.2,0.2])
        image_mask_config_r = {"target_mask":target_mask_r,
                            "target_mask_lb":target_mask_r-np.asarray([0.1,0.1,0.1]),
                            "target_mask_ub":target_mask_r+np.asarray([0.2,0.5,0.5])}


        # target_mask_y = np.asarray([0.0,1.0,0.0])
        # image_mask_config_y = {"target_mask":target_mask_y,
        #                     "target_mask_lb":target_mask_y-np.asarray([0.1,0.1,0.1]),
        #                     "target_mask_ub":target_mask_y+np.asarray([0.1,0.0,0.1])}

        # target_mask_b = np.asarray([0.0,0.0,1.0])
        # image_mask_config_b = {"target_mask":target_mask_b,
        #                     "target_mask_lb":target_mask_b-np.asarray([0.1,0.1,0.1]),
        #                     "target_mask_ub":target_mask_b+np.asarray([0.1,0.1,0.0])}


        # target_mask_r = np.asarray([1.0,0.0,0.0])
        # image_mask_config_r = {"target_mask":target_mask_r,
        #                     "target_mask_lb":target_mask_r-np.asarray([0.1,0.1,0.1]),
        #                     "target_mask_ub":target_mask_r+np.asarray([0.0,0.1,0.1])}


        mask_image_blue  = color_range_mask(mini_batch_image,image_mask_config_b)
        mask_image_red   = color_range_mask(mini_batch_image,image_mask_config_r)
        mask_image_yellow = color_range_mask(mini_batch_image,image_mask_config_y)


        # current task setting, we only consider target mask blue
        mask_image = mask_image_blue
        target_mask_cutting_cost    = mask_image.mean(2).sum()

        return target_mask_cutting_cost


    def step(self,action_idx,partial_obs={}) -> DismantlingStepResult:
        """_summary_
            slice voxel model based on action index and return obs,reward,done,info
        Args:
            action_idx (np.int): Serial number of the action index
            partial_obs (dict, optional):   Information about the slice range that will not be observed due to the split by the cutting.
                                            Defaults to {}.
                                            e.g.,{'[0, 2]': {'axis': 'z', 'range': [0, 2], 'offset': 0}}
        Returns:
            _type_: _description_
        """

        """Performs a step in the environment by applying an cutting action.

        Args:
            action_idx (int): The index of the action to take.
            partial_obs (dict, optional): Information about previously unobserved slices due to the current cutting action.
            e.g.,partial_obs = {'[0, 2]': {'axis': 'z', 'range': [0, 2], 'offset': 0}}
        Returns:
            tuple: Contains the following:
                - obs (dict): The updated observations (sliced images and history).
                - reward (float): The reward obtained from the action.
                - done (bool): A flag indicating whether the task is complete.
                - info (dict): Additional information (e.g., target removal rate, volume).
        """

        action               = self.action_table[action_idx] # map action index to action dict
        mini_batch_image     = self.oracle_obs_model.get_obs(action= action)


        # partial_obs ={}
        ###########################################################################
        ## Helper function to apply partial observations to the mini-batch image.
        ###########################################################################
        update_flag = 1
        if len(partial_obs.keys()) != 0:
            if action["axis"] == "z":

                for idx,val in enumerate(partial_obs):
                    if partial_obs[val]["axis"]=="x":
                        start = partial_obs[val]["range"][0]
                        end   = partial_obs[val]["range"][1]+1
                        mini_batch_image[:,start:end,:]=0.0
                    elif partial_obs[val]["axis"]=="y":
                        start = partial_obs[val]["range"][0]
                        end   = partial_obs[val]["range"][1]+1
                        mini_batch_image[start:end,:,:]=0.0
                    elif partial_obs[val]["axis"]=="z":
                        if action["loc"]==partial_obs[val]["range"][1] or action["loc"]==partial_obs[val]["range"][0]:
                            mini_batch_image = mini_batch_image
                        elif partial_obs[val]["range"][0]<action["loc"]<partial_obs[val]["range"][1]:
                            mini_batch_image[:,:,:] = 0.0
                            update_flag = 0

                # z image (slice view)
                #   axis_x >
                #   axis_Y v
                #   +------------+
                #   |            |
                #   |            |
                #   |            |
                #   |            |
                #   +------------+

            if action["axis"] == "x":
                for idx,val in enumerate(partial_obs):
                    if partial_obs[val]["axis"]=="z":
                        start = partial_obs[val]["range"][0]
                        end   = partial_obs[val]["range"][1]+1
                        # mini_batch_image[:,2:6,:]=0.1
                        mini_batch_image[:,start:end,:]=0.0
                    elif partial_obs[val]["axis"]=="y":
                        start = partial_obs[val]["range"][0]
                        end   = partial_obs[val]["range"][1]+1
                        mini_batch_image[start:end,:,:]=0.0
                    elif partial_obs[val]["axis"]=="x":
                        if action["loc"]==partial_obs[val]["range"][1] or action["loc"]==partial_obs[val]["range"][0]:
                            mini_batch_image = mini_batch_image
                        elif partial_obs[val]["range"][0]<action["loc"]<partial_obs[val]["range"][1]:
                            mini_batch_image[:,:,:] = 0.0
                            update_flag = 0

                # X image (slice view)
                #   axis_Z >
                #   axis_Y v
                #   +------------+
                #   |            |
                #   |            |
                #   |            |
                #   |            |
                #   +------------+



            if action["axis"] == "y":
                for idx,val in enumerate(partial_obs):
                    if partial_obs[val]["axis"]=="x":
                        start = partial_obs[val]["range"][0]
                        end   = partial_obs[val]["range"][1]+1
                        mini_batch_image[:,start:end,:]=0.0
                    elif partial_obs[val]["axis"]=="z":
                        start = partial_obs[val]["range"][0]
                        end   = partial_obs[val]["range"][1]+1
                        mini_batch_image[start:end,:,:]=0.0
                    elif partial_obs[val]["axis"]=="y":
                        if action["loc"]==partial_obs[val]["range"][1] or action["loc"]==partial_obs[val]["range"][0]:
                            mini_batch_image = mini_batch_image
                        elif partial_obs[val]["range"][0]<action["loc"]<partial_obs[val]["range"][1]:
                            mini_batch_image[:,:,:] = 0.0

                            update_flag = 0

                # Y image (slice view)
                #   axis_X >
                #   axis_Z v
                #   +------------+
                #   |            |
                #   |            |
                #   |            |
                #   |            |
                #   +------------+


        if update_flag == 1:
            self.seq_obs_model.update_color(mini_batch_image=mini_batch_image,config=action)
        elif update_flag == 0:
            pass
        else:
            NotImplementedError()

        self.observation_history.update({action_idx:action})

        return DismantlingStepResult(
            observation = self.get_obs(),
            reward      = self.get_reward(mini_batch_image=mini_batch_image),
            done        = False,
            info        = self.get_info(),
        )


    def get_obs(self) -> DismantlingObservation:
        return DismantlingObservation(
            axis_images = AxisImages(
                x = self.seq_obs_model.get_2d_image(axis="x"),
                y = self.seq_obs_model.get_2d_image(axis="y"),
                z = self.seq_obs_model.get_2d_image(axis="z"),
            ),
            observation_history = self.observation_history,
        )


    def get_info(self):
        """Generates additional information about the current state.

        Returns:
            dict: Contains information such as the target removal rate, remaining volume, and the target shape volume.
        """
        oc_slice_image_x = self.oracle_obs_model.init_imgs_x
        oc_slice_image_y = self.oracle_obs_model.init_imgs_y
        oc_slice_image_z = self.oracle_obs_model.init_imgs_z

        current_target_removal_vol = self.get_reward(self.seq_obs_model.get_2d_image(axis="z"))
        target_removal_rate        = (current_target_removal_vol/self.oracle_target_shape_vol)*100.0

        ################################################
        ## get sum(unobserved pixels) values
        #################################################
        target_mask        =  np.asarray([0.0,0.0,0.0])
        image_mask_config  = {
            "target_mask"   : target_mask,
            "target_mask_lb": target_mask-0.0,
            "target_mask_ub": target_mask+0.0,
        }
        mask_image                   = color_range_mask(self.seq_obs_model.get_2d_image(axis="z"),image_mask_config)
        remaining_vol                = mask_image.mean(2).sum()+1e-6
        target_remaining_vol         = self.oracle_target_shape_vol-current_target_removal_vol
        target_to_remaining_vol_rate = (target_remaining_vol/remaining_vol)*100

        return DismantlingInfo(
            oracle_axis_images = AxisImages(
                x = oc_slice_image_x,
                y = oc_slice_image_y,
                z = oc_slice_image_z,
            ),
            observation_history  = self.observation_history,
            action_table         = self.action_table,
            target_removal_rate  = target_removal_rate,
            removal_performance  = target_to_remaining_vol_rate,
            remaining_vol        = remaining_vol,
            target_remaining_vol = target_remaining_vol,
        )


    def reset(self):
        return DismantlingStepResult(
            observation = self.get_obs(),
            reward      = 0.0,
            done        = False,
            info        = self.get_info(),
        )
