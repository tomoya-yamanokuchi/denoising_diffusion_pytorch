from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array_multi_type_obj
from .index_map import IndexMap

index_map = IndexMap


class VoxelCutHandler():
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

        # import ipdb; ipdb.set_trace()

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

