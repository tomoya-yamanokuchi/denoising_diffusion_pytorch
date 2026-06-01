#!/usr/bin/env python
# coding: utf-8

import numpy as np

from denoising_diffusion_pytorch.env.voxel.index_map import IndexMap
from denoising_diffusion_pytorch.env.voxel.voxel_cut_handler import VoxelCutHandler
from denoising_diffusion_pytorch.env.metrics.cutting_metric_calculator import CuttingMetricCalculator

from .types import AxisImages, DismantlingObservation, DismantlingInfo, DismantlingStepResult


index_map = IndexMap
voxel_cut_handler = VoxelCutHandler


class dismantling_env():
    def __init__(
        self,
        grid_config,
        mesh_components,
        pre_near_by_cells=None,
        metric_calculator=None,
    ):
        if metric_calculator is None:
            raise ValueError(
                "metric_calculator must be provided. "
                "Build it from Hydra config in EvalBuilder."
            )

        # --- static config ---
        self.grid_config       = grid_config
        self.mesh_components   = mesh_components
        self.pre_near_by_cells = pre_near_by_cells
        self.metric_calculator = metric_calculator
        # --- observation model ---
        self.oracle_obs_model = voxel_cut_handler(
            grid_config       = self.grid_config,
            mesh_components   = self.mesh_components,
            zero_initialize   = False,
            pre_near_by_cells = self.pre_near_by_cells,
        )

        self.seq_obs_model = voxel_cut_handler(
            grid_config       = self.grid_config,
            mesh_components   = self.mesh_components,
            zero_initialize   = True,
            pre_near_by_cells = self.pre_near_by_cells,
        )

        self.action_table           = self.get_action_table(grid_config=self.grid_config)
        self.observation_history    = {}

        oracle_slice_image_z         = self.oracle_obs_model.init_imgs_z
        self.oracle_target_shape_vol = self.calculate_cutting_error_volume(oracle_slice_image_z)


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


    def calculate_cutting_error_volume(self, mini_batch_image):
        return self.metric_calculator.calculate_cutting_error_volume(mini_batch_image)


    def step(self, action_idx, partial_obs=None) -> DismantlingStepResult:
        if partial_obs is None:
            partial_obs = {}


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
                            # import ipdb;ipdb.set_trace()
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
                            # import ipdb;ipdb.set_trace()
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
                            # import ipdb;ipdb.set_trace()

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
            observation          = self.get_obs(),
            cutting_error_volume = self.calculate_cutting_error_volume(mini_batch_image=mini_batch_image),
            done                 = False,
            info                 = self.get_info(),
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


    def get_info(self) -> DismantlingInfo:
        oracle_axis_images = AxisImages(
            x=self.oracle_obs_model.init_imgs_x,
            y=self.oracle_obs_model.init_imgs_y,
            z=self.oracle_obs_model.init_imgs_z,
        )
        sequential_observation_z = self.seq_obs_model.get_2d_image(axis="z")

        return self.metric_calculator.build_dismantling_info(
            oracle_axis_images       = oracle_axis_images,
            sequential_observation_z = sequential_observation_z,
            oracle_target_shape_vol  = self.oracle_target_shape_vol,
            observation_history      = self.observation_history,
            action_table             = self.action_table,
        )


    def _reset_sequential_state(self) -> None:
        self.seq_obs_model = voxel_cut_handler(
            grid_config       = self.grid_config,
            mesh_components   = self.mesh_components,
            zero_initialize   = True,
            pre_near_by_cells = self.pre_near_by_cells,
        )
        self.observation_history = {}


    def reset(self):
        # ----
        self._reset_sequential_state()
        # ----
        return DismantlingStepResult(
            observation          = self.get_obs(),
            cutting_error_volume = 0.0,
            done                 = False,
            info                 = self.get_info(),
        )

