from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pyvista as pv
import ray
from PIL import Image
from scipy.spatial.transform import Rotation

from denoising_diffusion_pytorch.utils.voxel_handlers import pv_box_array

pv.global_theme.allow_empty_mesh = True


def get_random_transformation_matrix(translation: Any, rotation: Any) -> np.ndarray:
    rot = Rotation.from_euler("xyz", rotation, degrees=True)
    rot_matrix = rot.as_matrix()
    bottom = np.zeros((1, 3))
    matrix = np.vstack((rot_matrix, bottom))
    trans = np.asarray([[translation[0]], [translation[1]], [translation[2]], [1]])
    return np.hstack((matrix, trans))


def get_rotated_mesh(mesh: pv.DataSet, rotation: Any) -> pv.DataSet:
    origin = mesh.center
    mesh2 = mesh.translate(np.asarray(origin) * -1.0)
    homo_matrix = get_random_transformation_matrix([0.0, 0.0, 0.0], rotation=rotation)
    mesh3 = mesh2.transform(homo_matrix)
    return mesh3.translate(np.asarray(origin))


def _make_cutting_plane(s_grid_config: dict, action_idx: int, action_table: Any) -> pv.DataSet:
    action_axis = action_table[action_idx]["axis"]
    loc_idx = action_table[action_idx]["loc"]

    action_pos_candidate = np.linspace(
        s_grid_config["bounds"][0],
        s_grid_config["bounds"][1],
        s_grid_config["side_length"],
    )
    action_pos = action_pos_candidate[loc_idx]

    if action_axis == "z":
        cutting_plane_translation = np.asarray([0, 0, action_pos])
        cutting_plane_rotation = np.asarray([0, 0, 0])
    elif action_axis == "y":
        cutting_plane_translation = np.asarray([0, action_pos, 0])
        cutting_plane_rotation = np.asarray([90, 0, 0])
    elif action_axis == "x":
        cutting_plane_translation = np.asarray([action_pos, 0, 0])
        cutting_plane_rotation = np.asarray([0, 90, 0])
    else:
        raise ValueError(f"Unsupported action axis: {action_axis}")

    cutting_plane_base = pv.Box(
        bounds=(
            s_grid_config["bounds"][0] - 0.01,
            s_grid_config["bounds"][1] + 0.01,
            s_grid_config["bounds"][2] - 0.01,
            s_grid_config["bounds"][3] + 0.01,
            -0.0001,
            0.0001,
        )
    )
    translated = cutting_plane_base.translate(cutting_plane_translation)
    return get_rotated_mesh(translated, cutting_plane_rotation)


def one_step_voxel_render_for_cutting_process_local(
    k: int,
    s_grid_config: dict,
    sample_images: np.ndarray,
    action: np.ndarray,
    action_table: Any,
    save_path: str,
    save_eps: bool = False,
) -> Image.Image:
    """Render one cutting-process frame.

    This is the local implementation used by both serial rendering and the Ray
    worker wrapper. EPS export is intentionally optional because
    ``plotter.save_graphic(... .eps)`` is fragile in headless Docker + Ray
    environments, while GIF generation only needs ``plotter.screenshot()``.
    """
    tmp_mesh = pv.Box(
        bounds=(
            s_grid_config["bounds"][0],
            s_grid_config["bounds"][1],
            s_grid_config["bounds"][2],
            s_grid_config["bounds"][3],
            s_grid_config["bounds"][4],
            s_grid_config["bounds"][5],
        )
    )

    action_idx = int(action[k])
    cutting_plane = _make_cutting_plane(
        s_grid_config=s_grid_config,
        action_idx=action_idx,
        action_table=action_table,
    )

    box_array_handler = pv_box_array(grid_config=s_grid_config)
    _ = box_array_handler.cast_mesh_to_box_array(mesh=copy.copy(tmp_mesh))
    box_arrays_data = box_array_handler.get_box_array_data()
    nearby_cells = box_arrays_data.boxes
    centers = box_arrays_data.grid_centers

    plotter = pv.Plotter(window_size=(800, 800), off_screen=True)

    try:
        step_image = sample_images[k] / 255.0
        step_image = step_image.clip(0, 1, step_image)
        updated_colors = box_array_handler.cast_2d_image_to_box_color(
            image=step_image,
            permute="z",
        )

        for elements in nearby_cells:
            color = updated_colors[int(elements)]

            # normal visualization mode
            if np.all(color >= np.asarray([0.0, 0.0, 0.0])) and np.all(
                color < np.asarray([0.5, 0.5, 0.5])
            ):
                pass
            else:
                if np.all(color >= np.asarray([0.5, 0.5, 0.5])) and np.all(
                    color < np.asarray([1.3, 1.3, 1.3])
                ):
                    opacity = 0.1
                    plotter.add_mesh(
                        nearby_cells[elements],
                        style="wireframe",
                        opacity=0.001,
                        show_edges=True,
                        edge_opacity=0.01,
                        color=[0.8, 0.8, 0.8],
                    )
                    plotter.add_mesh(
                        nearby_cells[elements],
                        color=color,
                        opacity=opacity,
                        show_edges=True,
                    )
                else:
                    plotter.add_mesh(
                        nearby_cells[elements],
                        color=color,
                        opacity=0.9,
                        show_edges=True,
                    )

            plotter.add_mesh(
                nearby_cells[elements],
                color=color,
                opacity=1e-10,
                show_edges=True,
            )

        plotter.add_points(centers, render_points_as_spheres=True, color=[0, 0, 0], opacity=1e-10)

        if k % 2 == 0:
            plotter.add_mesh(
                cutting_plane,
                color=(226 / 255.0, 220 / 255.0, 222 / 255.0),
                opacity=0.8,
                show_edges=False,
                diffuse=1.0,
            )
        else:
            plotter.add_mesh(
                cutting_plane,
                color=(0.7, 0.7, 0.0),
                opacity=0.0,
                show_edges=False,
            )

        cube = pv.Cube(
            center=(
                s_grid_config["bounds"][0],
                s_grid_config["bounds"][0],
                s_grid_config["bounds"][0],
            )
        )
        plotter.set_focus(cube.center)
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 0.1
        plotter.camera.position = (0.1 + 0.2, 0.35 + 0.2, 0.1 + 0.2)
        plotter.camera.up = (0.0, 0.0, 1.0)

        if save_eps:
            plotter.save_graphic(save_path + f"/screenshot_{k}.eps")

        image = plotter.screenshot()
        return Image.fromarray(np.asarray(image))
    finally:
        plotter.close()


@ray.remote
def one_step_voxel_render_for_cutting_process(
    k: int,
    s_grid_config: dict,
    sample_images: np.ndarray,
    action: np.ndarray,
    action_table: Any,
    save_path: str,
    save_eps: bool = False,
) -> Image.Image:
    return one_step_voxel_render_for_cutting_process_local(
        k=k,
        s_grid_config=s_grid_config,
        sample_images=sample_images,
        action=action,
        action_table=action_table,
        save_path=save_path,
        save_eps=save_eps,
    )
