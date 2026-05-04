from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, List, Optional

import hydra
import numpy as np
import ray
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import dismantling_env
from denoising_diffusion_pytorch.utils.os_utils import (
    create_folder,
    get_folder_name,
    pickle_utils,
)
from denoising_diffusion_pytorch.utils.pil_utils import (
    numpy_to_pil,
    pil_image_load_to_numpy,
    pil_image_save_from_numpy,
)
from denoising_diffusion_pytorch.utils.voxel_render import pv_voxel_render_parallel


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, ListConfig):
        return list(value)

    if isinstance(value, list):
        return value

    return [value]


def _select_by_indices(items: List[str], indices: Optional[Iterable[int]]) -> List[str]:
    if indices is None:
        return items

    return [items[int(i)] for i in indices]


def _resolve_episode_indices(
    episodes: List[str],
    episode_indices: Optional[Iterable[int]],
    max_episodes: Optional[int],
) -> List[int]:
    if episode_indices is not None:
        return [int(i) for i in episode_indices]

    if max_episodes is None:
        return list(range(len(episodes)))

    return list(range(min(int(max_episodes), len(episodes))))


def _build_root_folders(cfg: DictConfig) -> List[Path]:
    root_folder = Path(str(cfg.visualization.root_folder)).expanduser()
    tags = _to_list(cfg.visualization.get("tags", None))

    if not tags:
        return [root_folder]

    return [root_folder / str(tag) for tag in tags]


def _build_grid_config(cfg: DictConfig) -> dict:
    bounds = tuple(float(x) for x in cfg.visualization.bounds)
    dim_3d = int(cfg.visualization.dim_3d)

    return {
        "bounds": bounds,
        "side_length": dim_3d,
    }


def _init_ray(cfg: DictConfig) -> None:
    ray_cfg = cfg.visualization.ray

    if not bool(ray_cfg.enabled):
        return

    kwargs = {
        "log_to_driver": bool(ray_cfg.log_to_driver),
        "ignore_reinit_error": True,
    }

    if ray_cfg.get("num_cpus", None) is not None:
        kwargs["num_cpus"] = int(ray_cfg.num_cpus)

    ray.init(**kwargs)


def _shutdown_ray(cfg: DictConfig) -> None:
    ray_cfg = cfg.visualization.ray

    if bool(ray_cfg.enabled) and bool(ray_cfg.shutdown_on_finish):
        ray.shutdown()


def _make_paper_interleaved_frames(
    oracle_2d_map: np.ndarray,
    cutting_process_2d_map: np.ndarray,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    元コードの paper 用処理を関数化したもの。

    - 最初に oracle*0 を追加
    - 各フレームを2回ずつ並べる
    - action を1つ先へ roll
    - 最後のフレームを1つ追加
    """
    cutting_process_2d_map_base = np.concatenate(
        [oracle_2d_map[None, :, :, :] * 0.0, cutting_process_2d_map],
        axis=0,
    )

    action_base = np.concatenate(
        [np.asarray([0]), action],
        axis=0,
    )

    step_num, width, _, channel = cutting_process_2d_map_base.shape

    interleaved_maps = np.empty(
        (int(step_num * 2), width, width, channel),
        dtype=cutting_process_2d_map_base.dtype,
    )
    interleaved_maps[0::2] = cutting_process_2d_map_base
    interleaved_maps[1::2] = cutting_process_2d_map_base.copy()

    interleaved_actions = np.empty(
        (int(step_num * 2),),
        dtype=action_base.dtype,
    )
    interleaved_actions[0::2] = action_base
    interleaved_actions[1::2] = action_base.copy()

    interleaved_actions = np.roll(interleaved_actions, -1)

    interleaved_maps = np.concatenate(
        [interleaved_maps, interleaved_maps[-1:]],
        axis=0,
    )
    interleaved_actions = np.concatenate(
        [interleaved_actions, interleaved_actions[-1:]],
        axis=0,
    )

    return interleaved_maps, interleaved_actions


def _resize_cutting_process_maps(
    cutting_process_2d_map: np.ndarray,
    dim_2d: int,
) -> np.ndarray:
    resized = []

    for frame_idx in range(cutting_process_2d_map.shape[0]):
        pil_image = numpy_to_pil(cutting_process_2d_map[frame_idx])
        resized_image = pil_image.resize((dim_2d, dim_2d))
        resized.append(np.asarray(resized_image) / 255.0)

    return np.asarray(resized)


def _make_remaining_voxel_maps(
    oracle_2d_map: np.ndarray,
    cutting_process_2d_map: np.ndarray,
) -> np.ndarray:
    return np.where(
        (cutting_process_2d_map >= oracle_2d_map - 0.05)
        & (cutting_process_2d_map <= oracle_2d_map + 0.05),
        np.asarray([0.0, 0.0, 0.0]),
        oracle_2d_map,
    ) * 255.0


def _mask_over_cutting_voxels(
    oracle_2d_map: np.ndarray,
    cutting_process_2d_map_flip: np.ndarray,
) -> np.ndarray:
    masked = cutting_process_2d_map_flip.copy()

    for frame_idx in range(masked.shape[0]):
        over_cutting_voxels = (
            np.all(oracle_2d_map == np.asarray([0.2, 0.8, 0.8]), axis=-1)
            & np.all(masked[frame_idx] / 255.0 == [0, 0, 0], axis=-1)
        )

        frame = (masked[frame_idx] / 255.0).copy()
        frame[over_cutting_voxels] = np.asarray([148 / 255, 0.0, 211 / 255])
        masked[frame_idx] = frame * 255.0

    return masked


def render_episode(
    *,
    data_folder: Path,
    cfg: DictConfig,
    s_grid_config: dict,
    action_table: np.ndarray,
) -> None:
    dim_2d = int(cfg.visualization.dim_2d)
    dim_3d = int(cfg.visualization.dim_3d)

    save_prefix = str(cfg.visualization.save_prefix)
    save_name = f"dim_{dim_3d}_{save_prefix}"

    save_folder = data_folder / str(cfg.visualization.save_subdir)
    create_folder(str(save_folder))

    rollout_path = data_folder / str(cfg.visualization.rollout_filename)
    oracle_obs_path = data_folder / str(cfg.visualization.oracle_obs_filename)

    print(f"load_data: {data_folder}")

    load_data = pickle_utils().load(load_path=str(rollout_path))

    oracle_2d_map = pil_image_load_to_numpy(
        str(oracle_obs_path),
        resize=(dim_2d, dim_2d),
    )

    cutting_process_2d_map = load_data["observations"]
    action = load_data["actions"]

    if bool(cfg.visualization.paper_frame_interleave):
        cutting_process_2d_map, action = _make_paper_interleaved_frames(
            oracle_2d_map=oracle_2d_map,
            cutting_process_2d_map=cutting_process_2d_map,
            action=action,
        )

    print(f"action_idx: {action}")

    cutting_process_2d_map = _resize_cutting_process_maps(
        cutting_process_2d_map=cutting_process_2d_map,
        dim_2d=dim_2d,
    )

    cutting_process_2d_map_flip = _make_remaining_voxel_maps(
        oracle_2d_map=oracle_2d_map,
        cutting_process_2d_map=cutting_process_2d_map,
    )

    pil_image_save_from_numpy(
        cutting_process_2d_map_flip[-1] / 255.0,
        str(data_folder / "last_remain_voxels.png"),
    )

    cutting_process_2d_map_flip = _mask_over_cutting_voxels(
        oracle_2d_map=oracle_2d_map,
        cutting_process_2d_map_flip=cutting_process_2d_map_flip,
    )

    pil_image_save_from_numpy(
        cutting_process_2d_map_flip[-1] / 255.0,
        str(data_folder / "last_remain_voxels_w_ocv_masked.png"),
    )

    start_time = perf_counter()

    pv_voxel_render_parallel().render_cutting_process_v3(
        save_path=str(save_folder),
        s_grind_config=s_grid_config,
        action=action,
        action_table=action_table,
        sample_images=cutting_process_2d_map_flip,
        save_tag=save_name,
        use_ray=bool(cfg.visualization.renderer.use_ray),
        max_in_flight=cfg.visualization.ray.get("max_in_flight", None),
        save_eps=bool(cfg.visualization.renderer.save_eps),
    )

    elapsed = perf_counter() - start_time
    print(f"[render] {data_folder} finished in {elapsed:.2f} sec")


def render_root_folder(
    *,
    root_folder: Path,
    cfg: DictConfig,
    s_grid_config: dict,
    action_table: np.ndarray,
) -> None:
    if not root_folder.exists():
        raise FileNotFoundError(f"root_folder does not exist: {root_folder}")

    model_type_folders = get_folder_name(str(root_folder))

    model_type_indices = cfg.visualization.get("model_type_indices", None)
    model_type_folders = _select_by_indices(
        items=model_type_folders,
        indices=model_type_indices,
    )

    for model_type_folder in model_type_folders:
        episodes_folder = root_folder / model_type_folder
        episodes = get_folder_name(str(episodes_folder))

        episode_indices = cfg.visualization.get("episode_indices", None)
        max_episodes = cfg.visualization.get("max_episodes", None)

        target_episode_indices = _resolve_episode_indices(
            episodes=episodes,
            episode_indices=episode_indices,
            max_episodes=max_episodes,
        )

        for episode_idx in tqdm(target_episode_indices):
            data_folder = episodes_folder / episodes[episode_idx]

            render_episode(
                data_folder=data_folder,
                cfg=cfg,
                s_grid_config=s_grid_config,
                action_table=action_table,
            )


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg.visualization))

    _init_ray(cfg)

    try:
        s_grid_config = _build_grid_config(cfg)

        # __new__ を使って __init__ を呼ばずにインスタンスを作成
        cutting_env = object.__new__(dismantling_env)
        action_table = cutting_env.get_action_table(s_grid_config)

        for root_folder in _build_root_folders(cfg):
            print(f"root_folder: {root_folder}")

            render_root_folder(
                root_folder=root_folder,
                cfg=cfg,
                s_grid_config=s_grid_config,
                action_table=action_table,
            )
    finally:
        _shutdown_ray(cfg)


if __name__ == "__main__":
    main()
