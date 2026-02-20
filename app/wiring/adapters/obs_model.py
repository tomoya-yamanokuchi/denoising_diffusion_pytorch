from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import copy

from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler


@dataclass
class VoxelObsModel:
    """
    Compatibility wrapper around voxel_cut_handler.

    Adds:
      - reset_episode(): restore mutable internal state (colors) to the initial snapshot

    Guarantees (for cutting_surface_planner_v9):
      - get_2d_image(axis)
      - cast_2d_image_to_box_color(img, config)
      - voxel_hander attribute passthrough

    責務：
      - 1) メッシュをボクセルへキャストして内部状態を作る（初期化）
        2) “現在のボクセル色状態”を保持する
        3) 現状態から 2D 観測（スライス画像）を生成する
        4) 2D 観測（ミニバッチ）で内部状態（colors）を更新する
    """
    handler: voxel_cut_handler  # voxel_cut_handler

    def __post_init__(self) -> None:
        if not hasattr(self.handler, "colors"):
            raise AttributeError("handler must have `.colors` (voxel_cut_handler expected)")
        if not hasattr(self.handler, "voxel_hander"):
            raise AttributeError("handler must have `.voxel_hander`")

        # Snapshot mutable state
        self._initial_colors = copy.deepcopy(self.handler.colors)


    @property
    def voxel_hander(self) -> Any:
        # policy accesses obs_model.voxel_hander directly
        return self.handler.voxel_hander

    def reset_episode(self) -> None:
        # Restore the mutable voxel color state
        self.handler.colors = copy.deepcopy(self._initial_colors)

    # ---- Methods used by policy (explicit for clarity/compatibility) ----
    def get_2d_image(self, axis: str):
        return self.handler.get_2d_image(axis=axis)

    def cast_2d_image_to_box_color(self, img, config: Dict[str, Any]):
        """
        cutting_surface_planner_v9 calls this method on obs_model.
        Some implementations may expose it on handler; otherwise delegate to voxel_hander.
        """
        if hasattr(self.handler, "cast_2d_image_to_box_color"):
            return self.handler.cast_2d_image_to_box_color(img=img, config=config)

        # Fallback: voxel_hander API (seen in voxel_cut_handler.update_color)
        axis = config.get("axis", None)
        if axis is None:
            raise ValueError("config must contain 'axis'")

        # voxel_hander.cast_2d_image_to_box_color signature is (image, axis, colors)
        # We update handler.colors to keep the state consistent.
        self.handler.colors = self.handler.voxel_hander.cast_2d_image_to_box_color(
            image=img, axis=axis, colors=self.handler.colors
        )
        return self.handler.colors

    # ---- Transparent passthrough for any other legacy calls ----
    def __getattr__(self, name: str) -> Any:
        return getattr(self.handler, name)
