from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pyvista as pv

from denoising_diffusion_pytorch.env.mesh_components import MeshComponent, MeshComponentSet
from denoising_diffusion_pytorch.utils.os_utils import load_yaml  # 既存を流用

@dataclass(frozen=True)
class DatasetLayout:
    dataset_dir: Path

    @property
    def config_path(self) -> Path:
        return self.dataset_dir / "generated_configs_w_multi_color.yaml"

    def component_stl_path(self, component_name: str) -> Path:
        # 既存の命名規約をここに閉じ込める
        return self.dataset_dir / "blend" / f"Boxy_0_cut0_{component_name}.stl"


class MeshComponentRepository:
    def load_from_dataset_dir(self, dataset_dir: str | Path) -> MeshComponentSet:
        layout      = DatasetLayout(Path(dataset_dir))
        mesh_config = load_yaml(str(layout.config_path))
        inner_box   = mesh_config.get("inner_box", {})

        comps = {}
        for key in inner_box:
            if "Component" not in key:
                continue

            stl_path = layout.component_stl_path(key)
            mesh = pv.read(str(stl_path))
            color = tuple(inner_box[key]["color"])  # (r,g,b)

            comps[key] = MeshComponent(name=key, mesh=mesh, color=color)

        return MeshComponentSet(comps)
