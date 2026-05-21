from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml

from denoising_diffusion_pytorch.utils.benchmark_model_utils import get_benchmark_model
from denoising_diffusion_pytorch.env.mesh_components import MeshComponent, MeshComponentSet


@dataclass
class ProductPartsMeshRepository:
    def load_from_dataset_dir(
        self,
        dataset_dir: str | Path,
        model_config: str,
    ) -> MeshComponentSet:
        dataset_path = Path(dataset_dir)

        config_path = dataset_path / model_config
        with open(config_path, encoding="utf-8") as f:
            model_config_dict = yaml.safe_load(f)

        # get_benchmark_model は内部で dataset_path + relative_path を使うため、
        # trailing slash を付けて旧実装と同じ挙動に寄せる
        legacy_components = get_benchmark_model(
            dataset_path=str(dataset_path) + "/",
            model_config=model_config_dict,
        )

        components = {}
        for name, item in legacy_components.items():
            components[name] = MeshComponent(
                name=name,
                mesh=item["mesh"],
                color=tuple(item["color"]),
            )

        return MeshComponentSet(components)
