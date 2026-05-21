from dataclasses import dataclass
from pathlib import Path
from omegaconf import DictConfig

from app.infra.boxy_generated_mesh_repository import BoxyGeneratedMeshRepository
from app.infra.product_parts_mesh_repository import ProductPartsMeshRepository


@dataclass
class MeshComponentFactory:
    def create(self, case_spec: DictConfig):
        layout = str(getattr(case_spec, "dataset_format"))

        # import ipdb; ipdb.set_trace()

        if layout == "boxy_generated_yaml":
            repo = BoxyGeneratedMeshRepository()
            return repo.load_from_dataset_dir(Path(case_spec.dataset_dir))

        if layout == "product_parts_yaml":
            repo = ProductPartsMeshRepository()
            return repo.load_from_dataset_dir(
                dataset_dir=case_spec.dataset_dir,
                model_config=case_spec.model_config,
            )

        raise ValueError(f"Unsupported dataset_layout: {layout}")
