from dataclasses import dataclass
from pathlib import Path
from app.infra.mesh_component_repository import MeshComponentRepository
from denoising_diffusion_pytorch.env.mesh_components import MeshComponentSet


@dataclass
class MeshComponentFactory:
    def create(self, dataset_dir: str):
        repo = MeshComponentRepository()
        return repo.load_from_dataset_dir(Path(dataset_dir))
