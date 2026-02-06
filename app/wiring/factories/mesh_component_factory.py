from dataclasses import dataclass
from pathlib import Path
from app.infra.mesh_component_repository import MeshComponentRepository
from denoising_diffusion_pytorch.env.mesh_components import MeshComponentSet


@dataclass
class MeshComponentFactory:
    repo: "MeshComponentRepository"

    def create(self, dataset_dir: str | Path):
        return self.repo.load_from_dataset_dir(Path(dataset_dir))
