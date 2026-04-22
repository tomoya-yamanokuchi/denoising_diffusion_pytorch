from pathlib import Path
from denoising_diffusion_pytorch.utils.os_utils import create_folder


class EpisodeArtifactManager:
    def __init__(self, artifact_episodic_root: Path):
        self._root = artifact_episodic_root

    @property
    def root(self) -> Path:
        return self._root

    def create_episodic_artifact_root_directory(self):
        create_folder(str(self._root))
