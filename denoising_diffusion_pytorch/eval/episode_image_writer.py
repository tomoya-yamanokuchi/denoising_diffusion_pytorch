from pathlib import Path
from denoising_diffusion_pytorch.utils.pil_utils import pil_image_save_from_numpy



class EpisodeImageWriter:
    def __init__(self, artifact_episodic_root: Path):
        self._root = artifact_episodic_root

    def save_oracle_obs(self, info: dict):
        pil_image_save_from_numpy(info["oracle_obs"]["x"],f"{self._root}/oracle_obs_cast_x_axis{0}.png")
        pil_image_save_from_numpy(info["oracle_obs"]["y"],f"{self._root}/oracle_obs_cast_y_axis{0}.png")
        pil_image_save_from_numpy(info["oracle_obs"]["z"],f"{self._root}/oracle_obs_cast_z_axis{0}.png")

    def save_seq_obs(self, step_idx: int, seq_obs: dict) -> None:
        pil_image_save_from_numpy(seq_obs["x"], str(self._root / f"{step_idx}_seq_obs_cast_x_axis{step_idx}_0.png"))
        pil_image_save_from_numpy(seq_obs["y"], str(self._root / f"{step_idx}_seq_obs_cast_y_axis{step_idx}_0.png"))
        pil_image_save_from_numpy(seq_obs["z"], str(self._root / f"{step_idx}_seq_obs_cast_z_axis{step_idx}_0.png"))

    def save_ensemble_image(self, step_idx: int, ensemble_image: dict) -> None:
        pil_image_save_from_numpy(ensemble_image["x"], str(self._root / f"{step_idx}_ensemble_x_axis{step_idx}_0.png"))
        pil_image_save_from_numpy(ensemble_image["y"], str(self._root / f"{step_idx}_ensemble_y_axis{step_idx}_0.png"))
        pil_image_save_from_numpy(ensemble_image["z"], str(self._root / f"{step_idx}_ensemble_z_axis{step_idx}_0.png"))
