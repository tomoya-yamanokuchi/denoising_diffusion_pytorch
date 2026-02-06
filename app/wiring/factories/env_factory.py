from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import dismantling_env


@dataclass
class EnvFactory:
    grid_key: str = "env.grid"          # cfg.env.grid.bounds / side_length を想定

    def create(self, cfg: DictConfig, mesh_components) -> "Envs":
        grid_cfg = OmegaConf.select(cfg, self.grid_key)
        if grid_cfg is None:
            raise KeyError(f"Missing cfg key: {self.grid_key}")

        # 必須キー検証（今回の KeyError をここで潰す）
        if "side_length" not in grid_cfg:
            raise KeyError("Missing env.grid.side_length (required by dismantling_env)")
        if "bounds" not in grid_cfg:
            raise KeyError("Missing env.grid.bounds (required by dismantling_env)")


        import ipdb; ipdb.set_trace()
        eval   = dismantling_env(grid_config=grid_cfg, mesh_components=mesh_components)
        policy = dismantling_env(grid_config=grid_cfg, mesh_components=mesh_components)

        from app.wiring.types.envs import Envs
        return Envs(eval=eval, policy=policy)
