from __future__ import annotations

from pathlib import Path
import numpy as np

from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import dismantling_env


def main():
    grid_config = {
        "bounds"     : (-0.05, 0.05, -0.05, 0.05, -0.05, 0.05),
        "side_length": 16,
    }
    mesh_components = {
        # TODO: env が必要とするメッシュ情報（元 eval から持ってくる）
    }

    # env を作る
    env = dismantling_env(
        grid_config     = grid_config,
        mesh_components = mesh_components,
        # 他に必要なら引数追加（envの__init__に合わせる）
    )

    obs, reward, done, info = env.reset()

    print("=== reset ===")
    print("obs keys:", obs.keys())
    print("info keys:", info.keys())
    print("oracle_obs keys:", info["oracle_obs"].keys())

    # まずは action_idx を手で指定
    action_idx = 0

    # partial_obs を入れるならここ（まずは空でOK）
    partial_obs = {}

    obs, reward, done, info = env.step(action_idx, partial_obs=partial_obs)

    print("\n=== step ===")
    print("reward:", reward, "done:", done)
    print("obs.sequential_obs keys:", obs["sequential_obs"].keys())
    print("oracle z shape:", np.asarray(info["oracle_obs"]["z"]).shape)
    print("seq z shape:", np.asarray(obs["sequential_obs"]["z"]).shape)
    print("seq z min/max:",
          float(np.min(obs["sequential_obs"]["z"])),
          float(np.max(obs["sequential_obs"]["z"])))



if __name__ == "__main__":
    main()
