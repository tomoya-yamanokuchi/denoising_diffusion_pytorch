"""
Sanity check for I2SB reverse sampling with trajectory visualization.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from torchvision.utils import save_image

from denoising_diffusion_pytorch.data_loader.i2sb_cond_image_data_loader import (
    I2SBCondImageDataset,
)
from denoising_diffusion_pytorch.models.experimental.dit_i2sb import DiT
from denoising_diffusion_pytorch.models.i2sb_diffusion import I2SBDiffusion


def make_beta_schedule(
    n_timestep=1000,
    linear_start=1e-4,
    linear_end=2e-2,
):
    betas = (
        torch.linspace(
            linear_start ** 0.5,
            linear_end ** 0.5,
            n_timestep,
            dtype=torch.float64,
        )
        ** 2
    )
    return betas.numpy()


def make_sampling_steps(interval, nfe):
    steps = np.linspace(
        0,
        interval - 1,
        nfe + 1,
        dtype=int,
    )
    steps = np.unique(steps)
    if steps[0] != 0:
        steps = np.insert(steps, 0, 0)
    if steps[-1] != interval - 1:
        steps = np.append(steps, interval - 1)
    return steps.tolist()


def to_vis(x):
    x = x.detach().cpu()
    x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    return x.permute(1, 2, 0).numpy()


def save_basic_images(output_dir, x0, x1, cond, recon):
    save_image(((x0 + 1.0) / 2.0).clamp(0.0, 1.0), output_dir / "x0_gt.png")
    save_image(((x1 + 1.0) / 2.0).clamp(0.0, 1.0), output_dir / "x1_input.png")
    save_image(((cond + 1.0) / 2.0).clamp(0.0, 1.0), output_dir / "condition.png")
    save_image(((recon + 1.0) / 2.0).clamp(0.0, 1.0), output_dir / "reconstruction.png")


def select_visualization_indices(length, num_vis):
    num_vis = min(num_vis, length)
    indices = np.linspace(0, length - 1, num_vis, dtype=int)
    return indices[::-1]


def visualize_reverse_trajectory(xs, logged_steps, output_dir, num_vis=6):
    indices = select_visualization_indices(xs.shape[1], num_vis)
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        ax.imshow(to_vis(xs[0, idx]))
        ax.set_title(f"Xt @ t={logged_steps[idx]}")
        ax.axis("off")

    plt.tight_layout()
    output_path = output_dir / "reverse_trajectory.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def visualize_pred_x0_trajectory(pred_x0s, logged_steps, output_dir, num_vis=6):
    indices = select_visualization_indices(pred_x0s.shape[1], num_vis)
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        ax.imshow(to_vis(pred_x0s[0, idx]))
        ax.set_title(f"pred X0 @ t={logged_steps[idx]}")
        ax.axis("off")

    plt.tight_layout()
    output_path = output_dir / "pred_x0_trajectory.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def visualize_combined_trajectory(xs, pred_x0s, logged_steps, output_dir, num_vis=6):
    indices = select_visualization_indices(xs.shape[1], num_vis)
    fig, axes = plt.subplots(2, len(indices), figsize=(4 * len(indices), 8))

    if len(indices) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, idx in enumerate(indices):
        step_value = logged_steps[idx]

        axes[0, col].imshow(to_vis(xs[0, idx]))
        axes[0, col].set_title(f"Xt @ t={step_value}")
        axes[0, col].axis("off")

        axes[1, col].imshow(to_vis(pred_x0s[0, idx]))
        axes[1, col].set_title(f"pred X0 @ t={step_value}")
        axes[1, col].axis("off")

    plt.tight_layout()
    output_path = output_dir / "reverse_and_pred_x0_trajectory.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    cfg = cfg.usecase

    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu"
    )
    print("device =", device)

    # ------------------------------------------------------------
    # Sanity-check settings
    # ------------------------------------------------------------
    sample_index = 0
    nfe = 20
    num_vis = 6

    # Update this path if your run directory differs.
    checkpoint_path = Path(
            # "/path/to/model-500.pt"
            # "/home/dev/workspace/dataset/nedo_dismantling_log/train/i2sb/dit_i2sb_D128_T1000_B1.0_complex_2d_sheetsander_20260922_142732/model-500.pt"
            "/home/dev/workspace/dataset/nedo_dismantling_log/train/i2sb/dit_i2sb_D128_T1000_B1.0_complex_2d_sheetsander_20260922_153416/model-20000.pt"
        )

    # output_dir = Path("outputs/i2sb_reverse_sanity")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(
        f"outputs/i2sb_reverse_sampling_{timestamp}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    dataset = I2SBCondImageDataset(
        cfg=cfg,
        image_size=cfg.dataset.image_size,
    )
    sample = dataset[sample_index]

    x0 = sample["x0"].unsqueeze(0).to(device)
    x1 = sample["x1"].unsqueeze(0).to(device)
    cond = sample["cond"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)

    print("\n--- sample ---")
    print("sample_index =", sample_index)
    print("x0_path =", sample["x0_path"])
    print("x0 shape =", tuple(x0.shape))
    print("x1 shape =", tuple(x1.shape))
    print("cond shape =", tuple(cond.shape))

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    net_cfg = cfg.inferencer.network
    model = DiT(
        dim=net_cfg.dim,
        depth=net_cfg.depth,
        heads=net_cfg.heads,
        dim_head=net_cfg.dim_head,
        patch_size=net_cfg.patch_size,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain key `model`.")

    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("\nloaded checkpoint:", checkpoint_path)
    if "step" in checkpoint:
        print("checkpoint step =", checkpoint["step"])

    # ------------------------------------------------------------
    # I2SB diffusion
    # ------------------------------------------------------------
    interval = int(cfg.inferencer.diffusion.interval)
    beta_max = float(cfg.inferencer.diffusion.beta_max)

    betas = make_beta_schedule(
        n_timestep=interval,
        linear_end=beta_max / interval,
    )
    betas = np.concatenate(
        [
            betas[: interval // 2],
            np.flip(betas[: interval // 2]),
        ]
    )

    diffusion = I2SBDiffusion(
        betas=betas,
        device=device,
    )

    # ------------------------------------------------------------
    # Sampling steps
    # ------------------------------------------------------------
    steps = make_sampling_steps(
        interval=interval,
        nfe=nfe,
    )

    print("\nsampling steps:")
    print(steps)

    # ------------------------------------------------------------
    # Network -> predicted X0
    # ------------------------------------------------------------
    @torch.no_grad()
    def pred_x0_fn(xt, step_value):
        step = torch.full(
            (xt.shape[0],),
            int(step_value),
            device=device,
            dtype=torch.long,
        )

        net_out = model(
            xt,
            step,
            None,
            cond,
        )

        return diffusion.compute_pred_x0(
            step=step,
            xt=xt,
            net_out=net_out,
            clip_denoise=True,
        )

    # ------------------------------------------------------------
    # Reverse sampling
    # ------------------------------------------------------------
    with torch.no_grad():
        xs, pred_x0s = diffusion.ddpm_sampling(
            steps=steps,
            pred_x0_fn=pred_x0_fn,
            x1=x1,
            cond=cond,
            mask=mask,
            ot_ode=False,
            log_steps=steps,
            verbose=True,
        )

    print("\nxs shape =", xs.shape)
    print("pred_x0s shape =", pred_x0s.shape)

    if not torch.isfinite(xs).all():
        raise RuntimeError("Reverse trajectory contains NaN or Inf.")

    if not torch.isfinite(pred_x0s).all():
        raise RuntimeError("Predicted-X0 trajectory contains NaN or Inf.")

    # ddpm_sampling() stores states after each reverse transition and then
    # flips them to increasing-time order. Thus xs[:, 0] is t=0-side.
    logged_steps = steps[:-1]

    if xs.shape[1] != len(logged_steps):
        raise RuntimeError(
            f"Unexpected trajectory length: {xs.shape[1]} vs {len(logged_steps)}"
        )

    final_recon = xs[:, 0]

    mask_cpu = mask.detach().cpu()
    cond_cpu = cond.detach().cpu()

    final_recon = (
        (1.0 - mask_cpu) * cond_cpu
        + mask_cpu * final_recon
    )

    # ------------------------------------------------------------
    # Save basic images
    # ------------------------------------------------------------
    save_basic_images(
        output_dir=output_dir,
        x0=x0,
        x1=x1,
        cond=cond,
        recon=final_recon,
    )

    # ------------------------------------------------------------
    # Save trajectories
    # ------------------------------------------------------------
    reverse_path = visualize_reverse_trajectory(
        xs=xs,
        logged_steps=logged_steps,
        output_dir=output_dir,
        num_vis=num_vis,
    )

    pred_path = visualize_pred_x0_trajectory(
        pred_x0s=pred_x0s,
        logged_steps=logged_steps,
        output_dir=output_dir,
        num_vis=num_vis,
    )

    combined_path = visualize_combined_trajectory(
        xs=xs,
        pred_x0s=pred_x0s,
        logged_steps=logged_steps,
        output_dir=output_dir,
        num_vis=num_vis,
    )

    print("\nsaved:", output_dir.resolve())
    print("reverse trajectory:", reverse_path.resolve())
    print("pred X0 trajectory:", pred_path.resolve())
    print("combined trajectory:", combined_path.resolve())
    print("\nI2SB reverse sampling sanity check PASSED.")


if __name__ == "__main__":
    main()
