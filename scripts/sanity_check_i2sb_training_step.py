from __future__ import annotations

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from denoising_diffusion_pytorch.data_loader.i2sb_cond_image_data_loader \
    import I2SBCondImageDataset

from denoising_diffusion_pytorch.models.i2sb_diffusion \
    import I2SBDiffusion

from denoising_diffusion_pytorch.models.experimental.dit \
    import DiT


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
        ) ** 2
    )

    return betas.numpy()


@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):

    cfg = cfg.usecase

    device = torch.device(
        cfg.device
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device =", device)

    # ============================================================
    # Dataset
    # ============================================================

    dataset = I2SBCondImageDataset(
        cfg=cfg,
        image_size=cfg.dataset.image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    data = next(iter(loader))

    x0 = data["x0"].to(device)
    x1 = data["x1"].to(device)
    cond = data["cond"].to(device)
    mask = data["mask"].to(device)

    print("\n--- input shapes ---")
    print("x0   =", x0.shape)
    print("x1   =", x1.shape)
    print("cond =", cond.shape)
    print("mask =", mask.shape)

    # ============================================================
    # Network
    # ============================================================

    network_cfg = cfg.inferencer.network

    model = DiT(
        dim=network_cfg.dim,
        depth=network_cfg.depth,
        heads=network_cfg.heads,
        dim_head=network_cfg.dim_head,
        patch_size=network_cfg.patch_size,
    ).to(device)

    model.train()

    # ============================================================
    # I2SB diffusion
    # ============================================================

    interval = 1000
    beta_max = 0.3

    betas = make_beta_schedule(
        n_timestep=interval,
        linear_end=beta_max / interval,
    )

    betas = np.concatenate([
        betas[: interval // 2],
        np.flip(
            betas[: interval // 2]
        ),
    ])

    diffusion = I2SBDiffusion(
        betas=betas,
        device=device,
    )

    # ============================================================
    # Sample timestep
    # ============================================================

    batch_size = x0.shape[0]

    step = torch.randint(
        low=0,
        high=len(diffusion.betas),
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )

    print("\nstep =", step)

    # ============================================================
    # q(xt | x0, x1)
    # ============================================================

    xt = diffusion.q_sample(
        step=step,
        x0=x0,
        x1=x1,
        ot_ode=False,
    )

    # ============================================================
    # Eq. (12) target
    # ============================================================

    label = diffusion.compute_label(
        step=step,
        x0=x0,
        xt=xt,
    )

    # ============================================================
    # Network prediction
    # ============================================================

    pred = model(
        xt,
        step,
        None,
        cond,
        mask,
    )

    print("\n--- output shapes ---")
    print("xt    =", xt.shape)
    print("label =", label.shape)
    print("pred  =", pred.shape)

    assert xt.shape == x0.shape
    assert label.shape == x0.shape
    assert pred.shape == x0.shape

    # ============================================================
    # Masked I2SB loss
    # ============================================================

    pred_masked = mask * pred
    label_masked = mask * label

    loss = torch.nn.functional.mse_loss(
        pred_masked,
        label_masked,
    )

    print("\nloss =", loss.item())

    assert torch.isfinite(loss), \
        "Loss is NaN or Inf."

    # ============================================================
    # Backward
    # ============================================================

    model.zero_grad()

    loss.backward()

    # ============================================================
    # Gradient sanity check
    # ============================================================

    grad_norm_sq = 0.0
    grad_count = 0

    for param in model.parameters():

        if param.grad is None:
            continue

        grad_count += 1

        grad_norm_sq += (
            param.grad.detach()
            .float()
            .norm()
            .item()
            ** 2
        )

    grad_norm = grad_norm_sq ** 0.5

    print(
        "parameters with grad =",
        grad_count,
    )

    print(
        "total grad norm =",
        grad_norm,
    )

    assert grad_count > 0, \
        "No gradients were produced."

    assert np.isfinite(grad_norm), \
        "Gradient norm is NaN or Inf."

    assert grad_norm > 0.0, \
        "Gradient norm is zero."

    print(
        "\nI2SB training step sanity check PASSED."
    )


if __name__ == "__main__":
    main()
