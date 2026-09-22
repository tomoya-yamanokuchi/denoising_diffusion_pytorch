from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from denoising_diffusion_pytorch.data_loader.i2sb_cond_image_data_loader \
    import I2SBCondImageDataset

from denoising_diffusion_pytorch.models.i2sb_diffusion \
    import I2SBDiffusion


def make_beta_schedule(
    n_timestep=1000,
    linear_start=1e-4,
    linear_end=2e-2,
):
    """
    Same beta schedule as the official NVlabs/I2SB implementation.
    """
    betas = (
        torch.linspace(
            linear_start ** 0.5,
            linear_end ** 0.5,
            n_timestep,
            dtype=torch.float64,
        ) ** 2
    )

    return betas.numpy()


def to_display_image(x: torch.Tensor):
    """
    [C, H, W], [-1, 1]
        ->
    [H, W, C], [0, 1]
    """
    x = x.detach().cpu()

    x = (
        x.clamp(-1.0, 1.0)
        + 1.0
    ) / 2.0

    return x.permute(1, 2, 0).numpy()


def to_display_mask(mask: torch.Tensor):
    """
    [1, H, W] -> [H, W]
    """
    return (
        mask
        .detach()
        .cpu()
        .squeeze(0)
        .numpy()
    )


def to_display_diff(
    x0: torch.Tensor,
    x1: torch.Tensor,
):
    """
    Visualize mean absolute RGB difference.

    Returns [H, W].
    """
    diff = torch.abs(
        x0 - x1
    ).mean(dim=0)

    diff = diff.detach().cpu()

    # Normalize only for visualization.
    max_val = diff.max()

    if max_val > 0:
        diff = diff / max_val

    return diff.numpy()


@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):

    cfg = cfg.usecase

    # ============================================================
    # Dataset
    # ============================================================

    dataset = I2SBCondImageDataset(
        cfg=cfg,
        image_size=cfg.dataset.image_size,
    )

    sample_index = 0
    sample = dataset[sample_index]

    x0 = sample["x0"].unsqueeze(0)
    x1 = sample["x1"].unsqueeze(0)
    cond = sample["cond"].unsqueeze(0)
    mask = sample["mask"].unsqueeze(0)

    print("--- sample ---")
    print("index   =", sample_index)
    print("x0_path =", sample["x0_path"])

    print(
        "x0 range =",
        x0.min().item(),
        x0.max().item(),
    )

    print(
        "x1 range =",
        x1.min().item(),
        x1.max().item(),
    )

    print(
        "cond range =",
        cond.min().item(),
        cond.max().item(),
    )

    print(
        "mask unique =",
        torch.unique(mask),
    )

    print(
        "hidden ratio =",
        mask.mean().item(),
    )

    # ============================================================
    # I2SB beta schedule
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
        device="cpu",
    )

    # ============================================================
    # Bridge samples
    # ============================================================

    steps = [
        0,
        250,
        500,
        750,
        999,
    ]

    xt_list = []

    for step_value in steps:

        step = torch.tensor(
            [step_value],
            dtype=torch.long,
        )

        xt = diffusion.q_sample(
            step=step,
            x0=x0,
            x1=x1,
            ot_ode=False,
        )

        xt_list.append(
            xt[0]
        )

    # ============================================================
    # Visualization
    # ============================================================

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(16, 12),
    )

    axes = axes.flatten()

    # ------------------------------------------------------------
    # Row 1: Dataset components
    # ------------------------------------------------------------

    axes[0].imshow(
        to_display_image(x0[0])
    )
    axes[0].set_title(
        "X0: full slice"
    )

    axes[1].imshow(
        to_display_image(x1[0])
    )
    axes[1].set_title(
        "X1: exterior only"
    )

    axes[2].imshow(
        to_display_image(cond[0])
    )
    axes[2].set_title(
        "Condition C: masked X0"
    )

    axes[3].imshow(
        to_display_mask(mask[0]),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[3].set_title(
        "Mask: 0=observed, 1=hidden"
    )

    # ------------------------------------------------------------
    # Row 2: difference + early bridge
    # ------------------------------------------------------------

    axes[4].imshow(
        to_display_diff(
            x0[0],
            x1[0],
        ),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[4].set_title(
        "|X0 - X1|"
    )

    axes[5].imshow(
        to_display_image(
            xt_list[0]
        )
    )
    axes[5].set_title(
        "Xt @ t=0"
    )

    axes[6].imshow(
        to_display_image(
            xt_list[1]
        )
    )
    axes[6].set_title(
        "Xt @ t=250"
    )

    axes[7].imshow(
        to_display_image(
            xt_list[2]
        )
    )
    axes[7].set_title(
        "Xt @ t=500"
    )

    # ------------------------------------------------------------
    # Row 3: later bridge
    # ------------------------------------------------------------

    axes[8].imshow(
        to_display_image(
            xt_list[3]
        )
    )
    axes[8].set_title(
        "Xt @ t=750"
    )

    axes[9].imshow(
        to_display_image(
            xt_list[4]
        )
    )
    axes[9].set_title(
        "Xt @ t=999"
    )

    # ------------------------------------------------------------
    # Optional: deterministic bridge mean
    # ------------------------------------------------------------

    mid_step = torch.tensor(
        [500],
        dtype=torch.long,
    )

    xt_mean = diffusion.q_sample(
        step=mid_step,
        x0=x0,
        x1=x1,
        ot_ode=True,
    )

    axes[10].imshow(
        to_display_image(
            xt_mean[0]
        )
    )
    axes[10].set_title(
        "Bridge mean @ t=500"
    )

    # ------------------------------------------------------------
    # Difference without normalization
    # ------------------------------------------------------------

    raw_diff = torch.abs(
        x0[0] - x1[0]
    ).mean(dim=0)

    axes[11].imshow(
        raw_diff.detach().cpu().numpy(),
        cmap="gray",
    )
    axes[11].set_title(
        "Raw mean |X0-X1|"
    )

    # ------------------------------------------------------------
    # Final formatting
    # ------------------------------------------------------------

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_path = Path(
        "i2sb_dataset_visual_sanity.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(
        "\nsaved:",
        output_path.resolve(),
    )


if __name__ == "__main__":
    main()
