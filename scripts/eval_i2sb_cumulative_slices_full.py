"""
Evaluate cumulative slice observations for a trained multi-product I2SB model.

This version extends the basic cumulative-slice evaluation with:
1. common-hidden MAE / MSE / PSNR vs K
2. internal-structure visibility ratio vs K
3. newly revealed internal-structure ratio ΔV_K
4. prediction improvement ΔE_K
5. Pearson correlation between ΔV_K and ΔE_K
6. CSV + plots for paper-style analysis

Core idea
---------
For one fixed product/sample, evaluate:

    K=1 observed slice
    K=2 observed slices
    ...
    K=K_MAX observed slices

using ONE fixed cumulative slice order:

    O_1 subset O_2 subset ... subset O_K

Fairness choices
----------------
- Slice order is fixed once.
- Reverse-sampling RNG seed is reset to the same value for every K.
- Common-hidden metrics are evaluated on the SAME pixels for every K:
  pixels that remain hidden even at K=K_MAX.

Internal-structure visibility
-----------------------------
The script estimates an "internal structure mask" from the GT X0 image.
For this dataset, internal components are rendered in saturated colors
(blue/red/green/cyan/etc.), while background/exterior are near-white/gray.

By default, a pixel is considered "internal structure" if:

    max(R,G,B) - min(R,G,B) >= INTERNAL_COLOR_SPREAD_THRESHOLD

on the [0,1] RGB image.

This is intentionally simple and dataset-specific. If your rendering
palette changes, tune the threshold or replace `compute_internal_mask()`.

Outputs
-------
outputs/i2sb_cumulative_slice_<product>_<timestamp>/
    target.png
    x1.png
    internal_structure_mask.png
    slice_order.txt
    metrics.csv
    progression.png
    common_hidden_mae_vs_k.png
    internal_visibility_vs_k.png
    delta_visibility_vs_delta_error.png
    k_01/
        mask.png
        cond.png
        pred.png
        abs_error.png
    ...
"""

from __future__ import annotations

import csv
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


# ============================================================================
# USER SETTINGS
# ============================================================================

CHECKPOINT_PATH = Path(
    "/home/dev/workspace/dataset/nedo_dismantling_log/train/i2sb/dit_i2sb_D128_T1000_B1.0_complex_2d_multi_product_20260926_145317/model-50000.pt"
)

PRODUCT_NAME = "sheetsander"

# Index within the selected product.
PRODUCT_SAMPLE_INDEX = 0

# Cumulative observation count.
K_MAX = 10

# Fixed random order for selecting slices.
SLICE_ORDER_SEED = 12345

# Fixed stochastic reverse-sampling seed for every K.
SAMPLING_SEED = 54321

# Number of network evaluations during reverse sampling.
NFE = 20

# --------------------------------------------------------------------------
# Internal-structure mask settings
# --------------------------------------------------------------------------
# Internal structures in the current rendered dataset are saturated colors,
# while exterior/background are mostly near-gray / near-white.
#
# Pixel is internal if RGB channel spread is >= this threshold on [0,1].
INTERNAL_COLOR_SPREAD_THRESHOLD = 0.12

# Optional brightness floor. This helps suppress very dark background pixels
# if they ever appear in another visualization setting.
INTERNAL_MIN_BRIGHTNESS = 0.05


# ============================================================================
# I2SB helpers
# ============================================================================

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
    return betas.cpu().numpy()


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
        steps = np.append(
            steps,
            interval - 1,
        )

    return steps.tolist()


def to_01(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def to_vis(x):
    x = to_01(
        x.detach().cpu()
    )

    if x.ndim == 4:
        x = x[0]

    return x.permute(
        1,
        2,
        0,
    ).numpy()


# ============================================================================
# Checkpoint loading
# ============================================================================

def load_model_weights(
    model,
    checkpoint,
):
    """
    Prefer EMA weights when possible.
    Fall back to raw checkpoint["model"] otherwise.
    """
    ema_state = checkpoint.get(
        "ema",
        None,
    )

    if isinstance(
        ema_state,
        dict,
    ):
        prefix = "ema_model."

        ema_model_state = {
            key[len(prefix):]: value
            for key, value
            in ema_state.items()
            if key.startswith(prefix)
        }

        if ema_model_state:
            try:
                model.load_state_dict(
                    ema_model_state,
                    strict=True,
                )
                print(
                    "loaded EMA model weights"
                )
                return "ema"

            except RuntimeError as exc:
                print(
                    "[warning] EMA state was found but "
                    "could not be loaded directly:"
                )
                print(exc)
                print(
                    "falling back to checkpoint['model']"
                )

    if "model" not in checkpoint:
        raise KeyError(
            "Checkpoint contains neither a usable EMA model "
            "nor key `model`."
        )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    print(
        "loaded raw model weights"
    )

    return "model"


# ============================================================================
# Product/sample selection
# ============================================================================

def find_product_sample(
    dataset,
    product_name,
    product_sample_index,
):
    matching_indices = [
        idx
        for idx, record
        in enumerate(dataset.samples)
        if record["product_name"]
        == product_name
    ]

    if not matching_indices:
        available = sorted(
            {
                record["product_name"]
                for record
                in dataset.samples
            }
        )

        raise ValueError(
            f"Unknown product `{product_name}`. "
            f"Available products: {available}"
        )

    if not (
        0
        <= product_sample_index
        < len(matching_indices)
    ):
        raise IndexError(
            f"PRODUCT_SAMPLE_INDEX={product_sample_index} "
            f"is out of range for product `{product_name}` "
            f"with {len(matching_indices)} samples."
        )

    global_idx = matching_indices[
        product_sample_index
    ]

    record = dataset.samples[
        global_idx
    ]

    # Load directly so that __getitem__ does not generate an unrelated
    # random training mask.
    x0 = dataset._load_image(
        record["x0_path"]
    )

    x1 = dataset.x1_cache[
        product_name
    ].clone()

    return (
        global_idx,
        record,
        x0,
        x1,
    )


# ============================================================================
# Cumulative slice masks
# ============================================================================

def make_cumulative_slice_order(
    num_slices,
    k_max,
    seed,
):
    if k_max > num_slices:
        raise ValueError(
            f"K_MAX={k_max} exceeds num_slices={num_slices}."
        )

    rng = np.random.default_rng(
        seed
    )

    order = rng.permutation(
        num_slices
    ).astype(
        np.int64
    )

    return order


def make_slice_mask_from_indices(
    image_size,
    grid_rows,
    grid_cols,
    observed_indices,
):
    """
    Return [1,H,W]:
        0 = observed
        1 = hidden
    """
    H = int(
        image_size
    )
    W = int(
        image_size
    )

    mask = np.ones(
        (H, W),
        dtype=np.float32,
    )

    y_edges = np.linspace(
        0,
        H,
        grid_rows + 1,
        dtype=int,
    )

    x_edges = np.linspace(
        0,
        W,
        grid_cols + 1,
        dtype=int,
    )

    for idx in observed_indices:
        row = int(
            idx // grid_cols
        )

        col = int(
            idx % grid_cols
        )

        y0 = y_edges[row]
        y1 = y_edges[
            row + 1
        ]

        x0 = x_edges[col]
        x1 = x_edges[
            col + 1
        ]

        mask[
            y0:y1,
            x0:x1,
        ] = 0.0

    return torch.from_numpy(
        mask[
            None,
            :,
            :,
        ]
    )


# ============================================================================
# Internal-structure visibility
# ============================================================================

def compute_internal_mask(
    x0,
    color_spread_threshold=0.12,
    min_brightness=0.05,
):
    """
    Estimate internal-structure pixels from GT X0.

    x0:
        [1,3,H,W] or [3,H,W], range [-1,1]

    Returns:
        [1,1,H,W] float tensor in {0,1}

    Current dataset assumption:
    - exterior/background are close to achromatic (gray/white)
    - internal components use saturated colors
    """
    x = to_01(
        x0.detach().cpu()
    )

    if x.ndim == 3:
        x = x.unsqueeze(0)

    rgb_max = x.max(
        dim=1,
        keepdim=True,
    ).values

    rgb_min = x.min(
        dim=1,
        keepdim=True,
    ).values

    color_spread = (
        rgb_max - rgb_min
    )

    brightness = x.mean(
        dim=1,
        keepdim=True,
    )

    internal = (
        (
            color_spread
            >= color_spread_threshold
        )
        & (
            brightness
            >= min_brightness
        )
    ).float()

    return internal


def internal_visibility_ratio(
    internal_mask,
    observed_mask,
):
    """
    internal_mask:
        [1,1,H,W], 1 = GT internal structure pixel

    observed_mask:
        [1,1,H,W], 1 = observed pixel

    Returns:
        fraction of all GT internal-structure pixels that are visible.
    """
    denom = internal_mask.sum()

    if denom.item() <= 0:
        return 0.0

    visible = (
        internal_mask
        * observed_mask
    ).sum()

    return float(
        (
            visible
            / denom
        ).item()
    )


# ============================================================================
# Metrics
# ============================================================================

def masked_mae(
    pred,
    target,
    mask,
):
    """
    pred/target:
        [1,3,H,W]

    mask:
        [1,1,H,W], 1 = evaluate
    """
    expanded = mask.expand_as(
        pred
    )

    denom = expanded.sum().clamp_min(
        1.0
    )

    value = (
        (
            pred - target
        ).abs()
        * expanded
    ).sum() / denom

    return float(
        value.item()
    )


def masked_mse(
    pred,
    target,
    mask,
):
    expanded = mask.expand_as(
        pred
    )

    denom = expanded.sum().clamp_min(
        1.0
    )

    value = (
        (
            pred - target
        ).pow(2)
        * expanded
    ).sum() / denom

    return float(
        value.item()
    )


def psnr_from_mse(
    mse,
    data_range=2.0,
):
    if mse <= 0.0:
        return float("inf")

    return float(
        10.0
        * np.log10(
            data_range**2
            / mse
        )
    )


def pearson_correlation(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if len(x) < 2:
        return float("nan")

    if np.std(x) == 0.0:
        return float("nan")

    if np.std(y) == 0.0:
        return float("nan")

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


# ============================================================================
# Reverse sampling
# ============================================================================

@torch.no_grad()
def run_reverse_sampling(
    model,
    diffusion,
    x1,
    cond,
    mask,
    steps,
    device,
    seed,
):
    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    def pred_x0_fn(
        xt,
        step_value,
    ):
        step = torch.full(
            (
                xt.shape[0],
            ),
            int(
                step_value
            ),
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

    xs, _ = (
        diffusion.ddpm_sampling(
            steps=steps,
            pred_x0_fn=pred_x0_fn,
            x1=x1,
            cond=cond,
            mask=mask,
            ot_ode=False,
            log_steps=[0],
            verbose=False,
        )
    )

    recon = (
        xs[:, 0]
        .to(device)
    )

    # Exact data consistency at endpoint.
    recon = (
        (
            1.0 - mask
        )
        * cond
        + mask
        * recon
    )

    return recon


# ============================================================================
# Visualization
# ============================================================================

def save_progression_figure(
    output_dir,
    results,
    target,
):
    num_cols = len(
        results
    )

    fig, axes = plt.subplots(
        3,
        num_cols,
        figsize=(
            4 * num_cols,
            12,
        ),
    )

    if num_cols == 1:
        axes = np.asarray(
            axes
        ).reshape(
            3,
            1,
        )

    target_01 = to_01(
        target.detach().cpu()
    )

    for col, result in enumerate(
        results
    ):
        k = result["k"]

        cond = result["cond"]
        pred = result["pred"]

        abs_error = (
            to_01(
                pred.detach().cpu()
            )
            - target_01
        ).abs()

        axes[
            0,
            col,
        ].imshow(
            to_vis(
                cond
            )
        )
        axes[
            0,
            col,
        ].set_title(
            f"Condition K={k}\n"
            f"obs={result['observed_indices']}"
        )
        axes[
            0,
            col,
        ].axis(
            "off"
        )

        axes[
            1,
            col,
        ].imshow(
            to_vis(
                pred
            )
        )
        axes[
            1,
            col,
        ].set_title(
            f"Prediction K={k}\n"
            f"common MAE={result['common_hidden_mae']:.4f}\n"
            f"visibility={result['internal_visibility']:.3f}"
        )
        axes[
            1,
            col,
        ].axis(
            "off"
        )

        error_img = abs_error[
            0
        ].mean(
            dim=0
        ).numpy()

        axes[
            2,
            col,
        ].imshow(
            error_img,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axes[
            2,
            col,
        ].set_title(
            "Mean RGB abs. error"
        )
        axes[
            2,
            col,
        ].axis(
            "off"
        )

    plt.tight_layout()

    path = (
        output_dir
        / "progression.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    return path


def save_common_hidden_mae_plot(
    output_dir,
    results,
):
    ks = [
        r["k"]
        for r in results
    ]

    maes = [
        r["common_hidden_mae"]
        for r in results
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        ks,
        maes,
        marker="o",
    )

    plt.xlabel(
        "Observed slices K"
    )

    plt.ylabel(
        "Common-hidden MAE"
    )

    plt.title(
        "Prediction error vs cumulative observations"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    path = (
        output_dir
        / "common_hidden_mae_vs_k.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    return path


def save_internal_visibility_plot(
    output_dir,
    results,
):
    ks = [
        r["k"]
        for r in results
    ]

    vis = [
        r["internal_visibility"]
        for r in results
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        ks,
        vis,
        marker="o",
    )

    plt.xlabel(
        "Observed slices K"
    )

    plt.ylabel(
        "Internal structure visibility ratio"
    )

    plt.title(
        "Visible internal structure vs cumulative observations"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.tight_layout()

    path = (
        output_dir
        / "internal_visibility_vs_k.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    return path


def save_delta_scatter(
    output_dir,
    results,
    pearson_r,
):
    delta_v = [
        r["delta_visibility"]
        for r in results
        if r["k"] >= 2
    ]

    delta_e = [
        r["delta_error_reduction"]
        for r in results
        if r["k"] >= 2
    ]

    labels = [
        r["k"]
        for r in results
        if r["k"] >= 2
    ]

    plt.figure(
        figsize=(7, 6)
    )

    plt.scatter(
        delta_v,
        delta_e,
        s=50,
    )

    for x, y, k in zip(
        delta_v,
        delta_e,
        labels,
    ):
        plt.annotate(
            f"K={k}",
            (
                x,
                y,
            ),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.axvline(
        0.0,
        linewidth=1,
    )

    plt.xlabel(
        "Newly revealed internal structure ΔV"
    )

    plt.ylabel(
        "Common-hidden MAE reduction ΔE"
    )

    plt.title(
        "Information gained vs prediction improvement\n"
        f"Pearson r = {pearson_r:.4f}"
        if np.isfinite(
            pearson_r
        )
        else
        "Information gained vs prediction improvement\n"
        "Pearson r = NaN"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    path = (
        output_dir
        / "delta_visibility_vs_delta_error.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    return path


# ============================================================================
# Main
# ============================================================================

@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(
    cfg: DictConfig,
):
    cfg = cfg.usecase

    device = torch.device(
        cfg.device
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device =",
        device,
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    dataset = I2SBCondImageDataset(
        cfg=cfg,
        image_size=cfg.dataset.image_size,
    )

    if dataset.type != "slice":
        raise ValueError(
            "This cumulative-observation evaluation requires "
            f"`dataset.type: slice`, got `{dataset.type}`."
        )

    (
        global_idx,
        record,
        x0_single,
        x1_single,
    ) = find_product_sample(
        dataset=dataset,
        product_name=PRODUCT_NAME,
        product_sample_index=PRODUCT_SAMPLE_INDEX,
    )

    x0 = (
        x0_single
        .unsqueeze(0)
        .to(device)
    )

    x1 = (
        x1_single
        .unsqueeze(0)
        .to(device)
    )

    print(
        "\n--- evaluation sample ---"
    )

    print(
        "product =",
        PRODUCT_NAME,
    )

    print(
        "product sample index =",
        PRODUCT_SAMPLE_INDEX,
    )

    print(
        "global dataset index =",
        global_idx,
    )

    print(
        "x0_path =",
        record["x0_path"],
    )

    print(
        "x1_path =",
        record["x1_path"],
    )

    # ------------------------------------------------------------------------
    # Internal-structure mask
    # ------------------------------------------------------------------------

    internal_mask = compute_internal_mask(
        x0=x0,
        color_spread_threshold=
            INTERNAL_COLOR_SPREAD_THRESHOLD,
        min_brightness=
            INTERNAL_MIN_BRIGHTNESS,
    ).to(
        device
    )

    num_internal_pixels = int(
        internal_mask.sum().item()
    )

    print(
        "internal-structure pixels =",
        num_internal_pixels,
    )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    net_cfg = (
        cfg.inferencer.network
    )

    model = DiT(
        dim=net_cfg.dim,
        depth=net_cfg.depth,
        heads=net_cfg.heads,
        dim_head=net_cfg.dim_head,
        patch_size=net_cfg.patch_size,
    ).to(
        device
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    weight_source = (
        load_model_weights(
            model=model,
            checkpoint=checkpoint,
        )
    )

    model.eval()

    print(
        "checkpoint =",
        CHECKPOINT_PATH,
    )

    print(
        "checkpoint step =",
        checkpoint.get(
            "step",
            "unknown",
        ),
    )

    print(
        "weight source =",
        weight_source,
    )

    # ------------------------------------------------------------------------
    # I2SB process
    # ------------------------------------------------------------------------

    interval = int(
        cfg.inferencer.diffusion.interval
    )

    beta_max = float(
        cfg.inferencer.diffusion.beta_max
    )

    betas = make_beta_schedule(
        n_timestep=interval,
        linear_end=beta_max / interval,
    )

    betas = np.concatenate(
        [
            betas[
                : interval // 2
            ],
            np.flip(
                betas[
                    : interval // 2
                ]
            ),
        ]
    )

    diffusion = I2SBDiffusion(
        betas=betas,
        device=device,
    )

    steps = make_sampling_steps(
        interval=interval,
        nfe=NFE,
    )

    # ------------------------------------------------------------------------
    # Fixed cumulative slice order
    # ------------------------------------------------------------------------

    num_slices = (
        dataset.slice_grid_rows
        * dataset.slice_grid_cols
    )

    slice_order = (
        make_cumulative_slice_order(
            num_slices=num_slices,
            k_max=K_MAX,
            seed=SLICE_ORDER_SEED,
        )
    )

    selected_order = (
        slice_order[
            :K_MAX
        ]
        .tolist()
    )

    print(
        "\nfixed cumulative slice order =",
        selected_order,
    )

    final_mask_single = (
        make_slice_mask_from_indices(
            image_size=dataset.image_size,
            grid_rows=dataset.slice_grid_rows,
            grid_cols=dataset.slice_grid_cols,
            observed_indices=selected_order,
        )
    )

    common_hidden_mask = (
        final_mask_single
        .unsqueeze(0)
        .to(device)
    )

    # ------------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = Path(
        "outputs"
    ) / (
        f"i2sb_cumulative_slice_"
        f"{PRODUCT_NAME}_"
        f"{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_image(
        to_01(x0),
        output_dir
        / "target.png",
    )

    save_image(
        to_01(x1),
        output_dir
        / "x1.png",
    )

    save_image(
        internal_mask.detach().cpu(),
        output_dir
        / "internal_structure_mask.png",
    )

    (
        output_dir
        / "slice_order.txt"
    ).write_text(
        "\n".join(
            [
                f"product={PRODUCT_NAME}",
                f"product_sample_index={PRODUCT_SAMPLE_INDEX}",
                f"global_dataset_index={global_idx}",
                f"slice_order={selected_order}",
                f"slice_order_seed={SLICE_ORDER_SEED}",
                f"sampling_seed={SAMPLING_SEED}",
                f"nfe={NFE}",
                f"checkpoint={CHECKPOINT_PATH}",
                (
                    "internal_color_spread_threshold="
                    f"{INTERNAL_COLOR_SPREAD_THRESHOLD}"
                ),
                (
                    "internal_min_brightness="
                    f"{INTERNAL_MIN_BRIGHTNESS}"
                ),
                f"internal_pixel_count={num_internal_pixels}",
            ]
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # Evaluate K=1,...,K_MAX
    # ------------------------------------------------------------------------

    results = []

    previous_visibility = 0.0
    previous_common_mae = None

    for k in range(
        1,
        K_MAX + 1,
    ):
        observed_indices = (
            selected_order[
                :k
            ]
        )

        mask_single = (
            make_slice_mask_from_indices(
                image_size=dataset.image_size,
                grid_rows=dataset.slice_grid_rows,
                grid_cols=dataset.slice_grid_cols,
                observed_indices=observed_indices,
            )
        )

        mask = (
            mask_single
            .unsqueeze(0)
            .to(device)
        )

        observed_mask = (
            1.0 - mask
        )

        cond = (
            dataset
            ._make_condition(
                x0=x0_single,
                mask=mask_single,
            )
            .unsqueeze(0)
            .to(device)
        )

        pred = run_reverse_sampling(
            model=model,
            diffusion=diffusion,
            x1=x1,
            cond=cond,
            mask=mask,
            steps=steps,
            device=device,
            seed=SAMPLING_SEED,
        )

        current_hidden_mae = (
            masked_mae(
                pred=pred,
                target=x0,
                mask=mask,
            )
        )

        current_hidden_mse = (
            masked_mse(
                pred=pred,
                target=x0,
                mask=mask,
            )
        )

        common_hidden_mae = (
            masked_mae(
                pred=pred,
                target=x0,
                mask=common_hidden_mask,
            )
        )

        common_hidden_mse = (
            masked_mse(
                pred=pred,
                target=x0,
                mask=common_hidden_mask,
            )
        )

        common_hidden_psnr = (
            psnr_from_mse(
                common_hidden_mse
            )
        )

        visibility = (
            internal_visibility_ratio(
                internal_mask=
                    internal_mask,
                observed_mask=
                    observed_mask,
            )
        )

        delta_visibility = (
            visibility
            - previous_visibility
        )

        if previous_common_mae is None:
            delta_error_reduction = 0.0

        else:
            delta_error_reduction = (
                previous_common_mae
                - common_hidden_mae
            )

        k_dir = (
            output_dir
            / f"k_{k:02d}"
        )

        k_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_image(
            mask.detach().cpu(),
            k_dir
            / "mask.png",
        )

        save_image(
            to_01(
                cond.detach().cpu()
            ),
            k_dir
            / "cond.png",
        )

        save_image(
            to_01(
                pred.detach().cpu()
            ),
            k_dir
            / "pred.png",
        )

        abs_error = (
            to_01(
                pred.detach().cpu()
            )
            - to_01(
                x0.detach().cpu()
            )
        ).abs()

        save_image(
            abs_error,
            k_dir
            / "abs_error.png",
        )

        visible_internal = (
            internal_mask
            * observed_mask
        )

        save_image(
            visible_internal.detach().cpu(),
            k_dir
            / "visible_internal_structure.png",
        )

        result = {
            "k":
                k,
            "observed_indices":
                list(
                    observed_indices
                ),
            "current_hidden_mae":
                current_hidden_mae,
            "current_hidden_mse":
                current_hidden_mse,
            "common_hidden_mae":
                common_hidden_mae,
            "common_hidden_mse":
                common_hidden_mse,
            "common_hidden_psnr":
                common_hidden_psnr,
            "internal_visibility":
                visibility,
            "delta_visibility":
                delta_visibility,
            "delta_error_reduction":
                delta_error_reduction,
            "cond":
                cond.detach().cpu(),
            "pred":
                pred.detach().cpu(),
        }

        results.append(
            result
        )

        print(
            f"K={k:02d} "
            f"observed={observed_indices} "
            f"current_hidden_MAE={current_hidden_mae:.6f} "
            f"common_hidden_MAE={common_hidden_mae:.6f} "
            f"common_hidden_PSNR={common_hidden_psnr:.3f} "
            f"visibility={visibility:.4f} "
            f"dV={delta_visibility:.4f} "
            f"dE={delta_error_reduction:.6f}"
        )

        previous_visibility = visibility
        previous_common_mae = (
            common_hidden_mae
        )

    # ------------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------------

    delta_visibility_values = [
        r["delta_visibility"]
        for r in results
        if r["k"] >= 2
    ]

    delta_error_values = [
        r["delta_error_reduction"]
        for r in results
        if r["k"] >= 2
    ]

    pearson_r = (
        pearson_correlation(
            delta_visibility_values,
            delta_error_values,
        )
    )

    print(
        "\nPearson correlation "
        "(Δvisibility vs Δerror reduction) =",
        pearson_r,
    )

    # ------------------------------------------------------------------------
    # Metrics CSV
    # ------------------------------------------------------------------------

    metrics_path = (
        output_dir
        / "metrics.csv"
    )

    with metrics_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "k",
                "observed_indices",
                "current_hidden_mae",
                "current_hidden_mse",
                "common_hidden_mae",
                "common_hidden_mse",
                "common_hidden_psnr",
                "internal_visibility",
                "delta_visibility",
                "delta_error_reduction",
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result["k"],
                    " ".join(
                        str(idx)
                        for idx
                        in result[
                            "observed_indices"
                        ]
                    ),
                    result[
                        "current_hidden_mae"
                    ],
                    result[
                        "current_hidden_mse"
                    ],
                    result[
                        "common_hidden_mae"
                    ],
                    result[
                        "common_hidden_mse"
                    ],
                    result[
                        "common_hidden_psnr"
                    ],
                    result[
                        "internal_visibility"
                    ],
                    result[
                        "delta_visibility"
                    ],
                    result[
                        "delta_error_reduction"
                    ],
                ]
            )

        writer.writerow(
            []
        )

        writer.writerow(
            [
                "pearson_delta_visibility_vs_delta_error_reduction",
                pearson_r,
            ]
        )

    # ------------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------------

    progression_path = (
        save_progression_figure(
            output_dir=output_dir,
            results=results,
            target=x0,
        )
    )

    mae_plot_path = (
        save_common_hidden_mae_plot(
            output_dir=output_dir,
            results=results,
        )
    )

    visibility_plot_path = (
        save_internal_visibility_plot(
            output_dir=output_dir,
            results=results,
        )
    )

    scatter_path = (
        save_delta_scatter(
            output_dir=output_dir,
            results=results,
            pearson_r=pearson_r,
        )
    )

    print(
        "\nsaved:",
        output_dir.resolve(),
    )

    print(
        "metrics:",
        metrics_path.resolve(),
    )

    print(
        "progression:",
        progression_path.resolve(),
    )

    print(
        "MAE plot:",
        mae_plot_path.resolve(),
    )

    print(
        "visibility plot:",
        visibility_plot_path.resolve(),
    )

    print(
        "correlation scatter:",
        scatter_path.resolve(),
    )

    print(
        "\nCumulative slice evaluation with visibility analysis PASSED."
    )


if __name__ == "__main__":
    main()
