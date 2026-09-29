"""
Evaluate cumulative slice observations for a trained multi-product I2SB model.

For one fixed product/sample, evaluate K=1..K_MAX observed slices using one
fixed cumulative observation order. The reverse-sampling RNG is reset to the
same seed for every K so differences are driven mainly by the added
observations rather than different stochastic samples.

Outputs:
  outputs/i2sb_cumulative_slice_<product>_<timestamp>/
    target.png
    x1.png
    slice_order.txt
    metrics.csv
    progression.png
    k_01/{mask,cond,pred,abs_error}.png
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
PRODUCT_NAME         = "sheetsander"
PRODUCT_SAMPLE_INDEX = 0  # index within the selected product
K_MAX                = 25
SLICE_ORDER_SEED     = 12345
SAMPLING_SEED        = 54321
NFE                  = 20

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
    steps = np.linspace(0, interval - 1, nfe + 1, dtype=int)
    steps = np.unique(steps)
    if steps[0] != 0:
        steps = np.insert(steps, 0, 0)
    if steps[-1] != interval - 1:
        steps = np.append(steps, interval - 1)
    return steps.tolist()


def to_01(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def to_vis(x):
    x = to_01(x.detach().cpu())
    if x.ndim == 4:
        x = x[0]
    return x.permute(1, 2, 0).numpy()


# ============================================================================
# Checkpoint loading
# ============================================================================


def load_model_weights(model, checkpoint):
    """Prefer EMA weights if they can be extracted, otherwise raw model."""
    ema_state = checkpoint.get("ema", None)

    if isinstance(ema_state, dict):
        prefix = "ema_model."
        ema_model_state = {
            key[len(prefix):]: value
            for key, value in ema_state.items()
            if key.startswith(prefix)
        }
        if ema_model_state:
            try:
                model.load_state_dict(ema_model_state, strict=True)
                print("loaded EMA model weights")
                return "ema"
            except RuntimeError as exc:
                print("[warning] EMA weights could not be loaded directly:")
                print(exc)
                print("falling back to checkpoint['model']")

    if "model" not in checkpoint:
        raise KeyError(
            "Checkpoint contains neither a usable EMA model nor key `model`."
        )

    model.load_state_dict(checkpoint["model"], strict=True)
    print("loaded raw model weights")
    return "model"


# ============================================================================
# Product/sample selection
# ============================================================================


def find_product_sample(dataset, product_name, product_sample_index):
    matching_indices = [
        idx
        for idx, record in enumerate(dataset.samples)
        if record["product_name"] == product_name
    ]

    if not matching_indices:
        available = sorted(
            {record["product_name"] for record in dataset.samples}
        )
        raise ValueError(
            f"Unknown product `{product_name}`. Available products: {available}"
        )

    if not (0 <= product_sample_index < len(matching_indices)):
        raise IndexError(
            f"PRODUCT_SAMPLE_INDEX={product_sample_index} is out of range for "
            f"product `{product_name}` with {len(matching_indices)} samples."
        )

    global_idx = matching_indices[product_sample_index]
    record = dataset.samples[global_idx]

    # Load directly to avoid dataset.__getitem__ generating a random mask.
    x0 = dataset._load_image(record["x0_path"])
    x1 = dataset.x1_cache[product_name].clone()

    return global_idx, record, x0, x1


# ============================================================================
# Cumulative slice masks
# ============================================================================


def make_cumulative_slice_order(num_slices, seed):
    rng = np.random.default_rng(seed)
    return rng.permutation(num_slices).astype(np.int64)


def make_slice_mask_from_indices(
    image_size,
    grid_rows,
    grid_cols,
    observed_indices,
):
    """Return [1,H,W], with 0=observed and 1=hidden."""
    H = int(image_size)
    W = int(image_size)
    mask = np.ones((H, W), dtype=np.float32)

    y_edges = np.linspace(0, H, grid_rows + 1, dtype=int)
    x_edges = np.linspace(0, W, grid_cols + 1, dtype=int)

    for idx in observed_indices:
        row = int(idx // grid_cols)
        col = int(idx % grid_cols)
        y0, y1 = y_edges[row], y_edges[row + 1]
        x0, x1 = x_edges[col], x_edges[col + 1]
        mask[y0:y1, x0:x1] = 0.0

    return torch.from_numpy(mask[None, :, :])


# ============================================================================
# Metrics
# ============================================================================


def masked_mae(pred, target, mask):
    expanded = mask.expand_as(pred)
    denom = expanded.sum().clamp_min(1.0)
    value = ((pred - target).abs() * expanded).sum() / denom
    return float(value.item())


def masked_mse(pred, target, mask):
    expanded = mask.expand_as(pred)
    denom = expanded.sum().clamp_min(1.0)
    value = ((pred - target).pow(2) * expanded).sum() / denom
    return float(value.item())


def psnr_from_mse(mse, data_range=2.0):
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


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
    # Same random stream for every K => fairer qualitative comparison.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def pred_x0_fn(xt, step_value):
        step = torch.full(
            (xt.shape[0],),
            int(step_value),
            device=device,
            dtype=torch.long,
        )
        net_out = model(xt, step, None, cond)
        return diffusion.compute_pred_x0(
            step=step,
            xt=xt,
            net_out=net_out,
            clip_denoise=True,
        )

    xs, _ = diffusion.ddpm_sampling(
        steps=steps,
        pred_x0_fn=pred_x0_fn,
        x1=x1,
        cond=cond,
        mask=mask,
        ot_ode=False,
        log_steps=[0],
        verbose=False,
    )

    recon = xs[:, 0].to(device)

    # Exact endpoint data consistency.
    recon = (1.0 - mask) * cond + mask * recon
    return recon


# ============================================================================
# Visualization
# ============================================================================


def save_progression_figure(output_dir, results, target):
    num_cols = len(results)
    fig, axes = plt.subplots(3, num_cols, figsize=(4 * num_cols, 12))

    if num_cols == 1:
        axes = np.asarray(axes).reshape(3, 1)

    target_01 = to_01(target.detach().cpu())

    for col, result in enumerate(results):
        k = result["k"]
        cond = result["cond"]
        pred = result["pred"]

        abs_error = (to_01(pred.detach().cpu()) - target_01).abs()

        axes[0, col].imshow(to_vis(cond))
        axes[0, col].set_title(
            f"Condition K={k}\nobs={result['observed_indices']}"
        )
        axes[0, col].axis("off")

        axes[1, col].imshow(to_vis(pred))
        axes[1, col].set_title(
            f"Prediction K={k}\n"
            f"common MAE={result['common_hidden_mae']:.4f}"
        )
        axes[1, col].axis("off")

        error_img = abs_error[0].mean(dim=0).numpy()
        axes[2, col].imshow(error_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[2, col].set_title("Mean RGB abs. error")
        axes[2, col].axis("off")

    plt.tight_layout()
    path = output_dir / "progression.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
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
def main(cfg: DictConfig):
    cfg = cfg.usecase

    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu"
    )
    print("device =", device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    # ------------------------------------------------------------------------
    # Dataset / fixed sample
    # ------------------------------------------------------------------------
    dataset = I2SBCondImageDataset(
        cfg=cfg,
        image_size=cfg.dataset.image_size,
    )

    if dataset.type != "slice":
        raise ValueError(
            "This evaluation requires `dataset.type: slice`, "
            f"got `{dataset.type}`."
        )

    global_idx, record, x0_single, x1_single = find_product_sample(
        dataset=dataset,
        product_name=PRODUCT_NAME,
        product_sample_index=PRODUCT_SAMPLE_INDEX,
    )

    x0 = x0_single.unsqueeze(0).to(device)
    x1 = x1_single.unsqueeze(0).to(device)

    print("\n--- evaluation sample ---")
    print("product =", PRODUCT_NAME)
    print("product sample index =", PRODUCT_SAMPLE_INDEX)
    print("global dataset index =", global_idx)
    print("x0_path =", record["x0_path"])
    print("x1_path =", record["x1_path"])

    # ------------------------------------------------------------------------
    # Model / checkpoint
    # ------------------------------------------------------------------------
    net_cfg = cfg.inferencer.network
    model = DiT(
        dim=net_cfg.dim,
        depth=net_cfg.depth,
        heads=net_cfg.heads,
        dim_head=net_cfg.dim_head,
        patch_size=net_cfg.patch_size,
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    weight_source = load_model_weights(model, checkpoint)
    model.eval()

    print("checkpoint =", CHECKPOINT_PATH)
    print("checkpoint step =", checkpoint.get("step", "unknown"))
    print("weight source =", weight_source)

    # ------------------------------------------------------------------------
    # I2SB process
    # ------------------------------------------------------------------------
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

    diffusion = I2SBDiffusion(betas=betas, device=device)
    steps = make_sampling_steps(interval=interval, nfe=NFE)

    # ------------------------------------------------------------------------
    # Fixed cumulative slice order
    # ------------------------------------------------------------------------
    num_slices = dataset.slice_grid_rows * dataset.slice_grid_cols

    if K_MAX > num_slices:
        raise ValueError(
            f"K_MAX={K_MAX} exceeds num_slices={num_slices}."
        )

    slice_order = make_cumulative_slice_order(
        num_slices=num_slices,
        seed=SLICE_ORDER_SEED,
    )
    selected_order = slice_order[:K_MAX].tolist()

    print("\nfixed cumulative slice order =", selected_order)

    # Fair metric region: pixels still hidden after K_MAX.
    final_mask_single = make_slice_mask_from_indices(
        image_size=dataset.image_size,
        grid_rows=dataset.slice_grid_rows,
        grid_cols=dataset.slice_grid_cols,
        observed_indices=selected_order,
    )
    common_hidden_mask = final_mask_single.unsqueeze(0).to(device)

    # ------------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / (
        f"i2sb_cumulative_slice_{PRODUCT_NAME}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    save_image(to_01(x0), output_dir / "target.png")
    save_image(to_01(x1), output_dir / "x1.png")

    (output_dir / "slice_order.txt").write_text(
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
            ]
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # K = 1 ... K_MAX
    # ------------------------------------------------------------------------
    results = []

    for k in range(1, K_MAX + 1):
        observed_indices = selected_order[:k]

        mask_single = make_slice_mask_from_indices(
            image_size=dataset.image_size,
            grid_rows=dataset.slice_grid_rows,
            grid_cols=dataset.slice_grid_cols,
            observed_indices=observed_indices,
        )
        mask = mask_single.unsqueeze(0).to(device)

        cond = dataset._make_condition(
            x0=x0_single,
            mask=mask_single,
        ).unsqueeze(0).to(device)

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

        current_hidden_mae = masked_mae(pred, x0, mask)
        current_hidden_mse = masked_mse(pred, x0, mask)
        common_hidden_mae = masked_mae(pred, x0, common_hidden_mask)
        common_hidden_mse = masked_mse(pred, x0, common_hidden_mask)
        common_hidden_psnr = psnr_from_mse(common_hidden_mse)

        k_dir = output_dir / f"k_{k:02d}"
        k_dir.mkdir(parents=True, exist_ok=True)

        save_image(mask.detach().cpu(), k_dir / "mask.png")
        save_image(to_01(cond.detach().cpu()), k_dir / "cond.png")
        save_image(to_01(pred.detach().cpu()), k_dir / "pred.png")

        abs_error = (
            to_01(pred.detach().cpu()) - to_01(x0.detach().cpu())
        ).abs()
        save_image(abs_error, k_dir / "abs_error.png")

        results.append(
            {
                "k": k,
                "observed_indices": list(observed_indices),
                "current_hidden_mae": current_hidden_mae,
                "current_hidden_mse": current_hidden_mse,
                "common_hidden_mae": common_hidden_mae,
                "common_hidden_mse": common_hidden_mse,
                "common_hidden_psnr": common_hidden_psnr,
                "cond": cond.detach().cpu(),
                "pred": pred.detach().cpu(),
            }
        )

        print(
            f"K={k:02d} "
            f"observed={observed_indices} "
            f"current_hidden_MAE={current_hidden_mae:.6f} "
            f"common_hidden_MAE={common_hidden_mae:.6f} "
            f"common_hidden_PSNR={common_hidden_psnr:.3f}"
        )

    # ------------------------------------------------------------------------
    # Metrics CSV
    # ------------------------------------------------------------------------
    metrics_path = output_dir / "metrics.csv"

    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "k",
                "observed_indices",
                "current_hidden_mae",
                "current_hidden_mse",
                "common_hidden_mae",
                "common_hidden_mse",
                "common_hidden_psnr",
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result["k"],
                    " ".join(str(i) for i in result["observed_indices"]),
                    result["current_hidden_mae"],
                    result["current_hidden_mse"],
                    result["common_hidden_mae"],
                    result["common_hidden_mse"],
                    result["common_hidden_psnr"],
                ]
            )

    progression_path = save_progression_figure(
        output_dir=output_dir,
        results=results,
        target=x0,
    )

    print("\nsaved:", output_dir.resolve())
    print("metrics:", metrics_path.resolve())
    print("progression:", progression_path.resolve())
    print("\nCumulative slice evaluation PASSED.")


if __name__ == "__main__":
    main()
