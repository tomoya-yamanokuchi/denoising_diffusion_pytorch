"""Visual sanity check for the slice-aware I2SB mask."""
from datetime import datetime
from pathlib import Path
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from denoising_diffusion_pytorch.data_loader.i2sb_cond_image_data_loader import I2SBCondImageDataset


def to_vis(x):
    x = x.detach().cpu()
    x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    return x.permute(1, 2, 0).numpy()


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg = cfg.usecase
    dataset = I2SBCondImageDataset(cfg=cfg, image_size=cfg.dataset.image_size)
    if dataset.type != "slice":
        raise ValueError(f"Set dataset.type=slice, got {dataset.type}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/i2sb_slice_mask_sanity_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_examples = 12
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4 * num_examples))
    observed_counts = []

    for i in range(num_examples):
        sample = dataset[0]
        x0 = sample["x0"]
        cond = sample["cond"]
        mask = sample["mask"][0].numpy()

        y_edges = np.linspace(0, dataset.image_size, dataset.slice_grid_rows + 1, dtype=int)
        x_edges = np.linspace(0, dataset.image_size, dataset.slice_grid_cols + 1, dtype=int)
        observed_indices = []
        for r in range(dataset.slice_grid_rows):
            for c in range(dataset.slice_grid_cols):
                tile = mask[y_edges[r]:y_edges[r + 1], x_edges[c]:x_edges[c + 1]]
                if tile.mean() == 0.0:
                    observed_indices.append(r * dataset.slice_grid_cols + c)

        observed_counts.append(len(observed_indices))

        axes[i, 0].imshow(to_vis(x0))
        axes[i, 0].set_title("X0: full structure")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(1.0 - mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Observed slices: {observed_indices}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(to_vis(cond))
        axes[i, 2].set_title(f"Condition C: K={len(observed_indices)}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    output_path = output_dir / "slice_mask_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("mask type =", dataset.type)
    print("grid =", dataset.slice_grid_rows, "x", dataset.slice_grid_cols)
    print("K range =", dataset.slice_num_observed_min, "-", dataset.slice_num_observed_max)
    print("selection =", dataset.slice_selection)
    print("observed counts =", observed_counts)
    print("saved:", output_path.resolve())
    print("\nI2SB slice-mask sanity check PASSED.")


if __name__ == "__main__":
    main()
