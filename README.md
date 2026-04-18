# VoxelDiffusionCut

Non-destructive Internal-part Extraction via Iterative Cutting and Structure Estimation.

> **Paper:** T. Hachimine, Y. Kwon, C.-Y. Kuo, T. Yamanokuchi, T. Matsubara, "VoxelDiffusionCut: Non-destructive Internal-part Extraction via Iterative Cutting and Structure Estimation," *Applied Intelligence*, 2026. (Under Review)

## Overview

VoxelDiffusionCut iteratively estimates the internal structure of a product from observed cutting surfaces using a conditional diffusion model, and plans cutting positions to extract a target internal part (e.g., battery) without damage.

```
Step 1: Execute cutting action
Step 2: Observe cutting surface → condition the diffusion model
Step 3: Predict M internal structures → compute presence score map
Step 4: Plan next cut (maximize removal volume, avoid target part)
Step 5: Repeat for T task steps
```

## Project Structure

```
.
├── scripts/
│   ├── train/                  # Training entry points
│   │   ├── run_train.py            # Hydra-based training (recommended)
│   │   └── train_cond_image_diffusion_v1.py
│   ├── eval/                   # Evaluation & cutting planning
│   │   ├── eval_image_diffusion_v7.py            # Real (complex) models
│   │   ├── eval_image_diffusion_v7_simple_model.py  # Simple models
│   │   └── post_process_data_v3.py               # Metrics computation
│   └── data_generation/        # Dataset creation
│       ├── generate_voxel_image_w_multi_color.py       # Simple objects
│       └── generate_voxel_image_w_multi_color_real_obj.py  # Real objects
│
├── config/
│   ├── vae.py                  # Config for real (complex) models (344x344)
│   └── vae_simple_model.py     # Config for simple models (64x64)
│
├── denoising_diffusion_pytorch/
│   ├── models/
│   │   ├── proposed/               # Paper's proposed method
│   │   │   ├── conditional_diffusion_cfg.py   # CFG conditional diffusion
│   │   │   └── unet_2d_cond.py               # Mask-conditioned UNet
│   │   ├── baselines/              # Baseline methods (VAEAC, PCD-DM)
│   │   │   ├── vaeac/                  # VAEAC baseline
│   │   │   ├── point_e/                # PCD-DM baseline (1D diffusion)
│   │   │   └── cvae/                   # CVAE baseline
│   │   ├── experimental/           # New architectures (compute-opt)
│   │   │   ├── dit.py                  # Diffusion Transformer
│   │   │   └── uvit.py                 # U-ViT
│   │   ├── diffusion.py            # Core: GaussianDiffusion (2D)
│   │   ├── diffusion_1d.py         # Core: GaussianDiffusion1D
│   │   ├── unet_2d.py              # Core: UNet (unconditional)
│   │   ├── unet_1d.py              # Core: UNet1D (point cloud)
│   │   ├── attend.py               # Flash attention module
│   │   └── helpers.py              # Shared utilities
│   │
│   ├── trainer/                # Training loops
│   │   ├── diffusion_trainer.py                    # 2D unconditional
│   │   ├── diffusion_conditional_image_trainer.py  # 2D conditional (proposed)
│   │   ├── diffusion_1d_trainer.py                 # 1D point cloud (PCD-DM)
│   │   └── vaeac_trainer.py                        # VAEAC baseline
│   │
│   ├── policy/                 # Cutting action planning
│   │   ├── cutting_surface_planner_v8.py     # Real models
│   │   ├── cutting_surface_planner_v9.py     # Simple models
│   │   ├── decision/                         # Decision rules (UCB, etc.)
│   │   └── planning/                         # Action selection & candidates
│   │
│   ├── data_loader/            # Dataset classes
│   ├── eval/                   # Episode runner & artifact management
│   ├── env/                    # Voxel cutting simulation
│   ├── cost/                   # Cutting cost estimation
│   └── utils/                  # Utilities (arrays, serialization, etc.)
│
├── app/                        # Wiring layer (Hydra builders)
└── docs/                       # Reports & figures
```

## Methods & Config Mapping

| Method (Paper) | Config Key | Model | Diffusion | Trainer |
|---|---|---|---|---|
| **Proposed** | `conditional_image_diffusion` | `proposed.unet_2d_cond` | `proposed.conditional_diffusion_cfg` | `diffusion_conditional_image_trainer` |
| VAEAC | `vaeac` | `baselines.vaeac` | — | `vaeac_trainer` |
| PCD-DM | `diffusion_1d` | `unet_1d` | `diffusion_1d` | `diffusion_1d_trainer` |
| Proposed-Nocond | `conditional_image_diffusion` | same | same | same (`ctrl_mode="no_cond"`) |
| Proposed-GT | `conditional_image_diffusion` | same | same | same (`ctrl_mode="oracle_obs"`) |
| Random | — | — | — | same (`ctrl_mode="random"`) |

## Quick Start

### Setup

```bash
# Create environment
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -e . --no-deps
uv pip install accelerate einops ema-pytorch pillow tqdm tensorboard
```

### Training

```bash
# Proposed method (conditional diffusion with CFG)
python scripts/train/run_train.py

# Or with Hydra config override
python scripts/train/run_train.py diffusion.batch_size=32
```

### Evaluation

```bash
# Simple models
bash scripts/eval/eval_diffusion.sh

# Or directly
python scripts/eval/eval_image_diffusion_v7_simple_model.py --config config.vae_simple_model
```

## Compute Optimizations (`compute-opt` branch)

The `compute-opt` branch includes training speed optimizations:

| Optimization | Impact |
|---|---|
| `torch.compile` on UNet | 1.4x faster training |
| Flash attention fix (all Ampere+ GPUs) | Enabled for RTX 3090/4090 |
| TF32 + BF16 mixed precision | +10% speed |
| Fused AdamW optimizer | +5% speed |
| Async checkpoint saving | Non-blocking I/O |
| Gradient checkpointing | 36% less VRAM at 344x344 |
| Data pipeline rewrite | 312x less RAM (92MB vs 28.8GB) |
| **DiT architecture** (drop-in) | **4.5x faster at 344x344** |

See [docs/optimization_report.md](docs/optimization_report.md) and [docs/executive_summary.md](docs/executive_summary.md) for full details.

### Using DiT instead of UNet

```python
from denoising_diffusion_pytorch.models.experimental.dit import DiT
from denoising_diffusion_pytorch.models.diffusion import GaussianDiffusion

model = DiT(dim=384, depth=12, heads=6, patch_size=8)
diffusion = GaussianDiffusion(model, image_size=344, timesteps=1000)
# Same training pipeline — no other changes needed
```

## Hardware

| Setup | 344x344 Training Time (800k steps) |
|---|---|
| RTX 3090 Ti + UNet | ~62 hours |
| RTX 3090 Ti + DiT | ~10 hours |
| H100 x2 + DiT (estimated) | ~2-3 hours |

## Citation

```bibtex
@article{hachimine2026voxeldiffusioncut,
  title={VoxelDiffusionCut: Non-destructive Internal-part Extraction 
         via Iterative Cutting and Structure Estimation},
  author={Hachimine, Takumi and Kwon, Yuhwan and Kuo, Cheng-Yu 
          and Yamanokuchi, Tomoya and Matsubara, Takamitsu},
  journal={Applied Intelligence},
  year={2026}
}
```

## License

MIT
