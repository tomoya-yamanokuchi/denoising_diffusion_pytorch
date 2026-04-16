# Structural Rewrite Analysis

> Full architectural audit. All issues below are candidates for a ground-up rewrite.

---

## TL;DR — Where Time Actually Goes

```
Training step breakdown (estimated):
  ┌──────────────────────────────────────────────────────┐
  │  UNet forward + backward         ~55%               │
  │  Data loading + transforms       ~15%               │
  │  Optimizer step + EMA            ~10%               │
  │  GPU-CPU sync (.item(), logs)    ~5%                │
  │  Evaluation (every N steps)      ~15% (amortized)   │
  └──────────────────────────────────────────────────────┘
```

**Current optimizations (already applied)** target the 55% block: torch.compile, TF32, flash attn, fused AdamW.

**A rewrite should target everything else — the other 45%.**

---

## CRITICAL: Data Pipeline (cond_image_data_loader.py)

This is the single worst bottleneck in the codebase.

### Problem 1: 60,000 x 60,000 pixel array created at init

```python
# cond_image_data_loader.py:130-138
image = cv2.resize(image, (10000*dim_scale, 10000*dim_scale), cv2.INTER_CUBIC)
```

For 344x344 images, `dim_scale=6`, so this creates a **60K x 60K** array (~10GB RAM) just for mask pattern generation. This happens once at init but is architecturally insane.

**Fix:** Pre-tile the pattern at target resolution. A 344x344 periodic pattern can be generated directly without the 60K intermediate.

### Problem 2: Rejection sampling loop (unbounded)

```python
# cond_image_data_loader.py:154-167
while not (frac >= 0.05 and frac <= 0.9):
    y_coord, x_coord = np.random.randint(...)
    mask = self.pattern_mask[y_coord:y_coord+self.image_size, ...]
    frac = 1 - (mask).mean()
```

This loop has no upper bound. If the pattern is unlucky, it spins indefinitely.

**Fix:** Pre-compute valid crop positions at init, then sample from the valid set in O(1).

### Problem 3: 5 color space conversions per sample

```python
BGR → RGB → PIL → Tensor → NumPy → BGR  # actual flow in __getitem__
```

**Fix:** Stay in one format. Load with torchvision or keep everything in torch tensors.

### Rewrite recommendation

Replace the entire data pipeline with:
- `torchvision.datasets.ImageFolder` or custom dataset that loads directly to tensors
- Pre-computed masks stored as tensors (not generated per-sample)
- `torchdata` or FFCV for zero-copy loading if I/O bound

---

## CRITICAL: Training Loop Blocks on Evaluation

### Problem

Every `save_and_sample_every` steps (default 2000), training stops completely to:

1. Generate 25 sample images (25 full denoising passes × 1000 timesteps)
2. Optionally compute FID (50,000 generated samples)
3. Save checkpoint to disk synchronously

**Estimated time per eval:** 30-120 seconds depending on model size.
**Amortized cost:** ~15-20% of total training time.

### Rewrite recommendation

```
Training process          Eval process (background thread/process)
     │                           │
     ├── step 1999               │
     ├── step 2000 ──► snapshot EMA weights ──► generate samples
     ├── step 2001               │              compute FID
     ├── step 2002               │              save checkpoint
     ├── ...                     │              save images
     ├── step 2050               ◄── done
```

- Use `torch.multiprocessing` or a background thread for eval
- Snapshot EMA weights (not the full model) to CPU, eval on separate stream
- Async `torch.save()` via background thread

---

## CRITICAL: EMA Doubles GPU Memory

### Problem

```python
self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
self.ema.to(self.device)
```

This creates a full copy of the model on GPU. For the UNet at 344x344 with dim_mults=(1,2,4,8), this is ~200-500MB extra. Combined with optimizer states (2x params for Adam), total memory is:

```
Model params:     1x
Gradients:        1x
Adam momentum:    2x
EMA copy:         1x
─────────────────────
Total:            5x model size on GPU
```

### Rewrite recommendation

- **Option A:** Keep EMA on CPU, copy to GPU only for eval (saves 1x model memory on GPU)
- **Option B:** Use in-place EMA update without full model copy (maintain only the state dict)
- **Option C:** Increase `update_every` from 10 to 50-100 (reduces EMA update overhead but doesn't save memory)

---

## HIGH: Evaluation Sampling Not Batched

### Problem (conditional_image_trainer)

```python
for n in range(self.num_samples):        # 25 iterations
    data = next(self.val_dl)              # batch_size=1
    images = self.ema.ema_model.sample(batch_size=1, mask=mask)  # 1 sample at a time
```

25 serial forward passes instead of 1 batched pass.

### Problem (1D trainer post-processing)

```python
for i in range(all_samples_batch.shape[0]):
    aa = torch.zeros(grid, grid, grid, 3).to(device)  # NEW ALLOCATION per iteration
    aa[indexing] = values
    dd = self.get_slice_image(aa)
```

Creates and destroys GPU tensors 25 times in a loop.

### Rewrite recommendation

- Batch all validation samples into a single forward pass
- Pre-allocate output tensors outside the loop
- Vectorize the 1D→2D image reconstruction with scatter operations

---

## HIGH: GPU-CPU Sync in Hot Path

### Problem

```python
for _ in range(self.gradient_accumulate_every):
    ...
    total_loss += loss.item()    # ← SYNC POINT: blocks GPU pipeline
```

`.item()` forces CUDA synchronization on every gradient accumulation sub-step. With `gradient_accumulate_every=2`, this is 2 sync points per training step.

### Rewrite recommendation

```python
# Accumulate on GPU, sync once
total_loss_tensor = torch.zeros(1, device=device)
for _ in range(self.gradient_accumulate_every):
    ...
    total_loss_tensor += loss.detach()
total_loss = total_loss_tensor.item()  # single sync after loop
```

---

## HIGH: Synchronous Checkpoint Saving

### Problem

```python
torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
```

Blocks training for 30-60 seconds per save (model + optimizer + EMA states).

### Rewrite recommendation

```python
import threading

def _async_save(data, path):
    torch.save(data, path)

# In training loop:
save_thread = threading.Thread(target=_async_save, args=(data, path))
save_thread.start()
```

Or use `torch.save` with `_use_new_zipfile_serialization=True` for faster serialization.

---

## MEDIUM: UNet Forward Pass

### Problem 1: Unnecessary clone

```python
# unet_2d.py:453
r = x.clone()  # held in memory through entire forward + backward pass
```

This doubles memory for the initial feature map. The clone is only needed at the end for the final residual connection.

**Fix:** Use `x.detach()` or restructure to avoid the clone entirely (the skip connection pattern doesn't need a clone if no in-place ops modify `x` before it's used).

### Problem 2: Multiple torch.cat allocations

```python
x = torch.cat((x, h.pop()), dim=1)  # happens 2x per up block = 6-8 times total
x = torch.cat((x, r), dim=1)        # final
```

Each `cat` allocates a new tensor.

**Fix:** Pre-allocate output buffer and write into slices, or accept this as inherent to U-Net architecture.

### Problem 3: LinearAttention uses einsum instead of matmul

```python
context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
```

**Fix:** Replace with `torch.bmm` or let `torch.compile` handle fusion (already done).

---

## MEDIUM: Redundant Operations in Diffusion Loss

### Problem

```python
# conditional_image_diffusion.py
binary_mask = (mask != -1.0).any(dim=1, keepdim=True).float()
x = binary_mask * mask + (1 - binary_mask) * x  # 3 intermediate tensors
```

### Fix

```python
x = torch.where(binary_mask.bool(), mask, x)  # single operation, no intermediates
```

### Problem

```python
x_self_cond.detach_()  # redundant inside torch.no_grad()
```

### Problem

```python
loss = F.mse_loss(model_out, target, reduction='none')
loss = reduce(loss, 'b ... -> b', 'mean')
loss = loss * extract(self.loss_weight, t, loss.shape)
return loss.mean()
```

Three separate reduction operations. Could be fused into one.

---

## LOW: Inconsistencies Across Trainers

The codebase has 6 trainer variants that are ~80% copy-pasted:

```
diffusion_trainer.py                 — 268 lines
diffusion_1d_trainer.py              — 322 lines
diffusion_conditional_image_trainer.py — 310 lines
vaeac_trainer.py                     — 203 lines
point_e_diffusion_trainer.py         — (similar)
cvae_trainer.py                      — (similar)
```

### Rewrite recommendation

Create a single `BaseTrainer` with hooks:

```python
class BaseTrainer:
    def train_step(self, batch) -> torch.Tensor:
        """Override: return loss"""
        raise NotImplementedError

    def eval_step(self, step: int):
        """Override: generate samples, compute metrics"""
        raise NotImplementedError

    def train(self):
        # Single canonical training loop with all optimizations
        ...
```

This ensures all optimizations apply uniformly and prevents regression.

---

## Rewrite Priority Matrix

| Change | Speedup | Memory | Effort | Risk |
|--------|---------|--------|--------|------|
| Data pipeline rewrite | **+15-30%** | -50% RAM | 1 day | Low |
| Background eval | **+15-20%** | neutral | 0.5 day | Low |
| GPU-CPU sync fix | **+5-10%** | neutral | 10 min | None |
| Async checkpoint | **+2-5%** | neutral | 30 min | None |
| EMA to CPU | neutral | **-20% VRAM** | 30 min | Low |
| Batch eval sampling | **+5%** (eval only) | -1GB | 1 hour | Low |
| BaseTrainer refactor | maintenance | neutral | 1 day | Medium |
| UNet clone removal | neutral | **-5% VRAM** | 10 min | None |
| torch.where for masks | **+1-2%** | neutral | 10 min | None |
| Pre-allocate eval tensors | **+1%** (eval only) | neutral | 30 min | None |

### Estimated total impact (all changes):

```
Training throughput:  +40-65% additional (on top of existing 1.41x)
Combined with existing optimizations:  ~2.0-2.3x total speedup
GPU memory:           -20-25% VRAM (enables larger batch sizes)
RAM:                  -50% (data pipeline fix)
```

---

## Suggested Rewrite Architecture

```
project/
├── core/
│   ├── unet.py              # Single UNet with channels_last, no clone
│   ├── diffusion.py          # Fused loss computation, torch.where masks
│   ├── attention.py          # Flash attn for all Ampere+, SDPA API
│   └── ema.py                # CPU-resident EMA with lazy GPU transfer
├── data/
│   ├── image_dataset.py      # torchvision-native, no CV2 dependency
│   ├── cond_dataset.py       # Pre-computed masks, O(1) sampling
│   └── transforms.py         # Torch-native transforms only
├── training/
│   ├── base_trainer.py       # Canonical loop: compile, async eval, async save
│   ├── diffusion_trainer.py  # Thin subclass: train_step + eval_step only
│   └── eval_worker.py        # Background evaluation process
└── configs/
    └── ...                   # Hydra/OmegaConf (keep existing)
```
