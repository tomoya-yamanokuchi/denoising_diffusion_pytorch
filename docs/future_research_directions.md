# Future Research Directions for VoxelDiffusionCut

> Target: M1 students, low-effort but novel contributions

---

## Current Limitations (from paper §6 + code analysis)

| # | Limitation | Paper Section | Severity |
|---|---|---|---|
| L1 | Initial cutting position must be predefined | §6.1 | High |
| L2 | Requires large training data per product type (~20K) | §5.2 | High |
| L3 | No observation noise / cutting error modeling | §1, footnote 1 | Medium |
| L4 | Diffusion runs in pixel space (slow at high-res) | — | Medium |
| L5 | 2D slice-by-slice prediction (no cross-slice consistency) | §4.2 | Medium |
| L6 | Color-based part detection (assumes known colors) | §5.1.5, §6.2 | High |
| L7 | Simulation only — no real-world validation | §6.2 | High |

---

## Proposed Research Tasks (for M1 students)

### Task A: Latent Diffusion for Voxel Completion
**Addresses: L4 (slow training)**
**Effort: Low | Novelty: Medium | Paper potential: Workshop / short paper**

**Idea:** Move diffusion from pixel space (344×344) to VAEAC's latent space (~43×43).

```
Current:  x (344×344) → noise → UNet denoise → x̂ (344×344)
Proposed: x → VAEAC encode → z (43×43) → noise → UNet denoise → ẑ → VAEAC decode → x̂
```

**Why it's easy:**
- VAEAC already exists and is trained
- Diffusion code stays the same (just change input size)
- Expected 10-50x speedup — easy to measure and report

**What to do:**
1. Train VAEAC on dataset (already done)
2. Encode all training images to latent vectors
3. Train diffusion model on latent vectors (same GaussianDiffusion, smaller image_size)
4. At inference: encode condition → diffuse in latent → decode
5. Compare: training time, FID/reconstruction quality, task performance

**Related work:** BridgeShape (AAAI 2026), DiffRF (CVPR 2023) — both do latent diffusion for 3D, but not for iterative structure estimation.

---

### Task B: Few-Shot Product Adaptation with LoRA
**Addresses: L2 (requires large data per product)**
**Effort: Low | Novelty: High | Paper potential: Full paper**

**Idea:** Pre-train one foundation model on ALL product types, then adapt to a new product with only 5-10 cutting surface images using LoRA (Low-Rank Adaptation) — like DreamBooth but lighter.

```
Phase 1: Pre-train on mixed data (all objects A-F)
Phase 2: Given 5 images of new Object G → LoRA fine-tune → ready to predict
```

**Why it's easy:**
- LoRA is ~20 lines of code (use `peft` library)
- Don't need to change model architecture
- Experiment: train on Objects A-E, test adaptation to Object F with N={1,3,5,10} shots
- Clear metric: task performance vs number of adaptation samples

**What to do:**
1. Train base model on all training distributions
2. Hold out one object configuration as "new product"
3. LoRA fine-tune with N shots
4. Measure: Cutting Error Volume, Part Remaining Rate, Part Occupancy Rate
5. Compare vs: full retraining, zero-shot, DreamBooth

**Related work:** LoRA (Hu et al., 2022), DreamBooth (Ruiz et al., 2023) — neither applied to industrial 3D structure estimation.

**Why this is a strong paper:** Directly solves real-world problem — recycling plants encounter new products constantly, can't retrain from scratch each time.

---

### Task C: Observation Noise Robustness
**Addresses: L3 (no noise modeling)**
**Effort: Very Low | Novelty: Medium | Paper potential: Workshop / extension**

**Idea:** Add noise augmentation during training to make the model robust to imperfect cutting surface observations.

```
Current training:  condition = clean cutting surface
Proposed training: condition = cutting surface + noise (Gaussian, dropout, color jitter)
```

**Why it's easy:**
- Just add data augmentation in `cond_image_data_loader.py`
- No model change needed
- Experiment: train with noise → test on clean and noisy inputs
- Show degradation curves: model trained with noise degrades gracefully

**What to do:**
1. Add noise augmentations: Gaussian noise, random pixel dropout, color shift
2. Train with augmented conditions
3. Test on: clean, mild noise, heavy noise
4. Compare: augmented model vs non-augmented model
5. Ablation: which noise type matters most?

**Related work:** Noise-robust diffusion (Bansal et al., 2024) — studied for natural images, not applied to conditional 3D completion.

---

### Task D: DiT Architecture for VoxelDiffusionCut
**Addresses: L4 (speed) + architecture modernization**
**Effort: Low | Novelty: Medium | Paper potential: Short paper / tech report**

**Idea:** Replace UNet with DiT (already implemented in this repo!) and validate on the actual task.

**Why it's easy:**
- DiT is already implemented (`models/experimental/dit.py`)
- Benchmark already shows 4.5x speedup at 344×344
- Just need to verify task performance matches UNet

**What to do:**
1. Train DiT on same dataset as UNet (both simple and complex models)
2. Run full eval pipeline (cutting planning)
3. Compare: speed, memory, Cutting Error Volume, Part Remaining Rate
4. Ablation: patch_size = {2, 4, 8}, depth = {6, 12, 24}

**This can extend into the conditional version** — add mask conditioning to DiT (like the current UNet has in `proposed/unet_2d_cond.py`).

---

### Task E: Automatic Initial Cut Selection
**Addresses: L1 (predefined initial cut)**
**Effort: Medium | Novelty: High | Paper potential: Full paper**

**Idea:** Use the product's external appearance (shape, color, weight) to predict the best initial cutting position, removing the manual specification requirement.

```
Current:  human specifies start_action_idx → begin iterative cutting
Proposed: external image/shape → predict start_action_idx → begin
```

**Approach options (pick one):**
- (a) Simple classifier: external shape features → best initial axis & position
- (b) Multi-view diffusion: generate "imagined" internal structure from external view, use it for first cut
- (c) Product-type retrieval: match external shape to database, use known good initial cuts

**Why it's tractable for M1:**
- Option (a) is essentially a small classification problem
- Training data: for each object, optimal initial cut can be computed from ground truth
- Clear experiment: compare random initial cut vs predicted initial cut

**Related work:** None — this is an open problem stated explicitly in the paper (§6.1).

---

### Task F: Cross-Slice Consistency via 3D-Aware Conditioning
**Addresses: L5 (no cross-slice consistency)**
**Effort: Medium | Novelty: High | Paper potential: Full paper**

**Idea:** Current method predicts each 2D slice independently. Add information from adjacent slices as additional conditioning.

```
Current:  predict slice x(i,j) conditioned only on observed mask
Proposed: predict slice x(i,j) conditioned on observed mask + predictions from x(i-1,j), x(i+1,j)
```

**Why it's interesting:**
- Currently, predicted slice 24 and slice 25 might be inconsistent
- Adjacent slices should be spatially coherent
- This is essentially "video diffusion" applied to spatial sequences

**Approach:**
- Add temporal attention (cross-attention between adjacent slices)
- Or: autoregressive generation (predict slice by slice, each conditioned on previous)
- Or: simple post-processing (3D smoothing on predicted voxel grid)

**The simplest version** (3D smoothing post-processing) is very easy and could be enough for a workshop paper.

---

## Recommendation Matrix

| Task | Effort | Novelty | Paper Type | Addresses | Best for |
|---|---|---|---|---|---|
| **A: Latent Diffusion** | Low | Medium | Workshop | L4 | Hands-on student |
| **B: LoRA Few-Shot** | Low | **High** | **Full paper** | L2 | Research-oriented student |
| **C: Noise Robustness** | Very Low | Medium | Workshop | L3 | Beginner student |
| **D: DiT Architecture** | Low | Medium | Short paper | L4 | Engineering student |
| **E: Auto Initial Cut** | Medium | **High** | **Full paper** | L1 | Ambitious student |
| **F: Cross-Slice** | Medium | **High** | **Full paper** | L5 | Theory-oriented student |

### My recommendation for M1 students:

**First priority: Task B (LoRA Few-Shot)**
- Most impactful, directly solves stated limitation
- Very little code change needed
- Clear, quantifiable results
- Reviewer-friendly narrative: "adapt to new products with 5 images"

**Second priority: Task C + D combined**
- "Robust and Efficient VoxelDiffusionCut"
- Add noise augmentation + swap UNet for DiT
- Both are quick to implement
- Together they make a solid workshop/short paper

**If student is strong: Task E**
- Opens a new research question
- High novelty, clear contribution
- But requires more thinking about the approach

---

## Recent Related Work (2024-2025)

| Paper | Venue | Relevance |
|---|---|---|
| BridgeShape (Kong et al.) | AAAI 2026 | Latent diffusion for 3D completion — validates Task A's approach |
| Simba (Zhang et al.) | 2025 | Point cloud completion with symmetry priors — alternative to PCD-DM baseline |
| "Repurposing 2D Diffusion for 3D" (He et al.) | 2025 | 2D→3D via Shape Atlas — similar spirit to your 2D slice approach |
| "Evaluating Latent Generative Paradigms" (Humt et al.) | 2025 | Shows diffusion > autoregressive for multi-modal 3D completion |
| FruitNinja (Wu & Chen) | CVPR 2025 | Cross-section inpainting for cut objects — closest domain work |
| LoRA (Hu et al.) | ICLR 2022 | Low-rank adaptation — key technique for Task B |
| DreamBooth (Ruiz et al.) | CVPR 2023 | Few-shot diffusion personalization — inspiration for Task B |
