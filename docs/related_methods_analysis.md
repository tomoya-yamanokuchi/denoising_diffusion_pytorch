# Related Methods Analysis: VoxelDiffusionCut vs Stable Diffusion / DreamBooth / FruitNinja

## 1. Methods Overview

### Our Method: VoxelDiffusionCut
- **Task:** Predict 3D internal structure from observed cutting surfaces, plan non-destructive cuts
- **Representation:** Voxel grid (K×K×K), color-coded parts as attributes
- **Model:** Conditional diffusion (UNet/DiT) with CFG, conditioned on observed mask
- **Output:** 3D voxel completion → presence score map → cutting action

### Stable Diffusion
- **Task:** Text-to-image / image-to-image generation
- **Representation:** Latent space (VAE encodes 512×512 → 64×64 latent)
- **Model:** UNet diffusion in latent space, conditioned on CLIP text embeddings
- **Output:** 2D images

### DreamBooth
- **Task:** Personalize a pre-trained diffusion model to generate a specific subject
- **Technique:** Fine-tune with 3-5 images + rare token identifier [V] + prior preservation loss
- **Output:** New images of the subject in novel contexts/styles

### FruitNinja (CVPR 2025)
- **Task:** Generate realistic internal textures when 3D objects are cut open
- **Representation:** 3D Gaussian Splatting (3DGS)
- **Model:** Pre-trained diffusion model for cross-section inpainting + voxel-grid smoothing
- **Output:** Real-time sliceable 3D objects with realistic interiors

---

## 2. Similarities & Differences

### VoxelDiffusionCut vs Stable Diffusion

| Aspect | VoxelDiffusionCut | Stable Diffusion |
|---|---|---|
| Diffusion algorithm | DDPM/DDIM, same math | DDPM/DDIM, same math |
| Conditioning | Mask (observed cutting surface) | Text (CLIP embeddings) |
| Space | **Pixel space** (direct voxel image) | **Latent space** (VAE encoded) |
| Architecture | UNet with mask concat | UNet with cross-attention to text |
| Resolution | 64×64 or 344×344 | 64×64 latent (= 512×512 pixel) |
| Training data | Domain-specific voxel images (~20K) | LAION-5B (billions of images) |

**Key insight:** VoxelDiffusionCut could adopt **Latent Diffusion** (run diffusion in VAEAC's latent space) for massive speedup — this is literally what Stable Diffusion does vs raw pixel diffusion.

### VoxelDiffusionCut vs DreamBooth

| Aspect | VoxelDiffusionCut | DreamBooth |
|---|---|---|
| Goal | Predict unseen structure | Generate seen subject in new context |
| Training | From scratch on domain data | **Fine-tune** pre-trained model |
| Data needed | ~20K voxel images | **3-5 images** |
| Key technique | CFG for conditional/unconditional | Prior preservation loss |
| Personalization | Per-product-type model | Per-subject model |

**Key insight:** DreamBooth's few-shot fine-tuning could enable **rapid adaptation** of VoxelDiffusionCut to new product types with very few examples.

### VoxelDiffusionCut vs FruitNinja

| Aspect | VoxelDiffusionCut | FruitNinja |
|---|---|---|
| Domain | Industrial products (battery extraction) | Natural objects (fruits) |
| Goal | **Plan where to cut** (decision-making) | **Visualize cuts** (rendering) |
| Internal structure | Functional parts (battery, PCB, motor) | Texture (flesh, seeds) |
| Representation | Voxel grid (discrete) | 3D Gaussian Splatting (continuous) |
| Diffusion role | **Core predictor** (estimates structure) | **Inpainting tool** (fills cross-sections) |
| Uncertainty | Explicitly modeled (UCB score map) | Not modeled |
| Cutting | Sequential planning (iterative) | Arbitrary slicing (real-time) |

**This is the closest related work.** Both address "what's inside when you cut," but from opposite angles:
- FruitNinja: "How should the cut **look**?" (visual realism)
- VoxelDiffusionCut: "Where should we **cut**?" (decision-making under uncertainty)

---

## 3. Potential Future Directions

### Direction 1: Latent Diffusion for VoxelDiffusionCut
**From Stable Diffusion's architecture**

Currently running diffusion at 344×344 pixel space. Moving to latent space (like Stable Diffusion) would:
- Train on ~43×43 latent instead of 344×344 pixel → **10-50x faster**
- VAEAC already exists as the encoder/decoder
- Implementation: train diffusion in VAEAC's bottleneck space

```
Current:  Image (344×344) → Diffusion → Image (344×344)
Proposed: Image (344×344) → VAEAC Encoder → Latent (43×43) → Diffusion → Latent → VAEAC Decoder → Image
```

**Feasibility:** High. You already have a trained VAEAC.

### Direction 2: DreamBooth-style Product Adaptation
**From DreamBooth's fine-tuning approach**

Currently: train a separate model per product type (20K samples each).
With DreamBooth-style fine-tuning:
- Pre-train a **foundation model** on all product types
- Fine-tune on **3-5 cutting surface images** of a new product
- Prior preservation loss prevents forgetting general cutting patterns

**Impact:** Deploy to new products in minutes, not days of training.

**Feasibility:** Medium. Need to verify that prior preservation loss works for voxel images (not just natural images).

### Direction 3: Cross-Section Inpainting from FruitNinja
**From FruitNinja's progressive inpainting**

FruitNinja uses diffusion to **inpaint cross-sections progressively** with voxel-grid smoothing for 3D consistency.

Potential for VoxelDiffusionCut:
- Instead of predicting the entire 3D voxel grid at once, **progressively refine** predictions as more cuts are observed
- Use FruitNinja's voxel-grid smoothing to enforce spatial consistency between adjacent slices
- Apply 3DGS for **real-time visualization** during cutting planning

**Feasibility:** Medium-High. The progressive inpainting idea fits naturally with VoxelDiffusionCut's iterative framework.

### Direction 4: Real Image Conditioning (DreamBooth + VoxelDiffusionCut)
**Bridging sim-to-real gap**

Currently: conditioning on idealized voxel images from simulation.
Future: condition on **real camera images** of cutting surfaces.

Pipeline:
1. Use SAM/Mask R-CNN to segment real cutting surface photos
2. Map segmented regions to voxel color codes
3. DreamBooth fine-tune the diffusion model on a few real cutting images
4. Run VoxelDiffusionCut planning on real observations

This directly addresses **Paper Section 6.2 (Real-World Deployment)**.

### Direction 5: Uncertainty-Aware 3DGS (FruitNinja + VoxelDiffusionCut)
**Novel contribution potential**

Neither FruitNinja nor standard 3DGS models uncertainty. VoxelDiffusionCut's UCB-based presence score map could be extended to:
- Render **uncertainty heatmaps** on 3D objects in real-time (via 3DGS)
- Visualize which regions the model is confident/uncertain about
- Enable interactive cutting planning with visual feedback

**This could be a strong follow-up paper.**

---

## 4. Summary Table

| Direction | Source | Impact | Effort | Paper Potential |
|---|---|---|---|---|
| Latent Diffusion | Stable Diffusion | **10-50x training speedup** | Medium | Incremental |
| Few-shot product adaptation | DreamBooth | **New products in minutes** | Medium | Strong |
| Progressive cross-section inpainting | FruitNinja | Better 3D consistency | Medium-High | Moderate |
| Real image conditioning | DreamBooth + SAM | **Sim-to-real transfer** | High | Strong |
| Uncertainty-aware 3DGS visualization | FruitNinja + Ours | Interactive planning | High | **Very Strong** |

---

## 5. Recommended Priority

1. **Latent Diffusion** — immediate speedup, low risk, directly applicable
2. **DreamBooth adaptation** — high research value, addresses paper's limitation (§6.1)
3. **Uncertainty-aware 3DGS** — novel contribution, strong paper potential
