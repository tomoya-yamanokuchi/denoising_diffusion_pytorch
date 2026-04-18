"""
Diffusion Transformer (DiT) — drop-in replacement for UNet.

Reference: Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023)

Interface matches the existing UNet so it can be used directly with GaussianDiffusion:
    model = DiT(dim=384, depth=12, channels=3, image_size=64, patch_size=4)
    diffusion = GaussianDiffusion(model, image_size=64, ...)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Norm with scale and shift (AdaLN-Zero)."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, cond):
        # cond: (B, D) → 6 modulation parameters
        params = self.linear(cond).unsqueeze(1)  # (B, 1, 6D)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)
        return gamma1, beta1, alpha1, gamma2, beta2, alpha2


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning."""
    def __init__(self, dim, heads=8, dim_head=64, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaln = AdaLayerNorm(dim)

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x, cond):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(x, cond)

        # Self-attention with AdaLN
        h = self.norm1(x) * (1 + gamma1) + beta1
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        x = x + alpha1 * out

        # MLP with AdaLN
        h = self.norm2(x) * (1 + gamma2) + beta2
        x = x + alpha2 * self.mlp(h)

        return x


class DiT(nn.Module):
    """
    Diffusion Transformer — drop-in replacement for UNet.

    Args:
        dim: hidden dimension of transformer
        depth: number of transformer blocks
        heads: number of attention heads
        dim_head: dimension per attention head
        mlp_ratio: MLP hidden dim = dim * mlp_ratio
        channels: input image channels
        patch_size: size of each image patch (image_size must be divisible by this)
        learned_variance: if True, output 2x channels
        self_condition: if True, accept self-conditioning input
        gradient_checkpointing: if True, trade compute for memory
    """
    def __init__(
        self,
        dim = 384,
        depth = 12,
        heads = 6,
        dim_head = 64,
        mlp_ratio = 4.0,
        channels = 3,
        patch_size = 4,
        learned_variance = False,
        self_condition = False,
        gradient_checkpointing = False,
        # ignored args for UNet compat
        init_dim = None,
        out_dim = None,
        dim_mults = None,
        resnet_block_groups = None,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = None,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = None,
        attn_heads = None,
        full_attn = None,
        flash_attn = None,
    ):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        self.patch_size = patch_size
        self.gradient_checkpointing = gradient_checkpointing
        self.random_or_learned_sinusoidal_cond = False

        input_channels = channels * (2 if self_condition else 1)
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default_out_dim

        # Patch embedding
        patch_dim = input_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, heads=heads, dim_head=dim_head, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Output
        self.final_norm = nn.LayerNorm(dim)
        self.final_linear = nn.Linear(dim, patch_size * patch_size * default_out_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize like DiT paper
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    @property
    def downsample_factor(self):
        return self.patch_size

    def forward(self, x, time, x_self_cond=None):
        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, f'Image size ({H}, {W}) must be divisible by patch_size {p}'

        if self.self_condition:
            x_self_cond = x_self_cond if x_self_cond is not None else torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)

        # Patchify: (B, C, H, W) → (B, N, patch_dim)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_embed(x)
        n_patches = x.shape[1]

        # Add positional embedding (learned via sinusoidal)
        pos = torch.arange(n_patches, device=x.device).float()
        pos_emb = SinusoidalPosEmb(x.shape[-1])(pos)  # (N, D)
        x = x + pos_emb.unsqueeze(0)

        # Time conditioning
        t = self.time_mlp(time)  # (B, D)

        # Transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(block, x, t, use_reentrant=False)
            else:
                x = block(x, t)

        # Unpatchify
        x = self.final_norm(x)
        x = self.final_linear(x)  # (B, N, p*p*out_channels)

        h_patches = H // p
        w_patches = W // p
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                       h=h_patches, w=w_patches, p1=p, p2=p, c=self.out_dim)
        return x
