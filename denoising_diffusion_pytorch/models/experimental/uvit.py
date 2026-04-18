"""
U-ViT — Vision Transformer with U-Net-style skip connections.

Reference: Bao et al., "All are Worth Words: A ViT Backbone for Diffusion Models" (2023)

Interface matches the existing UNet so it can be used directly with GaussianDiffusion:
    model = UViT(dim=384, depth=12, channels=3, image_size=64, patch_size=4)
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


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block."""
    def __init__(self, dim, heads=8, dim_head=64, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        inner_dim = dim_head * heads
        self.heads = heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        # Self-attention
        h = self.norm1(x)
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        x = x + self.to_out(out)

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class SkipLinear(nn.Module):
    """Linear projection for skip connections (concatenation → projection)."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim)

    def forward(self, x, skip):
        return self.linear(torch.cat([x, skip], dim=-1))


class UViT(nn.Module):
    """
    U-ViT — Vision Transformer with U-Net-style long skip connections.

    The architecture has:
    - N//2 encoder blocks
    - 1 middle block
    - N//2 decoder blocks with skip connections from encoder

    Args:
        dim: hidden dimension of transformer
        depth: total number of transformer blocks (must be odd for symmetric skip)
        heads: number of attention heads
        dim_head: dimension per attention head
        mlp_ratio: MLP hidden dim = dim * mlp_ratio
        channels: input image channels
        patch_size: size of each image patch
        learned_variance: if True, output 2x channels
        self_condition: if True, accept self-conditioning input
        gradient_checkpointing: if True, trade compute for memory
    """
    def __init__(
        self,
        dim = 384,
        depth = 13,  # should be odd for symmetric encoder/decoder
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

        # Ensure odd depth for symmetric skip connections
        if depth % 2 == 0:
            depth += 1
        n_skip = depth // 2

        # Patch embedding
        patch_dim = input_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)

        # Time embedding (added as an extra token)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # Encoder blocks (first half)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(dim, heads=heads, dim_head=dim_head, mlp_ratio=mlp_ratio)
            for _ in range(n_skip)
        ])

        # Middle block
        self.mid_block = TransformerBlock(dim, heads=heads, dim_head=dim_head, mlp_ratio=mlp_ratio)

        # Decoder blocks (second half) + skip projections
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(dim, heads=heads, dim_head=dim_head, mlp_ratio=mlp_ratio)
            for _ in range(n_skip)
        ])
        self.skip_linears = nn.ModuleList([
            SkipLinear(dim) for _ in range(n_skip)
        ])

        # Output
        self.final_norm = nn.LayerNorm(dim)
        out_patch_dim = patch_size * patch_size * default_out_dim
        self.final_linear = nn.Linear(dim, out_patch_dim)

        # Learnable positional embedding
        # Will be initialized lazily based on input size
        self.pos_embed = None
        self._cached_n_patches = 0

    def _get_pos_embed(self, n_patches, device):
        # +1 for time token
        total = n_patches + 1
        if self.pos_embed is None or self._cached_n_patches != total:
            self.pos_embed = nn.Parameter(
                torch.randn(1, total, self.patch_embed.out_features, device=device) * 0.02
            )
            self._cached_n_patches = total
        return self.pos_embed

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
        x = self.patch_embed(x)  # (B, N, D)

        # Time token
        t = self.time_mlp(time)  # (B, D)
        t_token = t.unsqueeze(1)  # (B, 1, D)

        # Prepend time token
        x = torch.cat([t_token, x], dim=1)  # (B, N+1, D)

        # Positional embedding
        pos = self._get_pos_embed(x.shape[1] - 1, x.device)
        if pos.shape[1] == x.shape[1]:
            x = x + pos

        # Encoder with skip storage
        skips = []
        for block in self.encoder_blocks:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            skips.append(x)

        # Middle
        if self.gradient_checkpointing and self.training:
            x = grad_checkpoint(self.mid_block, x, use_reentrant=False)
        else:
            x = self.mid_block(x)

        # Decoder with skip connections
        for block, skip_linear, skip in zip(self.decoder_blocks, self.skip_linears, reversed(skips)):
            x = skip_linear(x, skip)
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Remove time token
        x = x[:, 1:, :]

        # Unpatchify
        x = self.final_norm(x)
        x = self.final_linear(x)

        h_patches = H // p
        w_patches = W // p
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                       h=h_patches, w=w_patches, p1=p, p2=p, c=self.out_dim)
        return x
