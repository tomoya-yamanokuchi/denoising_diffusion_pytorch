"""
I2SB-specific Diffusion Transformer (DiT).

Separated from the existing experimental/dit.py so the Conditional DDPM /
VoxelDiffusionCut pipeline remains unchanged.

Inputs:
    x    : current bridge state X_t, [B, 3, H, W]
    time : discrete timestep, [B]
    cond : partial observation C, [B, 3, H, W]

The binary mask is intentionally NOT given to the network.
It remains available to the trainer for masked-loss computation.
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


def get_1d_sincos_pos_embed(embed_dim, positions, theta=10000.0):
    assert embed_dim % 2 == 0
    positions = positions.reshape(-1).float()
    frequencies = torch.arange(
        embed_dim // 2,
        device=positions.device,
        dtype=torch.float32,
    )
    frequencies = torch.exp(
        -math.log(theta) * frequencies / (embed_dim // 2)
    )
    angles = positions[:, None] * frequencies[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_height,
    grid_width,
    device,
    theta=10000.0,
):
    assert embed_dim % 4 == 0

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_height, device=device, dtype=torch.float32),
        torch.arange(grid_width, device=device, dtype=torch.float32),
        indexing="ij",
    )

    y_embedding = get_1d_sincos_pos_embed(
        embed_dim // 2, grid_y, theta=theta
    )
    x_embedding = get_1d_sincos_pos_embed(
        embed_dim // 2, grid_x, theta=theta
    )

    return torch.cat([y_embedding, x_embedding], dim=-1)


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, cond):
        params = self.linear(cond).unsqueeze(1)
        return params.chunk(6, dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaln = AdaLayerNorm(dim)

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

    def forward(self, x, cond):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(
            x, cond
        )

        h = self.norm1(x) * (1 + gamma1) + beta1
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(
                t, "b n (h d) -> b h n d", h=self.heads
            ),
            qkv,
        )

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        x = x + alpha1 * out

        h = self.norm2(x) * (1 + gamma2) + beta2
        x = x + alpha2 * self.mlp(h)

        return x


class DiT(nn.Module):
    """I2SB-specific conditional Diffusion Transformer."""

    def __init__(
        self,
        dim=384,
        depth=12,
        heads=6,
        dim_head=64,
        mlp_ratio=4.0,
        channels=3,
        cond_channels=3,
        patch_size=4,
        learned_variance=False,
        self_condition=False,
        gradient_checkpointing=False,
        # ignored compatibility args
        init_dim=None,
        out_dim=None,
        mask_dim=None,
        dim_mults=None,
        resnet_block_groups=None,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=None,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=None,
        attn_heads=None,
        full_attn=None,
        flash_attn=None,
    ):
        super().__init__()

        self.channels = channels
        self.cond_channels = cond_channels
        self.self_condition = self_condition
        self.patch_size = patch_size
        self.gradient_checkpointing = gradient_checkpointing
        self.pos_emb_theta = sinusoidal_pos_emb_theta
        self.random_or_learned_sinusoidal_cond = False

        # x_t (3ch) + cond C (3ch) = 6ch
        input_channels = channels + cond_channels

        if self_condition:
            input_channels *= 2

        default_out_dim = channels * (
            1 if not learned_variance else 2
        )
        self.out_dim = default_out_dim

        patch_dim = input_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

        self.final_norm = nn.LayerNorm(dim)
        self.final_linear = nn.Linear(
            dim,
            patch_size * patch_size * default_out_dim,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    @property
    def downsample_factor(self):
        return self.patch_size

    def forward(
        self,
        x,
        time,
        x_self_cond=None,
        cond=None,
    ):
        B, C, H, W = x.shape
        p = self.patch_size

        if C != self.channels:
            raise ValueError(
                f"Expected {self.channels} X_t channels, got {C}."
            )

        if H % p != 0 or W % p != 0:
            raise ValueError(
                f"Image size ({H}, {W}) must be divisible by patch_size {p}."
            )

        if cond is None:
            cond = torch.zeros(
                B,
                self.cond_channels,
                H,
                W,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            expected = (B, self.cond_channels, H, W)
            if tuple(cond.shape) != expected:
                raise ValueError(
                    f"Expected cond shape {expected}, got {tuple(cond.shape)}."
                )

            cond = cond.to(
                device=x.device,
                dtype=x.dtype,
            )

        # Network sees only X_t and C.
        x = torch.cat([x, cond], dim=1)

        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            elif x_self_cond.shape != x.shape:
                raise ValueError(
                    "x_self_cond must match concatenated [X_t, C]."
                )

            x = torch.cat((x_self_cond, x), dim=1)

        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=p,
            p2=p,
        )
        x = self.patch_embed(x)

        h_patches = H // p
        w_patches = W // p

        pos_emb = get_2d_sincos_pos_embed(
            embed_dim=x.shape[-1],
            grid_height=h_patches,
            grid_width=w_patches,
            device=x.device,
            theta=self.pos_emb_theta,
        )

        x = x + pos_emb.to(dtype=x.dtype).unsqueeze(0)

        t = self.time_mlp(time)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(
                    block,
                    x,
                    t,
                    use_reentrant=False,
                )
            else:
                x = block(x, t)

        x = self.final_norm(x)
        x = self.final_linear(x)

        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h_patches,
            w=w_patches,
            p1=p,
            p2=p,
            c=self.out_dim,
        )

        return x
