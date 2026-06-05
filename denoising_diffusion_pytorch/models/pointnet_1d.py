import math
import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)

        emb = torch.exp(
            torch.arange(half_dim, device=device) * -emb_scale
        )
        emb = time[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb


class PointNetBlock(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim),
        )

        self.net = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x, time_emb):
        # x: [B, D, N]
        h = x + self.time_mlp(time_emb)[:, :, None]
        return self.net(h) + x


class PointNet1D(nn.Module):
    """
    Drop-in replacement for Unet1D.

    Expected input:
        x: [B, C, N]
        time: [B]
        x_self_cond: [B, C, N] or None

    Output:
        [B, C, N]
    """

    def __init__(
        self,
        dim,
        channels=6,
        self_condition=False,
        depth=6,
        time_dim=None,
        global_dim=None,
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition

        input_channels = channels * (2 if self_condition else 1)

        time_dim = time_dim or dim * 4
        global_dim = global_dim or dim

        self.init_conv = nn.Conv1d(input_channels, dim, kernel_size=1)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.local_blocks = nn.ModuleList([
            PointNetBlock(dim, time_dim)
            for _ in range(depth)
        ])

        self.global_mlp = nn.Sequential(
            nn.Linear(dim, global_dim),
            nn.SiLU(),
            nn.Linear(global_dim, dim),
        )

        self.fuse = nn.Sequential(
            nn.Conv1d(dim * 2, dim, kernel_size=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

        self.final_conv = nn.Conv1d(dim, channels, kernel_size=1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat([x_self_cond, x], dim=1)

        t = self.time_mlp(time)

        h = self.init_conv(x)

        for block in self.local_blocks:
            h = block(h, t)

        # PointNet global feature
        g = h.max(dim=-1).values          # [B, D]
        g = self.global_mlp(g)            # [B, D]
        g = g[:, :, None].expand_as(h)    # [B, D, N]

        h = torch.cat([h, g], dim=1)
        h = self.fuse(h)

        return self.final_conv(h)
