
import math
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch import nn, einsum
import torch.nn.functional as F



from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


from denoising_diffusion_pytorch.models.attend import Attend
from denoising_diffusion_pytorch.models.helpers import default, cast_tuple, divisible_by , exists


from denoising_diffusion_pytorch.version import __version__


def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


# class MaskImageEmbedding(nn.Module):
#     def __init__(self, time_dim=256, input_size=(32, 32)):
#         super().__init__()
#         self.time_dim = time_dim
#         self.H, self.W = input_size

#         # 位置情報（x, y）を正規化された座標として埋め込み
#         xs = torch.linspace(0, 1, self.W).view(1, 1, 1, self.W).expand(1, 1, self.H, self.W)
#         ys = torch.linspace(0, 1, self.H).view(1, 1, self.H, 1).expand(1, 1, self.H, self.W)
#         self.register_buffer("pos_enc", torch.cat([xs, ys], dim=1))  # (1, 2, H, W)

#         # マスク＋位置の3ch入力をエンコードして時刻次元に
#         self.encoder = nn.Sequential(
#             nn.Conv2d(5, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(32, time_dim)
#         )

#     def forward(self, mask):
#         B, _, H, W = mask.shape # (B, 1, H, W)
#         if (H, W) != (self.H, self.W):
#             mask = nn.functional.interpolate(mask, size=(self.H, self.W), mode='bilinear')
#         pos_enc = self.pos_enc.expand(B, -1, -1, -1)
#         x = torch.cat([mask, pos_enc], dim=1)  # (B, 3, H, W)
#         return self.encoder(x)  # (B, time_dim)


## v4
# class MaskImage_ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         )
#         self.activation = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.activation(x + self.block(x))

# class MaskImageEmbedding(nn.Module):
#     def __init__(self, time_dim=256, input_size=(32, 32)):
#         super().__init__()
#         self.time_dim = time_dim

#         self.encoder = nn.Sequential(
#             nn.Conv2d(4, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             MaskImage_ResidualBlock(32),
#             MaskImage_ResidualBlock(32),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(32, time_dim)
#         )

#     def forward(self, mask):
#         # mask: (B, 3, H, W)
#         mask_label = (mask != -1.0).any(dim=1, keepdim=True).float()  # (B, 1, H, W)
#         x = torch.cat([mask, mask_label], dim=1)  # -> (B, 4, H, W)
#         return self.encoder(x)  # -> (B, time_dim)
    


class MaskImageEmbedding(nn.Module):
    def __init__(self, time_dim=256, input_size=(32, 32)):
        super().__init__()
        self.time_dim = time_dim
        self.H, self.W = input_size

        # 正規化された (x, y) 座標をバッファとして保持
        xs = torch.linspace(0, 1, self.W).view(1, 1, 1, self.W).expand(1, 1, self.H, self.W)
        ys = torch.linspace(0, 1, self.H).view(1, 1, self.H, 1).expand(1, 1, self.H, self.W)
        self.register_buffer("pos_enc", torch.cat([xs, ys], dim=1))  # (1, 2, H, W)

        # 入力は mask (3ch) + mask_label (1ch) + pos_enc (2ch) = 6ch
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            MaskImage_ResidualBlock(32),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, time_dim)
        )

    def forward(self, mask):
        B, _, H, W = mask.shape  # (B, 3, H, W)
        if (H, W) != (self.H, self.W):
            mask = F.interpolate(mask, size=(self.H, self.W), mode='bilinear', align_corners=False)

        # マスク領域の有無を示すラベル (B, 1, H, W)
        mask_label = (mask != -1.0).any(dim=1, keepdim=True).float()

        # 位置エンコーディングをバッチに展開
        pos_enc = self.pos_enc.expand(B, -1, -1, -1)

        # 入力統合: (B, 6, H, W)
        x = torch.cat([mask, mask_label, pos_enc], dim=1)
        return self.encoder(x)  # (B, time_dim)

class MaskImage_ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))
    
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        mask_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False,
        gradient_checkpointing = False
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        # input_channels = channels * (2 if self_condition else 1)
        input_channels = 5 * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # import ipdb;ipdb.set_trace()
        # if mask_dim is not None:
        #     self.mask_img_mlp = MaskImageEmbedding(time_dim,(mask_dim,mask_dim))

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None, mask_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if mask_cond is not None: 
            mask_label = (mask_cond != -1.0).any(dim=1, keepdim=True).float() # tmp13以降
            # mask_label = (mask_cond != -1.0).all(dim=1, keepdim=True).float()  # tmp13以降
            x  = torch.cat([x,mask_cond,mask_label], dim=1)

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x


        # if mask_cond is not None:
        #     t = self.time_mlp(time)+self.mask_img_mlp(mask_cond)
        # else:
        #     t = self.time_mlp(time)

        t = self.time_mlp(time)


        h = []

        def _down_block(block1, block2, attn, downsample, x, t):
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)
            x = downsample(x)
            return x

        for block1, block2, attn, downsample in self.downs:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(_down_block, block1, block2, attn, downsample, x, t, use_reentrant=False)
            else:
                x = _down_block(block1, block2, attn, downsample, x, t)

        if self.gradient_checkpointing and self.training:
            x = grad_checkpoint(self.mid_block1, x, t, use_reentrant=False)
            x = self.mid_attn(x) + x
            x = grad_checkpoint(self.mid_block2, x, t, use_reentrant=False)
        else:
            x = self.mid_block1(x, t)
            x = self.mid_attn(x) + x
            x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)