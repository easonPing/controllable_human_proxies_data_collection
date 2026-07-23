"""Diffusion-policy network (Conditional 1D U-Net).

Self-contained model definition — depends only on ``torch`` and ``einops``.
It is the noise-prediction network used by the diffusion behavior-cloning
agent: given a noisy action sequence, a diffusion timestep, and the observation
as a global condition, it predicts the noise.

    net = ConditionalUnet1D(input_dim=5, global_cond_dim=242, ...)
    noise_pred = net(noisy_actions, timestep, obs)   # (B, T, 5)

For inference use ``diffusion_policy.DiffusionPolicy`` instead of calling this
directly.
"""
import math
from typing import Union

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish."""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3,
                 n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )
        self.out_channels = out_channels
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, cond=None):
        out = self.blocks[0](x)
        if cond is not None:
            embed = self.cond_encoder(cond)
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """1D U-Net that predicts diffusion noise for an action sequence,
    conditioned on the diffusion timestep and a global observation vector."""

    def __init__(self, input_dim, global_cond_dim=None, use_obs_encoder=True,
                 diffusion_step_embed_dim=256, down_dims=(256, 512, 1024),
                 kernel_size=3, n_groups=8):
        super().__init__()
        down_dims = list(down_dims)
        dsed = diffusion_step_embed_dim
        all_dims = [dsed] + down_dims
        start_dim = down_dims[0]

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 2),
            nn.Mish(),
            nn.Linear(dsed * 2, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            if use_obs_encoder:
                cond_dim += dsed
                self.global_cond_encoder = nn.Linear(global_cond_dim, dsed)
            else:
                cond_dim += global_cond_dim
                self.global_cond_encoder = nn.Identity()
        self.input_encoder = nn.Linear(input_dim, dsed)
        self.input_decoder = nn.Linear(dsed, input_dim)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim,
                                       kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim,
                                       kernel_size=kernel_size, n_groups=n_groups),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, dsed, 1),
        )
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                global_cond: torch.Tensor, **kwargs):
        """sample: (B, T, input_dim); timestep: (B,) or scalar;
        global_cond: (B, global_cond_dim). Returns (B, T, input_dim)."""
        sample = self.input_encoder(sample)
        sample = einops.rearrange(sample, "b h t -> b t h")

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long,
                                     device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if len(global_cond.shape) == 3:
            global_cond = global_cond.flatten(start_dim=1)
        global_cond_enc = self.global_cond_encoder(global_cond)
        global_cond_enc = self.dropout(global_cond_enc)
        global_feature = torch.cat([global_feature, global_cond_enc], axis=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        x = self.input_decoder(x)
        return x


# Model construction config used by the shipped checkpoints. If you retrain
# with different dims, store the new config alongside the weights (see
# convert / diffusion_policy.DiffusionPolicy).
DEFAULT_MODEL_CONFIG = dict(
    input_dim=5,
    diffusion_step_embed_dim=128,
    down_dims=[128, 256, 512],
    kernel_size=5,
    n_groups=8,
    use_obs_encoder=True,
)
