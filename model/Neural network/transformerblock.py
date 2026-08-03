"""
Transformer Block with SDRM (Spectral-Dynamic Routing Mixer)

Replaces the EPA (Efficient Paired Attention) module in UNETR++'s TransformerBlock
with our proposed SDRM module that combines:
- GCA-based global context mixing
- DLK-based multi-scale local enhancement
- BRA-based sparse routing attention
"""

import torch
import torch.nn as nn
from unetr_pp.network_architecture.dynunet_block import UnetResBlock
from .sdrm import SDRM


class TransformerBlockSDRM(nn.Module):
    """
    A transformer block using SDRM instead of EPA.

    Based on UNETR++ TransformerBlock but replaces the EPA module with
    our Spectral-Dynamic Routing Mixer for improved feature mixing.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        pos_embed: bool = False,
        spatial_shape: tuple = None,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage (H*W*D).
            hidden_size: dimension of hidden layer.
            proj_size: projection size for the routing branch.
            num_heads: number of attention heads.
            dropout_rate: fraction of the input units to drop.
            pos_embed: whether to use positional embedding.
            spatial_shape: (H, W, D) tuple for non-cubic volumes.
        """
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)

        # SDRM replaces EPA here
        self.sdrm_block = SDRM(
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_heads=num_heads,
            channel_attn_drop=dropout_rate,
            spatial_attn_drop=dropout_rate,
            spatial_shape=spatial_shape,
        )

        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.sdrm_block(self.norm(x))

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x
