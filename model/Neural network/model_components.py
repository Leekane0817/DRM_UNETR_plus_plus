"""
Model Components for UNETR++ with SDRM

Modified encoder and decoder that use TransformerBlockSDRM instead of
the original TransformerBlock (which uses EPA).
"""

from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from unetr_pp.network_architecture.layers import LayerNorm
from .transformerblock import TransformerBlockSDRM
from unetr_pp.network_architecture.dynunet_block import get_conv_layer, UnetResBlock

einops, _ = optional_import("einops")


class UnetrPPEncoderSDRM(nn.Module):
    """
    UNETR++ Encoder using SDRM-based Transformer blocks.

    Same structure as the original UnetrPPEncoder but uses
    TransformerBlockSDRM instead of TransformerBlock.
    """

    def __init__(
        self,
        input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4],
        dims=[32, 64, 128, 256],
        proj_size=[64, 64, 64, 32],
        depths=[3, 3, 3, 3],
        num_heads=4,
        spatial_dims=3,
        in_channels=1,
        dropout=0.0,
        transformer_dropout_rate=0.15,
        spatial_shapes=None,   # list of (H,W,D) tuples per stage, for non-cubic volumes
        stem_kernel_size=(2, 4, 4),  # Synapse=(2,4,4), ACDC=(1,4,4)
        **kwargs
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(
                spatial_dims, in_channels, dims[0],
                kernel_size=stem_kernel_size, stride=stem_kernel_size,
                dropout=dropout, conv_only=True,
            ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(
                    spatial_dims, dims[i], dims[i + 1],
                    kernel_size=(2, 2, 2), stride=(2, 2, 2),
                    dropout=dropout, conv_only=True,
                ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple SDRM-based Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                sh = spatial_shapes[i] if spatial_shapes else None
                stage_blocks.append(
                    TransformerBlockSDRM(
                        input_size=input_size[i],
                        hidden_size=dims[i],
                        proj_size=proj_size[i],
                        num_heads=num_heads,
                        dropout_rate=transformer_dropout_rate,
                        pos_embed=False,
                        spatial_shape=sh,
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class UnetrUpBlockSDRM(nn.Module):
    """
    UNETR++ Decoder Block using SDRM-based Transformer blocks.

    Same structure as the original UnetrUpBlock but uses
    TransformerBlockSDRM instead of TransformerBlock.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        proj_size: int = 64,
        num_heads: int = 4,
        out_size: int = 0,
        depth: int = 3,
        conv_decoder: bool = False,
        spatial_shape: tuple = None,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the routing module.
            num_heads: number of heads inside each SDRM module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
            conv_decoder: if True, use ConvBlock instead of SDRM blocks (for last decoder).
            spatial_shape: (H, W, D) tuple for the feature volume at this decoder stage.
        """
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.decoder_block = nn.ModuleList()

        # Last decoder uses ConvBlock (UnetResBlock) instead of SDRM blocks
        # (following the UNETR++ paper supplementary material)
        if conv_decoder:
            self.decoder_block.append(
                UnetResBlock(
                    spatial_dims, out_channels, out_channels,
                    kernel_size=kernel_size, stride=1,
                    norm_name=norm_name,
                )
            )
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(
                    TransformerBlockSDRM(
                        input_size=out_size,
                        hidden_size=out_channels,
                        proj_size=proj_size,
                        num_heads=num_heads,
                        dropout_rate=0.15,
                        pos_embed=False,
                        spatial_shape=spatial_shape,
                    )
                )
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)

        out = out + skip
        out = self.decoder_block[0](out)
        return out
