"""
UNETR++ with SDRM (Spectral-Dynamic Routing Mixer)

This model replaces the EPA (Efficient Paired Attention) module in UNETR++
with our proposed SDRM module, which combines three complementary paths:
1. GCA (Global Context Attention): Token-level global context
2. Dynamic (DLK): Multi-scale large-kernel local enhancement
3. Routing (BRA): Content-dependent sparse attention

The model maintains the same overall U-Net architecture as UNETR++
but with improved feature mixing in the transformer blocks.

Usage:
    >>> model = UNETR_PP_SDRM(
    ...     in_channels=1,
    ...     out_channels=14,
    ...     img_size=(64, 128, 128),
    ...     feature_size=16,
    ...     num_heads=4,
    ...     norm_name='batch',
    ...     depths=[3, 3, 3, 3],
    ...     dims=[32, 64, 128, 256],
    ...     do_ds=True,
    ... )
"""

from torch import nn
from typing import Tuple, Union
from unetr_pp.network_architecture.neural_network import SegmentationNetwork
from unetr_pp.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
from .model_components import UnetrPPEncoderSDRM, UnetrUpBlockSDRM


class UNETR_PP_SDRM(SegmentationNetwork):
    """
    UNETR++ with Spectral-Dynamic Routing Mixer (SDRM).

    Based on: "Shaker et al., UNETR++: Delving into Efficient and
    Accurate 3D Medical Image Segmentation" with the EPA module
    replaced by our proposed SDRM module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int] = (64, 128, 128),
        feature_size: int = 16,
        hidden_size: int = 256,
        num_heads: int = 4,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        depths=None,
        dims=None,
        conv_op=nn.Conv3d,
        do_ds=True,
        stem_kernel_size: Tuple[int, int, int] = (2, 4, 4),  # (2,4,4) for Synapse, (1,4,4) for ACDC
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size (base channels).
            hidden_size: dimension of the last encoder.
            num_heads: number of attention heads in routing branch.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: fraction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
            stem_kernel_size: stride of the stem convolution, e.g.
                (2,4,4) for Synapse, (1,4,4) for ACDC.
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = stem_kernel_size
        # Compute per-stage token counts dynamically from img_size
        f0 = (img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1], img_size[2] // self.patch_size[2])
        f1 = (f0[0] // 2, f0[1] // 2, f0[2] // 2)
        f2 = (f1[0] // 2, f1[1] // 2, f1[2] // 2)
        f3 = (f2[0] // 2, f2[1] // 2, f2[2] // 2)
        encoder_input_size = [
            f0[0] * f0[1] * f0[2],  # stage 0
            f1[0] * f1[1] * f1[2],  # stage 1
            f2[0] * f2[1] * f2[2],  # stage 2
            f3[0] * f3[1] * f3[2],  # stage 3
        ]
        self.feat_size = f3  # bottleneck shape for proj_feat
        self.decoder_sizes = encoder_input_size[:-1][::-1]  # [s2, s1, s0] for dec5,4,3
        self.hidden_size = hidden_size

        # SDRM-based encoder (replaces UnetrPPEncoder)
        self.unetr_pp_encoder = UnetrPPEncoderSDRM(
            dims=dims,
            depths=depths,
            num_heads=num_heads,
            input_size=encoder_input_size,
            spatial_shapes=[f0, f1, f2, f3],
            stem_kernel_size=stem_kernel_size,
            in_channels=in_channels,
        )

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        # Decoder depths mirror encoder: dec5→enc[2], dec4→enc[1], dec3→enc[0]
        # out_size computed dynamically from img_size
        self.decoder5 = UnetrUpBlockSDRM(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.decoder_sizes[0],
            depth=depths[2],
            spatial_shape=f2,
        )

        self.decoder4 = UnetrUpBlockSDRM(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.decoder_sizes[1],
            depth=depths[1],
            spatial_shape=f1,
        )

        self.decoder3 = UnetrUpBlockSDRM(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.decoder_sizes[2],
            depth=depths[0],
            spatial_shape=f0,
        )

        # dec2 outputs to original image size: f0 upsampled by stem_stride
        dec2_out_size = img_size[0] * img_size[1] * img_size[2]
        self.decoder2 = UnetrUpBlockSDRM(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=stem_kernel_size,
            norm_name=norm_name,
            out_size=dec2_out_size,
            conv_decoder=True,
        )

        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        # SDRM-based encoder
        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits


# =============================================================================
# Parameter count comparison (standalone test)
# =============================================================================

if __name__ == '__main__':
    import torch

    print("=" * 70)
    print("UNETR++ with SDRM — Model Verification")
    print("=" * 70)

    # Create model
    model = UNETR_PP_SDRM(
        in_channels=1,
        out_channels=14,
        img_size=(64, 128, 128),
        feature_size=16,
        num_heads=4,
        norm_name='batch',
        depths=[3, 3, 3, 3],
        dims=[32, 64, 128, 256],
        do_ds=True,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # Test forward pass
    print("\n--- Forward Pass Test ---")
    x = torch.randn(1, 1, 64, 128, 128)
    print(f"Input shape:  {x.shape}")

    model.eval()
    with torch.no_grad():
        output = model(x)

    if isinstance(output, list):
        for i, o in enumerate(output):
            print(f"Output[{i}] shape: {o.shape}")
    else:
        print(f"Output shape: {output.shape}")

    print("\n" + "=" * 70)
    print("Verification complete!")
    print("=" * 70)
