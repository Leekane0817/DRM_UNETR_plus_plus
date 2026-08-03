# UNETR++ SDRM — Spectral-Dynamic Routing Mixer

A novel feature mixing module for 3D medical image segmentation, replacing Efficient Paired Attention (EPA) in [UNETR++](https://github.com/AbdelrahmanShaker/UNETR_plus_plus).

## Architecture

```
UNETR_PP_SDRM (U-Net backbone)
├── Encoder (UnetrPPEncoderSDRM)
│   ├── Stem Conv + 3× Downsample Layers
│   └── 4 stages × N TransformerBlockSDRM
│       └── SDRM (replaces EPA)
│           ├── GCA  — Global Context Attention (O(N) token-level)
│           ├── DLK  — Dynamic Large Kernel (multi-scale local)
│           ├── BRA  — Bi-Level Routing Attention (sparse global)
│           └── DLK edge_scores → BRA attention bias
├── Bottleneck (256-dim → 4³ spatial)
├── Decoder (4× UnetrUpBlockSDRM)
│   ├── Transposed Conv upsample
│   ├── HF-PDC Skip Gate (boundary enhancement)
│   └── SDRM Transformer Blocks
└── Output (1×1×1 Conv + Deep Supervision)
```

## Key Components

| Component | Description |
|-----------|-------------|
| **GCA** | Token-level global context with O(N) complexity — learns per-token attention weights |
| **DLK** | Multi-scale large-kernel convolutions (3×3, 5×5, 7×7 DW) with spatial+channel attention |
| **BRA** | Bi-Level Routing Attention — partitions volume into blocks, routes top-k, sparse multi-head attention |
| **SDRM** | Combines GCA + DLK + BRA with staged channel split & gated fusion |
| **HF-PDC** | High-Frequency Pixel Difference Convolution — fixed 3D Laplacian enhances skip-connection boundaries (0 extra params) |

## Key Innovation: DLK → BRA Edge-Guided Attention

The DLK branch extracts structural edge features that serve as **per-head attention bias** for BRA. Each attention head receives its own structural similarity score, enabling diverse structure-aware attention patterns:

```
input → GCA → DLK ─→ edge_scores (per-head) ─→ BRA attention bias
            │                    └────────────────┘
            └→ BRA ←──────────────────────────────┘
```

## File Structure

```
acdc_sdrm_model/
├── model/
│   ├── __init__.py          # Package exports
│   ├── sdrm.py              # Core SDRM module + GCA, DLK, BRA, HF-PDC
│   ├── transformerblock.py  # TransformerBlockSDRM (wraps SDRM)
│   ├── model_components.py  # Encoder/Decoder blocks
│   └── unetr_pp_sdrm.py     # Full UNETR_PP_SDRM model
├── trainer/
│   ├── __init__.py
│   └── trainer_acdc.py      # Trainer for ACDC cardiac dataset
├── scripts/
│   ├── train.sh             # Training launch script
│   └── validate.sh          # Validation launch script
├── tests/
│   └── test_compat.py       # Compatibility tests
├── benchmark/
│   └── all_methods_dice.txt # ACDC benchmark results
└── README.md
```

## Usage

### Model Instantiation

```python
from model.unetr_pp_sdrm import UNETR_PP_SDRM

# Synapse (14 classes, cubic patches)
model = UNETR_PP_SDRM(
    in_channels=1,
    out_channels=14,
    img_size=(64, 128, 128),
    feature_size=16,
    num_heads=4,
    depths=[3, 3, 3, 3],
    dims=[32, 64, 128, 256],
    do_ds=True,
    stem_kernel_size=(2, 4, 4),  # downsample z-axis
)

# ACDC (4 classes, anisotropic patches)
model = UNETR_PP_SDRM(
    in_channels=1,
    out_channels=4,              # BG + RV + MYO + LV
    img_size=(16, 160, 160),
    feature_size=16,
    num_heads=4,
    depths=[3, 3, 3, 3],
    dims=[32, 64, 128, 256],
    do_ds=True,
    stem_kernel_size=(1, 4, 4),  # preserve z-axis
)
```

### Training on ACDC

```bash
# Set up PYTHONPATH
export PYTHONPATH="/path/to/unetr_plus_plus:/path/to/acdc_sdrm_model:$PYTHONPATH"

# Train
bash scripts/train.sh

# Validate
bash scripts/validate.sh
```

## ACDC Benchmark Results

On the ACDC test set (40 frames, 70/10/20 split):

| Method     | RV     | MYO    | LV     | **Mean** |
|------------|--------|--------|--------|----------|
| **Ours SDRM** | **0.911** | **0.901** | **0.959** | **0.924** |
| nnFormer   | 0.909  | 0.896  | 0.957  | 0.921    |
| nnUNet     | 0.902  | 0.892  | 0.954  | 0.916    |
| UNETR      | 0.853  | 0.865  | 0.940  | 0.886    |

## Dependencies

- PyTorch >= 1.12
- [UNETR++](https://github.com/AbdelrahmanShaker/UNETR_plus_plus) (nnU-Net based framework)
- MONAI, einops, timm
- batchgenerators

## Reference

- UNETR++: Shaker et al., "UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation", IEEE TMI, 2024
- GCNet: Cao et al., "GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond", ICCVW 2019
- BiFormer: Zhu et al., "BiFormer: Vision Transformer with Bi-Level Routing Attention", CVPR 2023
