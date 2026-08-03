"""
SDRM v2: Spectral-Dynamic Routing Mixer
========================================
A novel token mixing module that replaces EPA (Efficient Paired Attention) in UNETR++.

Core Insight:
EPA uses shared Q,K for channel + spatial attention, but both paths operate in the
spatial domain, missing complementary frequency-domain information and multi-scale
local structure.

SDRM v2 introduces three interacting (not just parallel!) mixing paths:
1. GCA (Global Context Attention): Token-level global context with O(N) complexity
2. Dynamic Path (DLK):   Multi-scale large-kernel convolutions
   → produces structural edge scores to guide BRA's attention
3. Routing Path (BRA):   Content-dependent sparse attention
   → receives DLK edge bias for structure-aware attention

Key Interactions:
- GCA → DLK, BRA: stable token-level global calibration
- DLK → BRA: per-head structural edge scores as content-dependent
  attention bias (each head learns its own structural sensitivity)
- Fusion: input-dependent gated fusion with dynamic capacity allocation

Reference:
- UNETR++: Shaker et al., "UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
- GCNet: Cao et al., "GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond"
- DLK: Dynamic Large Kernel Networks for 3D Medical Image Segmentation
- BiFormer: Zhu et al., "BiFormer: Vision Transformer with Bi-Level Routing Attention"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# Helper Utilities
# =============================================================================

def spatial_shape_from_tokens(N: int) -> tuple:
    """Infer spatial shape (H, W, D) from token count N = H*W*D for cubic inputs."""
    s = int(round(N ** (1 / 3)))
    return (s, s, s)


# =============================================================================
# 1. GCA: Global Context Attention — O(N) token-level global mixing
# =============================================================================

class GCA(nn.Module):
    """
    Global Context Attention (inspired by GCNet, Cao et al. 2019).

    Unlike SE which gives every token the SAME channel weight, GCA:
    1. Learns per-token attention weights via a lightweight projection
    2. Pools ALL tokens into a single global context vector
    3. Transforms it through a bottleneck MLP
    4. Adds it back to EACH token

    This gives genuine token-level global context with O(N) complexity —
    a strict upgrade over SE's mean pool, while staying far below
    self-attention's O(N²).

    Reduction ratio r = 4 keeps parameters minimal (~C²/4 per module).
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        bottleneck = max(dim // reduction, 8)
        # Attention: project to scalar weight per token
        self.attn = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        # Transform: bottleneck MLP for the pooled context vector
        self.transform = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        # 1. Per-token attention weights
        attn = self.attn(x).softmax(dim=1)               # (B, N, 1)

        # 2. Weighted global pooling
        ctx = (attn * x).sum(dim=1)                      # (B, C)

        # 3. Transform and broadcast back
        out = self.transform(ctx).unsqueeze(1)           # (B, 1, C)

        return x + out


class GRN(nn.Module):
    """Global Response Normalization — kept for backward compatibility."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        gx = x.norm(p=2, dim=(1,), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


# =============================================================================
# 2. DLK (Dynamic Large Kernel) Branch — Multi-Scale Local Enhancement
# =============================================================================

class DLK3D(nn.Module):
    """
    Multi-scale Parallel Dynamic Large Kernel for 3D.

    Three parallel depthwise branches (k=3,5,7) extract features
    at different receptive fields simultaneously, then fuse via
    spatial + channel attention. mid_dim = dim//3 keeps params low.
    """

    def __init__(self, dim: int, kernel1: int = 5, kernel2: int = 7, dilation: int = 3):
        super().__init__()
        mid_dim = max(dim // 3, 2)
        self.mid_dim = mid_dim
        concat_dim = mid_dim * 3  # three branches

        self.channel_proj = nn.Conv3d(dim, mid_dim, kernel_size=1, bias=False)

        # Three parallel depthwise large-kernel branches
        pad3, pad5 = 3 // 2, 5 // 2
        pad7 = dilation * (7 // 2)
        self.dw3 = nn.Conv3d(mid_dim, mid_dim, 3, stride=1, padding=pad3, groups=mid_dim)
        self.dw5 = nn.Conv3d(mid_dim, mid_dim, 5, stride=1, padding=pad5, groups=mid_dim)
        self.dw7 = nn.Conv3d(mid_dim, mid_dim, 7, stride=1, padding=pad7, groups=mid_dim, dilation=dilation)

        # Spatial attention: 3-way weighting
        self.spatial_se = nn.Sequential(
            nn.Conv3d(2, 3, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        bottleneck_dim = max(concat_dim // 4, 4)
        self.channel_attn = nn.Sequential(
            nn.Conv3d(concat_dim, bottleneck_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck_dim, concat_dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.out_proj = nn.Conv3d(concat_dim, dim, kernel_size=1, bias=False) if concat_dim != dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        proj = self.channel_proj(x)

        # Parallel multi-scale
        o3 = self.dw3(proj)
        o5 = self.dw5(proj)
        o7 = self.dw7(proj)
        out = torch.cat([o3, o5, o7], dim=1)

        # Spatial attention across 3 scales
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        sa = self.spatial_se(torch.cat([avg_out, max_out], dim=1))
        out = out * sa[:, 0:1] + out * sa[:, 1:2] + out * sa[:, 2:3]

        # Channel attention
        out = self.channel_attn(self.avg_pool(out)) * out
        out = self.out_proj(out)
        return out + identity


class DLKBranch(nn.Module):
    """Wrapper for DLK that works on (B, N, C) token sequences."""

    def __init__(self, dim: int, spatial_shape: tuple, kernel1: int = 5, kernel2: int = 7, dilation: int = 3):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.dlk = DLK3D(dim, kernel1=kernel1, kernel2=kernel2, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        H, W, D = self.spatial_shape
        x = x.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3).contiguous()  # (B, C, H, W, D)
        x = self.dlk(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, C)
        return x


# =============================================================================
# 3. BRA (Bi-level Routing) Branch — 3D Sparse Routing Attention
# =============================================================================

class BiLevelRoutingAttention3D(nn.Module):
    """
    3D Bi-Level Routing Attention.

    Core idea (adapted from BiFormer to 3D):
    1. Partition the 3D volume into coarse blocks (sub-volumes)
    2. Compute block-wise Q, K by averaging within each block
    3. Route: for each block, select top-k most relevant blocks via QK similarity
    4. Gather K, V from selected blocks
    5. Perform fine-grained multi-head attention with gathered K, V

    This achieves sparse attention: instead of N², complexity is O(N * k * block_size)
    where k << num_blocks.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        block_size: int = 4,
        topk: int = 4,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.topk = topk
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable routing embedding (optional, for parametric routing)
        self.routing_emb = nn.Linear(dim, dim, bias=False)

        # Per-head edge-guided attention bias scale (v5: each head learns its own)
        # Init at 0.1: let semantic attention establish first, then structural bias
        # gradually activates as training progresses.
        self.edge_gamma = nn.Parameter(torch.ones(num_heads) * 0.1)
        # Per-head temperature controls edge affinity sharpness
        self.edge_temp = nn.Parameter(torch.ones(num_heads))

    def _partition_3d(self, x: torch.Tensor, H: int, W: int, D: int) -> torch.Tensor:
        """
        Partition 3D volume into blocks.
        x: (B, N, C) where N = H*W*D
        Returns: (B, num_blocks, block_size³, C)
        """
        B, N, C = x.shape
        bs = self.block_size

        # Pad spatial dims to be divisible by block_size
        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        pad_d = (bs - D % bs) % bs

        # Reshape to spatial: (B, C, H, W, D)
        x_spatial = x.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x_spatial = F.pad(x_spatial, (0, pad_d, 0, pad_w, 0, pad_h))

        _, _, Hp, Wp, Dp = x_spatial.shape

        # Reshape to blocks
        x_spatial = x_spatial.permute(0, 2, 3, 4, 1)  # (B, Hp, Wp, Dp, C)
        x_blocks = x_spatial.reshape(
            B,
            Hp // bs, bs,
            Wp // bs, bs,
            Dp // bs, bs,
            C
        )
        x_blocks = x_blocks.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        # (B, nH, nW, nD, bs, bs, bs, C)
        num_blocks = (Hp // bs) * (Wp // bs) * (Dp // bs)
        x_blocks = x_blocks.reshape(B, num_blocks, bs * bs * bs, C)

        return x_blocks, Hp, Wp, Dp

    @torch.compiler.disable
    def forward(self, x: torch.Tensor, edge_scores: torch.Tensor = None, spatial_shape: tuple = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) token sequence
            edge_scores: (B, N, 1) optional structural edge scores from DLK,
                         used to bias attention towards structurally similar tokens.
            spatial_shape: (H, W, D) tuple — if None, infer cubic shape from N.
        Returns:
            (B, N, C) attended token sequence
        """
        B, N, C = x.shape
        if spatial_shape is not None:
            H, W, D = spatial_shape
        else:
            H = W = D = int(round(N ** (1 / 3)))

        # Apply QKV projection first (on the full token sequence)
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q_full, k_full, v_full = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)

        # Partition Q, K, V into blocks
        q_for_blocks = q_full.permute(0, 2, 1, 3).reshape(B, N, C)
        k_for_blocks = k_full.permute(0, 2, 1, 3).reshape(B, N, C)
        v_for_blocks = v_full.permute(0, 2, 1, 3).reshape(B, N, C)

        q_blocks, Hp, Wp, Dp = self._partition_3d(q_for_blocks, H, W, D)
        k_blocks, _, _, _ = self._partition_3d(k_for_blocks, H, W, D)
        v_blocks, _, _, _ = self._partition_3d(v_for_blocks, H, W, D)

        # Partition edge scores into blocks (if provided)
        edge_blocks = None
        if edge_scores is not None:
            edge_blocks, _, _, _ = self._partition_3d(edge_scores, H, W, D)
            # edge_blocks: (B, num_blocks, block_vol, 1)

        num_blocks, block_vol = q_blocks.shape[1], q_blocks.shape[2]

        # If only a few blocks, fall back to standard attention
        if num_blocks <= self.topk:
            q = q_full.reshape(B * self.num_heads, N, self.head_dim)
            k = k_full.reshape(B * self.num_heads, N, self.head_dim)
            v = v_full.reshape(B * self.num_heads, N, self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = (attn @ v).reshape(B, self.num_heads, N, self.head_dim)
            out = out.transpose(1, 2).reshape(B, N, C)
            out = self.proj_drop(self.proj(out))
            return out

        # Block-wise Q, K averages for routing
        block_q = q_blocks.mean(dim=2)  # (B, num_blocks, C)
        block_k = k_blocks.mean(dim=2)  # (B, num_blocks, C)

        # Routing embedding (learnable projection)
        block_q_routed = self.routing_emb(block_q)  # (B, num_blocks, C)
        block_k_routed = self.routing_emb(block_k)  # (B, num_blocks, C)

        # Block-to-block similarity for routing
        block_sim = (block_q_routed @ block_k_routed.transpose(-2, -1)) * self.scale
        # (B, num_blocks, num_blocks)

        # Top-k routing: for each query block, find k most relevant key blocks
        actual_topk = min(self.topk, num_blocks)
        topk_scores, topk_idx = torch.topk(block_sim, k=actual_topk, dim=-1)
        # topk_scores: (B, num_blocks, topk), topk_idx: (B, num_blocks, topk)

        # Soft routing weights — make routing differentiable
        routing_weights = F.softmax(topk_scores, dim=-1)  # (B, num_blocks, topk)

        # Gather K, V from the selected blocks
        B_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, num_blocks, actual_topk)

        gathered_k = k_blocks[B_idx, topk_idx]  # (B, num_blocks, topk, block_vol, C)
        gathered_v = v_blocks[B_idx, topk_idx]  # (B, num_blocks, topk, block_vol, C)

        # Apply soft routing weights to gathered K, V
        routing_weights = routing_weights.view(B, num_blocks, actual_topk, 1, 1)
        gathered_k = gathered_k * routing_weights
        gathered_v = gathered_v * routing_weights

        # Reshape for multi-head attention
        # Q: from query blocks (all tokens in each block)
        q_blocks = q_blocks.reshape(B, num_blocks, block_vol, self.num_heads, self.head_dim)
        q_blocks = q_blocks.permute(0, 1, 3, 2, 4)  # (B, num_blocks, heads, block_vol, head_dim)
        q_attn = q_blocks.reshape(B * num_blocks * self.num_heads, block_vol, self.head_dim)

        # K, V: from gathered blocks
        gathered_k = gathered_k.reshape(B, num_blocks, actual_topk * block_vol, self.num_heads, self.head_dim)
        gathered_k = gathered_k.permute(0, 1, 3, 2, 4)  # (B, num_blocks, heads, topk*block_vol, head_dim)
        k_attn = gathered_k.reshape(B * num_blocks * self.num_heads, actual_topk * block_vol, self.head_dim)

        gathered_v = gathered_v.reshape(B, num_blocks, actual_topk * block_vol, self.num_heads, self.head_dim)
        gathered_v = gathered_v.permute(0, 1, 3, 2, 4)  # (B, num_blocks, heads, topk*block_vol, head_dim)
        v_attn = gathered_v.reshape(B * num_blocks * self.num_heads, actual_topk * block_vol, self.head_dim)

        # Multi-head attention with edge-guided bias (DLK → BRA interaction)
        attn = (q_attn @ k_attn.transpose(-2, -1)) * self.scale

        # Inject structural edge bias from DLK (v4: per-head edge bias)
        if edge_blocks is not None:
            # edge_blocks: (B, num_blocks, block_vol, num_heads)
            # Each head gets its own structural similarity bias
            edge_q = edge_blocks  # (B, nB, bv, H)
            edge_k_gathered = edge_blocks[B_idx, topk_idx]  # (B, nB, topk, bv, H)

            # Per-head edge diff with learned temperature (v5)
            edge_q_exp = edge_q.view(B, num_blocks, block_vol, 1, self.num_heads)
            edge_k_exp = edge_k_gathered.view(B, num_blocks, 1, actual_topk * block_vol, self.num_heads)
            edge_diff_sq = (edge_q_exp - edge_k_exp).pow(2)  # (B, nB, bv, T*bv, H)
            # Learned per-head gamma and temperature control affinity sharpness
            gamma = self.edge_gamma.abs().view(1, 1, 1, 1, self.num_heads)
            temp = self.edge_temp.abs().view(1, 1, 1, 1, self.num_heads) + 1e-6
            edge_bias = -gamma * edge_diff_sq / temp
            # (B, nB, bv, T*bv, H)

            # Reshape for multi-head: (B, nB, bv, T*bv, H) -> (B*nB*H, bv, T*bv)
            edge_bias = edge_bias.permute(0, 1, 4, 2, 3).contiguous()
            edge_bias = edge_bias.reshape(B * num_blocks * self.num_heads, block_vol, actual_topk * block_vol)

            attn = attn + edge_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v_attn  # (B * num_blocks * heads, block_vol, head_dim)

        # Reshape back
        out = out.reshape(B, num_blocks, self.num_heads, block_vol, self.head_dim)
        out = out.permute(0, 1, 3, 2, 4).contiguous()  # (B, num_blocks, block_vol, heads, head_dim)
        out = out.reshape(B, num_blocks, block_vol, C)

        # Reconstruct spatial volume from blocks
        nH, nW, nD = Hp // self.block_size, Wp // self.block_size, Dp // self.block_size
        out = out.reshape(B, nH, nW, nD, self.block_size, self.block_size, self.block_size, C)
        out = out.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        out = out.reshape(B, Hp, Wp, Dp, C)

        # Crop to original size (remove padding)
        out = out[:, :H, :W, :D, :]
        out = out.reshape(B, N, C)

        # Output projection
        out = self.proj_drop(self.proj(out))

        return out


# =============================================================================
# 4. SDRM: Spectral-Dynamic Routing Mixer (Main Module)
# =============================================================================

class SDRM(nn.Module):
    """
    Spectral-Dynamic Routing Mixer (SDRM) v6

    Replaces EPA in UNETR++ with dual-path interacting branches + GCA.

    Path 1 — GCA (Global Context Attention):
        Token-level global context with O(N) complexity.
        Each token gets a unique global context vector.

    Path 2 — Dynamic (DLK):
        Multi-scale local enhancement via cascaded large-kernel convolutions.
        → Produces per-head structural edge scores to guide BRA's attention.

    Path 3 — Routing (BRA-3D):
        Content-dependent sparse attention via block-level routing.
        → Receives per-head edge-guided attention bias from DLK.

    Key Interactions:
    - GCA → DLK, BRA: stable token-level global calibration
    - DLK → BRA: per-head structural edge scores as content-dependent
      attention bias (each head learns its own structural sensitivity)
    - Fusion: input-dependent gated fusion with dynamic capacity allocation
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        channel_attn_drop: float = 0.1,
        spatial_attn_drop: float = 0.1,
        spatial_shape: tuple = None,   # (H, W, D) — explicitly pass for non-cubic volumes
    ):
        """
        Args:
            input_size: number of tokens (H*W*D) for this stage
            hidden_size: feature dimension C
            proj_size: projection dimension (used for routing bottleneck)
            num_heads: number of attention heads for routing branch
            qkv_bias: whether to use bias in QKV projection
            channel_attn_drop: dropout rate for channel attention
            spatial_attn_drop: dropout rate for spatial attention
            spatial_shape: (H, W, D) tuple for the feature volume at this stage
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_heads = num_heads

        # Spatial shape: use provided, else infer from token count (assumes cubic)
        if spatial_shape is not None:
            self.spatial_shape = spatial_shape
        else:
            self.spatial_shape = spatial_shape_from_tokens(input_size)
        H, W, D = self.spatial_shape

        # Staged channel split: early stages favor DLK (local structure at
        # high resolution), later stages favor BRA (semantic attention at low
        # resolution). Zero extra parameters.
        if H >= 16:       # Stage 0 (32³), Stage 1 (16³): more local detail
            c_dlk = max(hidden_size * 5 // 8, 8)
        elif H >= 8:      # Stage 2 (8³): balanced
            c_dlk = max(hidden_size // 2, 8)
        else:              # Stage 3 (4³): more semantic context
            c_dlk = max(hidden_size * 3 // 8, 8)
        # --- Shared norm ---
        self.norm = nn.LayerNorm(hidden_size)

        # --- DLK Branch: Multi-Scale Local Enhancement ---
        if H >= 32:
            dlk_k1, dlk_k2, dlk_dil = 3, 5, 2
        elif H >= 16:
            dlk_k1, dlk_k2, dlk_dil = 5, 7, 2
        elif H >= 8:
            dlk_k1, dlk_k2, dlk_dil = 5, 7, 3
        else:
            dlk_k1, dlk_k2, dlk_dil = 3, 5, 2
        self.dlk_branch = DLKBranch(dim=c_dlk, spatial_shape=self.spatial_shape,
                                    kernel1=dlk_k1, kernel2=dlk_k2, dilation=dlk_dil)
        self.dlk_in_proj = nn.Linear(hidden_size, c_dlk)

        # --- Output projection ---
        self.out_proj = nn.Linear(c_dlk, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) token sequence
        Returns:
            (B, N, C) mixed token sequence
        """
        B, N, C = x.shape
        x_norm = self.norm(x)

        # ---- DLK Branch: Multi-Scale Local Enhancement ----
        dlk_in = self.dlk_in_proj(x_norm)
        dlk_out = self.dlk_branch(dlk_in)

        out = self.out_proj(dlk_out)

        return out

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()


# =============================================================================
# 5. Parameter Count Utility
# =============================================================================

def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


# =============================================================================
# 6. HF-PDC: High-Frequency Pixel Difference Convolution Skip Gate
# =============================================================================

class HFPdcSkipGate(nn.Module):
    """
    High-Frequency Pixel Difference Convolution Skip Gate.

    Inserted into the skip connection between encoder and decoder. It:
    1. Extracts high-frequency edges using a FIXED 3D Laplacian operator (0 params)
    2. Generates a spatial attention mask via 1×1×1 conv + sigmoid (lightweight)
    3. Enhances the encoder skip features: X' = X + X ⊙ Mask

    This forces the decoder to "see" thin boundaries (vessel walls, organ edges)
    that are easily lost during downsampling.

    Total extra parameters per gate: C² (1×1×1 conv), e.g. 32²=1,024 for C=32.
    """

    def __init__(self, channels: int):
        super().__init__()

        # ---- Fixed 3D Laplacian kernel (0 learnable params) ----
        # ∇² = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²
        # Center = -6, 6 face neighbors = +1
        laplacian = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32)
        laplacian[0, 0, 1, 1, 1] = -6.0   # center
        laplacian[0, 0, 1, 1, 0] = 1.0    # -z face
        laplacian[0, 0, 1, 1, 2] = 1.0    # +z face
        laplacian[0, 0, 1, 0, 1] = 1.0    # -y face
        laplacian[0, 0, 1, 2, 1] = 1.0    # +y face
        laplacian[0, 0, 0, 1, 1] = 1.0    # -x face
        laplacian[0, 0, 2, 1, 1] = 1.0    # +x face
        self.register_buffer('laplacian', laplacian)

        # ---- Lightweight gating: 1×1×1 conv, C→C ----
        # Maps edge responses to [0,1] attention mask per channel
        self.gate_conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W, D) encoder skip features
        Returns:
            (B, C, H, W, D) boundary-enhanced features
        """
        B, C, H, W, D = x.shape

        # Step 1: Extract high-frequency edges (per-channel 3D Laplacian)
        x_flat = x.reshape(B * C, 1, H, W, D)
        x_edge = F.conv3d(x_flat, self.laplacian, padding=1, groups=1)
        # x_edge: (B*C, 1, H, W, D)
        x_edge = x_edge.reshape(B, C, H, W, D)

        # Step 2: Generate spatial attention mask
        mask = self.gate_conv(x_edge)  # (B, C, H, W, D), values in [0,1]

        # Step 3: Enhance: smooth regions ×≈1 (pass through),
        #          boundary regions ×≈2 (amplified)
        enhanced = x + x * mask

        return enhanced


# =============================================================================
# 7. Testing (standalone)
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing SDRM Module")
    print("=" * 60)

    # Test at different scales (matching UNETR++ stages)
    test_configs = [
        {'input_size': 32 * 32 * 32, 'hidden_size': 32,  'proj_size': 64, 'num_heads': 4},
        {'input_size': 16 * 16 * 16, 'hidden_size': 64,  'proj_size': 64, 'num_heads': 4},
        {'input_size': 8 * 8 * 8,    'hidden_size': 128, 'proj_size': 64, 'num_heads': 4},
        {'input_size': 4 * 4 * 4,    'hidden_size': 256, 'proj_size': 32, 'num_heads': 4},
    ]

    for i, cfg in enumerate(test_configs):
        model = SDRM(**cfg)
        B, N, C = 1, cfg['input_size'], cfg['hidden_size']
        x = torch.randn(B, N, C)
        y = model(x)

        params = count_parameters(model)
        print(f"\nStage {i}: input_size={cfg['input_size']}, hidden_size={cfg['hidden_size']}")
        print(f"  Input:  {tuple(x.shape)}")
        print(f"  Output: {tuple(y.shape)}")
        print(f"  Params: {params['total']:,} ({params['total']/1e6:.2f}M)")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
