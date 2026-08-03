"""
UNETR++ with SDRM (Spectral-Dynamic Routing Mixer)

This module replaces the EPA (Efficient Paired Attention) in UNETR++ with
our proposed SDRM module, which combines:

1. GCA (Global Context Attention): Token-level global context with O(N) complexity
2. DLK (Dynamic Large Kernel): Multi-scale local enhancement via large-kernel convolutions
3. BRA (Bi-Level Routing Attention): Content-dependent sparse attention via block-level routing

Key components:
- SDRM:              The core mixing module replacing EPA
- TransformerBlockSDRM: Transformer block using SDRM
- UnetrPPEncoderSDRM:   Encoder with SDRM-based transformer blocks
- UnetrUpBlockSDRM:     Decoder block with SDRM-based transformer blocks
- UNETR_PP_SDRM:        The full UNETR++ model with SDRM
"""

from .sdrm import SDRM
from .transformerblock import TransformerBlockSDRM
from .model_components import UnetrPPEncoderSDRM, UnetrUpBlockSDRM
from .unetr_pp_sdrm import UNETR_PP_SDRM

__all__ = [
    'SDRM',
    'TransformerBlockSDRM',
    'UnetrPPEncoderSDRM',
    'UnetrUpBlockSDRM',
    'UNETR_PP_SDRM',
]
