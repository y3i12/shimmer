"""
LIRA: Latent Iterative Refinement Architecture
Core components for the Shimmer language model.
"""

from .layers import RMSNorm, RotaryEmbedding, Attention, SwiGLU, TransformerBlock, RefineBlock
from .canvas import LatentCanvasConfig, LatentCanvasModel, count_parameters
from .mamba import MambaConfig, MambaBlock, MambaLayer, HybridBlock, HybridRefineBlock
from .hybrid import HybridConfig, HybridShimmer, GlobalAttentionLayer, GlobalCoherenceBlock, count_global_parameters
from .gpt import GPTConfig, GPTModel, CausalAttention, GPTBlock, count_gpt_parameters

# FP8 support (optional - graceful fallback if not available)
try:
    from .layers_fp8 import (
        FP8Config,
        FP8Context,
        FP8Linear,
        check_fp8_support,
        print_fp8_status,
        Attention as FP8Attention,
        SwiGLU as FP8SwiGLU,
        TransformerBlock as FP8TransformerBlock,
        RefineBlock as FP8RefineBlock,
    )
    _FP8_AVAILABLE = True
except ImportError:
    _FP8_AVAILABLE = False

__all__ = [
    # Standard layers
    "RMSNorm",
    "RotaryEmbedding",
    "Attention",
    "SwiGLU",
    "TransformerBlock",
    "RefineBlock",
    # Mamba / Hybrid layers
    "MambaConfig",
    "MambaBlock",
    "MambaLayer",
    "HybridBlock",
    "HybridRefineBlock",
    # LIRA Model
    "LatentCanvasConfig",
    "LatentCanvasModel",
    "count_parameters",
    # GPT Baseline Model
    "GPTConfig",
    "GPTModel",
    "CausalAttention",
    "GPTBlock",
    "count_gpt_parameters",
    # Hybrid Shimmer (Global Coherence)
    "HybridConfig",
    "HybridShimmer",
    "GlobalAttentionLayer",
    "GlobalCoherenceBlock",
    "count_global_parameters",
    # FP8 (if available)
    "FP8Config",
    "FP8Context",
    "FP8Linear",
    "FP8Attention",
    "FP8SwiGLU",
    "FP8TransformerBlock",
    "FP8RefineBlock",
    "check_fp8_support",
    "print_fp8_status",
]
