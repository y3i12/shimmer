"""
LIRA: Latent Iterative Refinement Architecture
Core components for the Shimmer language model.
"""

from .layers import RMSNorm, RotaryEmbedding, Attention, SwiGLU, TransformerBlock, RefineBlock
from .canvas import LatentCanvasConfig, LatentCanvasModel, count_parameters

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "Attention",
    "SwiGLU",
    "TransformerBlock",
    "RefineBlock",
    "LatentCanvasConfig",
    "LatentCanvasModel",
    "count_parameters",
]
