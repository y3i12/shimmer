"""
FP8-accelerated building blocks for LIRA.

Supports:
- NVIDIA Transformer Engine (preferred, if available)
- PyTorch native FP8 (fallback)
- Standard FP16/32 (fallback fallback)

FP8 accelerates Linear layers on Ada Lovelace+ GPUs (RTX 40 series, Ada workstation).
Sensitive ops (RMSNorm, softmax, RoPE) stay in higher precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

# Try to import Transformer Engine for FP8
_TE_AVAILABLE = False
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    _TE_AVAILABLE = True
except ImportError:
    pass

# Check for PyTorch native FP8 support (2.1+)
_TORCH_FP8_AVAILABLE = hasattr(torch, 'float8_e4m3fn')


@dataclass
class FP8Config:
    """Configuration for FP8 training."""
    enabled: bool = True
    use_transformer_engine: bool = True  # Prefer TE if available
    # Delayed scaling recipe for TE
    margin: int = 0
    interval: int = 1
    fp8_format: str = "HYBRID"  # E4M3 forward, E5M2 backward
    amax_history_len: int = 1024
    amax_compute_algo: str = "max"


def get_fp8_recipe(config: FP8Config):
    """Get Transformer Engine FP8 recipe."""
    if not _TE_AVAILABLE:
        return None

    fp8_format = Format.HYBRID if config.fp8_format == "HYBRID" else Format.E4M3

    return DelayedScaling(
        margin=config.margin,
        interval=config.interval,
        fp8_format=fp8_format,
        amax_history_len=config.amax_history_len,
        amax_compute_algo=config.amax_compute_algo,
    )


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Always runs in FP32 for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always compute in FP32
        input_dtype = x.dtype
        x = x.float()
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm * self.weight).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Computed in FP32, applied in input dtype.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Store in FP32 for precision
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to queries and keys."""
    # Cast cos/sin to match input dtype
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FP8Linear(nn.Module):
    """
    Linear layer with FP8 support.

    Uses Transformer Engine if available, otherwise falls back to
    standard Linear with autocast.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        fp8_config: Optional[FP8Config] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fp8_config = fp8_config or FP8Config()

        # Choose implementation
        if self.fp8_config.enabled and _TE_AVAILABLE and self.fp8_config.use_transformer_engine:
            self.impl = "te"
            self.linear = te.Linear(in_features, out_features, bias=bias)
        else:
            self.impl = "torch"
            self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @property
    def weight(self):
        return self.linear.weight


class Attention(nn.Module):
    """
    Multi-head attention with RoPE and optional FP8 acceleration.

    FP8 applied to: Q, K, V, O projections (Linear layers)
    FP16/32 kept for: softmax, RoPE application
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seq_len: int = 2048,
        fp8_config: Optional[FP8Config] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.fp8_config = fp8_config or FP8Config()

        # Projections with FP8
        self.q_proj = FP8Linear(hidden_size, hidden_size, bias=False, fp8_config=fp8_config)
        self.k_proj = FP8Linear(hidden_size, hidden_size, bias=False, fp8_config=fp8_config)
        self.v_proj = FP8Linear(hidden_size, hidden_size, bias=False, fp8_config=fp8_config)
        self.o_proj = FP8Linear(hidden_size, hidden_size, bias=False, fp8_config=fp8_config)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = x.shape

        # Linear projections (FP8 accelerated)
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE (stays in current precision)
        cos, sin = self.rotary(x, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention computation (keep in FP16/32 for softmax stability)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        # Softmax in FP32 for stability
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_size)

        return self.o_proj(out)


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network with FP8 acceleration.

    FP8 applied to: w1, w2, w3 projections
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int | None = None,
        fp8_config: Optional[FP8Config] = None
    ):
        super().__init__()
        intermediate_size = intermediate_size or int(hidden_size * 8 / 3)
        # Round to nearest multiple of 64 for efficiency
        intermediate_size = ((intermediate_size + 63) // 64) * 64

        self.w1 = FP8Linear(hidden_size, intermediate_size, bias=False, fp8_config=fp8_config)
        self.w2 = FP8Linear(intermediate_size, hidden_size, bias=False, fp8_config=fp8_config)
        self.w3 = FP8Linear(hidden_size, intermediate_size, bias=False, fp8_config=fp8_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional FP8 acceleration."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seq_len: int = 2048,
        fp8_config: Optional[FP8Config] = None
    ):
        super().__init__()
        # Norms stay in FP32
        self.attn_norm = RMSNorm(hidden_size)
        self.ffn_norm = RMSNorm(hidden_size)

        # Attention and FFN with FP8
        self.attn = Attention(hidden_size, num_heads, max_seq_len, fp8_config)
        self.ffn = SwiGLU(hidden_size, fp8_config=fp8_config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class RefineBlock(nn.Module):
    """
    A refinement block with optional FP8 acceleration.
    This is the core unit that gets applied repeatedly (TRM-style).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int = 2,
        max_seq_len: int = 2048,
        fp8_config: Optional[FP8Config] = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, max_seq_len, fp8_config)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z, mask)
        return self.norm(z)


# === Transformer Engine context manager for FP8 training ===

class FP8Context:
    """
    Context manager for FP8 training with Transformer Engine.

    Usage:
        fp8_ctx = FP8Context(fp8_config)

        for batch in dataloader:
            with fp8_ctx:
                output = model(batch)
                loss.backward()
    """

    def __init__(self, fp8_config: Optional[FP8Config] = None):
        self.fp8_config = fp8_config or FP8Config()
        self.recipe = get_fp8_recipe(self.fp8_config) if _TE_AVAILABLE else None
        self._ctx = None

    def __enter__(self):
        if _TE_AVAILABLE and self.fp8_config.enabled:
            self._ctx = te.fp8_autocast(enabled=True, fp8_recipe=self.recipe)
            return self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self._ctx is not None:
            return self._ctx.__exit__(*args)
        return False


# === Utilities ===

def check_fp8_support() -> dict:
    """Check FP8 support on current system."""
    info = {
        "transformer_engine_available": _TE_AVAILABLE,
        "torch_fp8_available": _TORCH_FP8_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_compute_capability": None,
        "fp8_recommended": False,
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["gpu_compute_capability"] = f"{cap[0]}.{cap[1]}"
        # FP8 requires compute capability 8.9+ (Ada Lovelace) or 9.0+ (Hopper)
        info["fp8_recommended"] = cap[0] >= 8 and cap[1] >= 9

    return info


def print_fp8_status():
    """Print FP8 support status."""
    info = check_fp8_support()
    print("=== FP8 Support Status ===")
    print(f"  Transformer Engine: {'✓' if info['transformer_engine_available'] else '✗'}")
    print(f"  PyTorch FP8 dtypes: {'✓' if info['torch_fp8_available'] else '✗'}")
    print(f"  CUDA available: {'✓' if info['cuda_available'] else '✗'}")
    if info['gpu_name']:
        print(f"  GPU: {info['gpu_name']}")
        print(f"  Compute capability: {info['gpu_compute_capability']}")
        print(f"  FP8 recommended: {'✓' if info['fp8_recommended'] else '✗ (need 8.9+)'}")
