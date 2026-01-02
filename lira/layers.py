"""
Building blocks for Latent Canvas Model.
Minimal, clean implementations inspired by TRM and LLaMA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

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


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to queries and keys."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with RoPE. Bidirectional (no causal mask)."""

    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(x, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention (bidirectional)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int | None = None):
        super().__init__()
        intermediate_size = intermediate_size or int(hidden_size * 8 / 3)
        # Round to nearest multiple of 64 for efficiency
        intermediate_size = ((intermediate_size + 63) // 64) * 64

        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with bidirectional attention."""

    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, max_seq_len)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ProgressEmbedding(nn.Module):
    """
    Embeds iteration progress (0.0 → 1.0) into a hidden dimension vector.

    Uses sinusoidal encoding (like positional encoding) for smooth interpolation,
    plus a learned projection to match hidden size.

    During training, optional noise is added for regularization.
    """

    def __init__(self, hidden_size: int, max_freqs: int = 16, noise_std: float = 0.05):
        super().__init__()
        self.max_freqs = max_freqs
        self.noise_std = noise_std
        # Sinusoidal encoding gives us 2*max_freqs features
        # Project to hidden_size
        self.proj = nn.Linear(max_freqs * 2, hidden_size, bias=False)

    def forward(self, progress: float, device: torch.device, training: bool = True) -> torch.Tensor:
        """
        Args:
            progress: Float in [0, 1] indicating refinement progress
            device: Device to create tensor on
            training: If True, add noise for regularization

        Returns:
            Progress embedding [1, 1, D] broadcastable to [B, L, D]
        """
        # Add noise during training (clamped to [0, 1])
        if training and self.noise_std > 0:
            noise = torch.randn(1, device=device).item() * self.noise_std
            progress = max(0.0, min(1.0, progress + noise))

        # Create sinusoidal features at different frequencies
        freqs = torch.arange(self.max_freqs, device=device).float()
        # Scale progress to [0, pi] for half-period coverage
        angles = progress * math.pi * (2 ** freqs) / (2 ** self.max_freqs)

        # Sin and cos features
        sin_feats = torch.sin(angles)
        cos_feats = torch.cos(angles)
        features = torch.cat([sin_feats, cos_feats], dim=-1)  # [2*max_freqs]

        # Project to hidden size
        embedding = self.proj(features)  # [D]
        return embedding.view(1, 1, -1)  # [1, 1, D] for broadcasting


class RefineBlock(nn.Module):
    """
    A refinement block that processes the latent canvas.
    This is the core unit that gets applied repeatedly (TRM-style).

    Optionally accepts progress signal (0.0 → 1.0) indicating refinement progress.
    """

    def __init__(self, hidden_size: int, num_heads: int, num_layers: int = 2,
                 max_seq_len: int = 2048, use_progress_embedding: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)

        # Optional progress embedding
        self.use_progress_embedding = use_progress_embedding
        if use_progress_embedding:
            self.progress_embed = ProgressEmbedding(hidden_size)

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None,
                progress: float | None = None, training: bool = True) -> torch.Tensor:
        """
        Refine the latent canvas.

        Args:
            z: Latent canvas [B, L, D]
            mask: Optional attention mask
            progress: Optional refinement progress in [0, 1]
                      0.0 = just starting, 1.0 = final iteration
            training: If True, add noise to progress for regularization

        Returns:
            Refined latent canvas [B, L, D]
        """
        # Inject progress signal if available
        if self.use_progress_embedding and progress is not None:
            progress_emb = self.progress_embed(progress, z.device, training=training)
            z = z + progress_emb  # Broadcast [1,1,D] to [B,L,D]

        for layer in self.layers:
            z = layer(z, mask)
        return self.norm(z)
