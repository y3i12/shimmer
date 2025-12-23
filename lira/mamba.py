"""
Pure PyTorch Mamba implementation for validation.

This is a reference implementation - NOT optimized for speed.
For production, use mamba-ssm with CUDA kernels.

Based on: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
https://arxiv.org/abs/2312.00752
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass


@dataclass
class MambaConfig:
    """Configuration for Mamba block."""
    hidden_size: int = 256
    state_size: int = 16       # N: SSM state dimension
    expand_factor: int = 2     # E: expansion factor for inner dim
    conv_kernel: int = 4       # Causal conv1d kernel size
    dt_rank: str | int = "auto"  # Rank for delta projection
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"    # "random" or "constant"
    dt_scale: float = 1.0
    bias: bool = False
    conv_bias: bool = True

    def __post_init__(self):
        self.inner_size = self.hidden_size * self.expand_factor
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.hidden_size / 16)


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model - the core of Mamba.

    Key insight: B, C, and delta are INPUT-DEPENDENT (selective),
    allowing the model to filter information based on content.

    State space equation:
        h'(t) = A h(t) + B x(t)
        y(t) = C h(t) + D x(t)

    Discretized (with input-dependent delta):
        h[k] = Ā h[k-1] + B̄ x[k]
        y[k] = C h[k] + D x[k]

    Where Ā = exp(delta * A), B̄ = (delta * A)^(-1) (exp(delta * A) - I) * delta * B
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_inner = config.inner_size
        d_state = config.state_size
        dt_rank = config.dt_rank

        # A is not input-dependent (learned log values for stability)
        # Shape: (d_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D is a simple skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

        # Projections for input-dependent B, C, delta
        # These make the SSM "selective"
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)

        # Delta projection (from dt_rank to d_inner)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize dt_proj bias for proper delta range
        dt_init_std = dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Inverse softplus for bias initialization
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias = nn.Parameter(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D_inner]

        Returns:
            Output tensor [B, L, D_inner]
        """
        B, L, D = x.shape
        d_state = self.config.state_size
        dt_rank = self.config.dt_rank

        # Get A (negative for stability)
        A = -torch.exp(self.A_log.float())  # [D, N]

        # Project x to get delta, B, C (the selective parts)
        x_proj = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]

        delta_proj = x_proj[:, :, :dt_rank]                    # [B, L, dt_rank]
        B_proj = x_proj[:, :, dt_rank:dt_rank + d_state]       # [B, L, N]
        C_proj = x_proj[:, :, dt_rank + d_state:]              # [B, L, N]

        # Get delta (positive, input-dependent time step)
        delta = F.softplus(self.dt_proj(delta_proj))  # [B, L, D]

        # Discretize A and B
        # Ā = exp(delta * A)
        deltaA = torch.einsum("bld,dn->bldn", delta, A)  # [B, L, D, N]
        A_bar = torch.exp(deltaA)

        # B̄ = delta * B (simplified discretization)
        deltaB = torch.einsum("bld,bln->bldn", delta, B_proj)  # [B, L, D, N]

        # === The Selective Scan (Sequential - this is what CUDA kernels optimize) ===
        # This is O(L) sequential, which is why we need CUDA kernels for speed
        y = self._selective_scan(x, A_bar, deltaB, C_proj)

        # Add skip connection (D * x)
        y = y + x * self.D

        return y

    def _selective_scan(
        self,
        x: torch.Tensor,      # [B, L, D]
        A_bar: torch.Tensor,  # [B, L, D, N]
        B_bar: torch.Tensor,  # [B, L, D, N]
        C: torch.Tensor,      # [B, L, N]
    ) -> torch.Tensor:
        """
        The selective scan operation.

        h[k] = A_bar[k] * h[k-1] + B_bar[k] * x[k]
        y[k] = C[k] @ h[k]

        This is sequential and is the bottleneck in pure PyTorch.
        CUDA kernels parallelize this via work-efficient parallel scan.
        """
        B, L, D = x.shape
        N = C.shape[-1]

        # Initialize hidden state
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)

        outputs = []
        for i in range(L):
            # h = A_bar * h + B_bar * x
            h = A_bar[:, i] * h + B_bar[:, i] * x[:, i:i+1].transpose(-1, -2)

            # y = C @ h (sum over state dimension)
            y_i = torch.einsum("bdn,bn->bd", h, C[:, i])
            outputs.append(y_i)

        return torch.stack(outputs, dim=1)  # [B, L, D]


class MambaBlock(nn.Module):
    """
    Complete Mamba block: Conv1d → SSM → Output projection.

    Architecture:
        x → Linear (expand) → split
                              ├→ Conv1d → SiLU → SSM → × ←┐
                              └→ SiLU ────────────────────┘
                                                    → Linear (contract) → y
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_inner = config.inner_size

        # Input projection (expand)
        self.in_proj = nn.Linear(config.hidden_size, d_inner * 2, bias=config.bias)

        # Causal conv1d (depthwise)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=config.conv_kernel,
            padding=config.conv_kernel - 1,
            groups=d_inner,  # Depthwise
            bias=config.conv_bias,
        )

        # Selective SSM
        self.ssm = SelectiveSSM(config)

        # Output projection (contract)
        self.out_proj = nn.Linear(d_inner, config.hidden_size, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]

        Returns:
            Output tensor [B, L, D]
        """
        B, L, D = x.shape

        # Expand and split
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_conv, z = xz.chunk(2, dim=-1)  # Each [B, L, d_inner]

        # Causal conv1d (need to transpose for conv1d)
        x_conv = x_conv.transpose(1, 2)  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[:, :, :L]  # Causal: truncate to original length
        x_conv = x_conv.transpose(1, 2)  # [B, L, d_inner]

        # Activation after conv
        x_conv = F.silu(x_conv)

        # Selective SSM
        x_ssm = self.ssm(x_conv)

        # Gated output (multiply with activated z)
        y = x_ssm * F.silu(z)

        # Contract back to hidden size
        return self.out_proj(y)


class MambaLayer(nn.Module):
    """
    Full Mamba layer with normalization and residual.
    Drop-in replacement for TransformerBlock.
    """

    def __init__(self, hidden_size: int, state_size: int = 16, expand_factor: int = 2):
        super().__init__()
        from .layers import RMSNorm

        config = MambaConfig(
            hidden_size=hidden_size,
            state_size=state_size,
            expand_factor=expand_factor,
        )

        self.norm = RMSNorm(hidden_size)
        self.mamba = MambaBlock(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
            mask: Ignored (Mamba doesn't use attention mask)

        Returns:
            Output tensor [B, L, D]
        """
        # Pre-norm + Mamba + residual
        return x + self.mamba(self.norm(x))


# === Hybrid Components ===

class HybridBlock(nn.Module):
    """
    Parallel hybrid block: Attention AND Mamba process same input (Hymba-style).

    Input → ┬─ Attention ─┬→ weighted sum → Output
            └─ Mamba ─────┘
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        state_size: int = 16,
        mamba_weight: float = 0.5,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        from .layers import RMSNorm, Attention, SwiGLU

        self.mamba_weight = mamba_weight
        self.attn_weight = 1.0 - mamba_weight

        # Normalization
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

        # Parallel heads
        self.attention = Attention(hidden_size, num_heads, max_seq_len)
        self.mamba = MambaBlock(MambaConfig(hidden_size=hidden_size, state_size=state_size))

        # FFN
        self.ffn = SwiGLU(hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Parallel attention + mamba
        normed = self.norm1(x)
        attn_out = self.attention(normed, mask)
        mamba_out = self.mamba(normed)

        # Weighted combination
        combined = self.attn_weight * attn_out + self.mamba_weight * mamba_out
        x = x + combined

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x


class HybridRefineBlock(nn.Module):
    """
    Hybrid refinement block supporting multiple modes:

    Modes:
        - "attention": Pure attention (original Shimmer)
        - "mamba": Pure Mamba
        - "parallel": Hymba-style parallel heads
        - "interleaved": Nemotron-style interleaved layers
        - "adaptive": Iteration-aware switching (coarse→fine)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int = 4,
        mode: str = "adaptive",
        mamba_ratio: float = 0.75,  # For interleaved: ratio of Mamba layers
        state_size: int = 16,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        from .layers import RMSNorm, TransformerBlock

        self.mode = mode
        self.mamba_ratio = mamba_ratio
        self.num_layers = num_layers

        if mode == "attention":
            self.layers = nn.ModuleList([
                TransformerBlock(hidden_size, num_heads, max_seq_len)
                for _ in range(num_layers)
            ])

        elif mode == "mamba":
            self.layers = nn.ModuleList([
                MambaLayer(hidden_size, state_size)
                for _ in range(num_layers)
            ])

        elif mode == "parallel":
            self.layers = nn.ModuleList([
                HybridBlock(hidden_size, num_heads, state_size, max_seq_len=max_seq_len)
                for _ in range(num_layers)
            ])

        elif mode == "interleaved":
            # Nemotron-style: mostly Mamba with some Attention
            num_mamba = int(num_layers * mamba_ratio)
            num_attn = num_layers - num_mamba

            self.layers = nn.ModuleList()
            attn_positions = self._get_attn_positions(num_layers, num_attn)

            for i in range(num_layers):
                if i in attn_positions:
                    self.layers.append(TransformerBlock(hidden_size, num_heads, max_seq_len))
                else:
                    self.layers.append(MambaLayer(hidden_size, state_size))

        elif mode == "adaptive":
            # Both types available, switch based on iteration
            self.mamba_layers = nn.ModuleList([
                MambaLayer(hidden_size, state_size)
                for _ in range(num_layers)
            ])
            self.attn_layers = nn.ModuleList([
                TransformerBlock(hidden_size, num_heads, max_seq_len)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.norm = RMSNorm(hidden_size)

    def _get_attn_positions(self, total: int, num_attn: int) -> set:
        """Distribute attention layers: first, middle, last (like Hymba)."""
        if num_attn == 0:
            return set()
        if num_attn == 1:
            return {total - 1}  # Last only
        if num_attn == 2:
            return {0, total - 1}  # First and last
        # First, middle, last, then distribute rest
        positions = {0, total // 2, total - 1}
        # Add more if needed
        remaining = num_attn - len(positions)
        step = total // (remaining + 1) if remaining > 0 else total
        for i in range(remaining):
            positions.add((i + 1) * step)
        return positions

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        iteration: int = 0,
        total_iterations: int = 1,
    ) -> torch.Tensor:
        """
        Refine the latent canvas.

        Args:
            z: Latent canvas [B, L, D]
            mask: Optional attention mask
            iteration: Current refinement iteration (for adaptive mode)
            total_iterations: Total iterations (for adaptive mode)
        """
        if self.mode == "adaptive":
            # Early iterations: Mamba (fast, broad context)
            # Late iterations: Attention (precise crystallization)
            use_attention = iteration >= int(total_iterations * (1 - self.mamba_ratio))

            layers = self.attn_layers if use_attention else self.mamba_layers
            for layer in layers:
                z = layer(z, mask)
        else:
            for layer in self.layers:
                z = layer(z, mask)

        return self.norm(z)


def test_mamba():
    """Quick test to verify implementation works."""
    print("Testing pure PyTorch Mamba implementation...")

    B, L, D = 2, 64, 256
    x = torch.randn(B, L, D)

    # Test MambaBlock
    config = MambaConfig(hidden_size=D)
    mamba = MambaBlock(config)
    y = mamba(x)
    print(f"MambaBlock: {x.shape} → {y.shape} ✓")

    # Test MambaLayer
    layer = MambaLayer(D)
    y = layer(x)
    print(f"MambaLayer: {x.shape} → {y.shape} ✓")

    # Test HybridRefineBlock modes
    for mode in ["attention", "mamba", "parallel", "interleaved", "adaptive"]:
        block = HybridRefineBlock(D, num_heads=8, num_layers=4, mode=mode)
        y = block(x, iteration=0, total_iterations=4)
        params = sum(p.numel() for p in block.parameters())
        print(f"HybridRefineBlock({mode}): {x.shape} → {y.shape}, {params/1e6:.2f}M params ✓")

    print("\nAll tests passed! 🎉")


if __name__ == "__main__":
    test_mamba()
