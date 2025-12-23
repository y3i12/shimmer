"""
Hybrid Shimmer: LIRA Core + Sparse Global Attention

Architecture:
    ┌─────────────────────────────────────┐
    │         SHIMMER CORE (LIRA)         │
    │  [RefineBlock] × K iterations       │
    │  - Latent canvas refinement         │
    │  - Confidence prediction            │
    └─────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────┐
    │    SPARSE GLOBAL ATTENTION (2-3)    │
    │  - Full sequence coherence check    │
    │  - Long-range dependency tracking   │
    └─────────────────────────────────────┘

The global attention layers provide coherence checking that the
local RefineBlocks miss. This addresses the remasking collapse issue.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

from .canvas import LatentCanvasModel, LatentCanvasConfig
from .layers import RMSNorm, RotaryEmbedding, apply_rotary_pos_emb


@dataclass
class HybridConfig(LatentCanvasConfig):
    """Configuration for Hybrid Shimmer."""

    # Global attention settings
    num_global_layers: int = 2  # Number of sparse global attention layers
    global_heads: int = 4  # Fewer heads than local (sparse)
    global_frequency_train: int = 4  # Apply global every N refinement steps during training
    global_frequency_gen: int = 1  # Apply global every N steps during generation (more frequent)

    # Coherence settings
    use_coherence_loss: bool = True  # Train with auxiliary coherence loss
    coherence_weight: float = 0.1  # Weight for coherence loss


class GlobalAttentionLayer(nn.Module):
    """
    Sparse global attention layer for coherence checking.

    Uses fewer heads than local attention (sparse) but attends
    to the full sequence for global coherence.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Pre-norm
        self.norm = RMSNorm(hidden_size)

        # QKV projection
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Rotary embeddings for position
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Learnable scale for residual connection
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply global attention.

        Args:
            x: Input tensor [B, L, D]

        Returns:
            Output tensor [B, L, D] with global coherence applied
        """
        B, L, D = x.shape
        residual = x

        # Pre-norm
        x = self.norm(x)

        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary embeddings
        cos, sin = self.rotary(x, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention (full sequence - no causal mask)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Scaled residual connection (start small, learn to increase)
        return residual + self.residual_scale * out


class GlobalCoherenceBlock(nn.Module):
    """
    Full global coherence block: Attention + FFN.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Global attention
        self.attention = GlobalAttentionLayer(
            hidden_size, num_heads, max_seq_len, dropout
        )

        # FFN for processing after attention
        self.norm = RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size, bias=False),
        )
        self.ffn_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global attention
        x = self.attention(x)

        # FFN with scaled residual
        residual = x
        x = self.norm(x)
        x = residual + self.ffn_scale * self.ffn(x)

        return x


class HybridShimmer(nn.Module):
    """
    Hybrid Shimmer: LIRA Core + Sparse Global Attention.

    The core LIRA model handles local refinement, while sparse
    global attention layers provide coherence checking.

    Training: Global attention applied every N refinement steps
    Generation: Global attention applied every step (more vigilant)
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config

        # Core LIRA model (existing Shimmer)
        self.core = LatentCanvasModel(config)

        # Sparse global attention layers
        self.global_layers = nn.ModuleList([
            GlobalCoherenceBlock(
                config.hidden_size,
                config.global_heads,
                config.max_seq_len,
                config.dropout,
            )
            for _ in range(config.num_global_layers)
        ])

        # Optional: Sequence-level coherence head
        if config.use_coherence_loss:
            self.coherence_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(config.hidden_size // 2, 1),
            )
        else:
            self.coherence_head = None

        self._init_global_weights()

    def _init_global_weights(self):
        """Initialize global layer weights."""
        for module in self.global_layers.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def apply_global_coherence(self, z: torch.Tensor) -> torch.Tensor:
        """Apply all global coherence layers."""
        for layer in self.global_layers:
            z = layer(z)
        return z

    def get_sequence_coherence(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute sequence-level coherence score.

        Uses mean pooling over sequence, then predicts coherence.
        Returns score in [0, 1] where 1 = coherent.
        """
        if self.coherence_head is None:
            return None

        # Mean pool over sequence
        z_pooled = z.mean(dim=1)  # [B, D]

        # Predict coherence
        coherence = torch.sigmoid(self.coherence_head(z_pooled))  # [B, 1]
        return coherence.squeeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        num_refine_steps: Optional[int] = None,
        return_confidence: bool = True,
        return_coherence: bool = True,
        training: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with interleaved global coherence.

        During training: Apply global every N refinement steps
        During inference: Apply global every step
        """
        num_steps = num_refine_steps or self.config.num_refine_steps
        global_freq = self.config.global_frequency_train if training else self.config.global_frequency_gen

        # Get initial latent canvas
        z = self.core.get_latent_canvas(input_ids)

        # Interleaved refinement with global coherence
        for step in range(num_steps):
            # Local refinement (single step)
            if self.core.use_hybrid:
                z = self.core.refine_block(z, iteration=step, total_iterations=num_steps)
            else:
                z = self.core.refine_block(z)

            # Global coherence check (every N steps, or every step during generation)
            if (step + 1) % global_freq == 0 or step == num_steps - 1:
                z = self.apply_global_coherence(z)

        # Decode to logits
        logits = self.core.decode(z)

        result = {"logits": logits, "latent": z}

        if return_confidence:
            conf_logits = self.core.confidence_head(z).squeeze(-1)
            result["confidence"] = torch.sigmoid(conf_logits)
            result["conf_logits"] = conf_logits

        if return_coherence and self.coherence_head is not None:
            result["coherence"] = self.get_sequence_coherence(z)

        return result

    @torch.no_grad()
    def generate_topk(
        self,
        prompt_ids: torch.Tensor,
        gen_length: int,
        num_steps: int = 10,
        temperature: float = 0.8,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate with top-k selection + global coherence.

        Global coherence is applied EVERY step during generation
        to prevent drift and collapse.
        """
        B = prompt_ids.size(0)
        device = prompt_ids.device
        mask_token = self.config.mask_token_id

        # Initialize canvas
        canvas = torch.cat([
            prompt_ids,
            torch.full((B, gen_length), mask_token, device=device, dtype=torch.long)
        ], dim=1)

        prompt_len = prompt_ids.size(1)
        history = [canvas.clone()]
        tokens_per_step = max(1, gen_length // num_steps)

        for step in range(num_steps):
            # Get latent canvas
            z = self.core.get_latent_canvas(canvas)

            # Single refinement step
            if self.core.use_hybrid:
                z = self.core.refine_block(z, iteration=0, total_iterations=1)
            else:
                z = self.core.refine_block(z)

            # GLOBAL COHERENCE CHECK (every step during generation!)
            z = self.apply_global_coherence(z)

            # Decode
            logits = self.core.decode(z)

            # Sample predictions
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                predictions = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, -1)
            else:
                predictions = logits.argmax(dim=-1)

            # Get confidence
            pred_probs = F.softmax(logits, dim=-1)
            confidence = pred_probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)

            # Only consider masked positions
            is_masked = canvas == mask_token
            is_masked[:, :prompt_len] = False
            confidence = torch.where(is_masked, confidence, torch.tensor(-float('inf'), device=device))

            # Select top-k
            num_masked = is_masked.sum(dim=1)
            k = min(tokens_per_step, num_masked.min().item())

            if k > 0:
                for b in range(B):
                    _, top_indices = confidence[b].topk(k)
                    canvas[b, top_indices] = predictions[b, top_indices]

            history.append(canvas.clone())

            if not (canvas[:, prompt_len:] == mask_token).any():
                break

        return canvas, history

    @torch.no_grad()
    def generate_with_remasking(
        self,
        prompt_ids: torch.Tensor,
        gen_length: int,
        num_steps: int = 20,
        temperature: float = 0.8,
        remask_ratio: float = 0.2,
        min_confident_ratio: float = 0.2,
        repetition_penalty: float = 1.2,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate with remasking + global coherence.

        The global coherence layers should help prevent the
        collapse that plagued the original remasking approach.
        """
        B = prompt_ids.size(0)
        device = prompt_ids.device
        mask_token = self.config.mask_token_id

        # Initialize canvas
        canvas = torch.cat([
            prompt_ids,
            torch.full((B, gen_length), mask_token, device=device, dtype=torch.long)
        ], dim=1)

        prompt_len = prompt_ids.size(1)
        history = [canvas.clone()]
        tokens_per_step = max(1, gen_length // (num_steps // 2))

        for step in range(num_steps):
            # Get latent canvas
            z = self.core.get_latent_canvas(canvas)

            # Refinement step
            if self.core.use_hybrid:
                z = self.core.refine_block(z, iteration=0, total_iterations=1)
            else:
                z = self.core.refine_block(z)

            # GLOBAL COHERENCE (the key addition!)
            z = self.apply_global_coherence(z)

            # Get predictions and confidence
            logits = self.core.decode(z)
            learned_confidence = self.core.get_confidence(z)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    existing_tokens = canvas[b][canvas[b] != mask_token].unique()
                    for token in existing_tokens:
                        logits[b, :, token] /= repetition_penalty

            # Sample
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                predictions = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, -1)
            else:
                predictions = logits.argmax(dim=-1)

            # Prediction confidence
            pred_probs = F.softmax(logits, dim=-1)
            pred_confidence = pred_probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
            combined_confidence = (learned_confidence + pred_confidence) / 2

            # === PHASE 1: Fill masked positions ===
            is_masked = (canvas == mask_token)
            is_masked[:, :prompt_len] = False

            fill_confidence = torch.where(
                is_masked, combined_confidence,
                torch.tensor(-float('inf'), device=device)
            )

            num_masked = is_masked.sum(dim=1)
            k_fill = min(tokens_per_step, num_masked.min().item())

            if k_fill > 0:
                for b in range(B):
                    _, top_indices = fill_confidence[b].topk(k_fill)
                    canvas[b, top_indices] = predictions[b, top_indices]

            # === PHASE 2: Re-mask low confidence (with global awareness) ===
            is_filled = (canvas != mask_token)
            is_filled[:, :prompt_len] = False
            num_filled = is_filled.sum(dim=1)

            if num_filled.min().item() > 0:
                k_remask = int(num_filled.min().item() * remask_ratio)
                k_keep = int(num_filled.min().item() * min_confident_ratio)
                k_remask = min(k_remask, num_filled.min().item() - k_keep)

                if k_remask > 0:
                    for b in range(B):
                        filled_conf = torch.where(
                            is_filled[b], combined_confidence[b],
                            torch.tensor(float('inf'), device=device)
                        )
                        _, low_indices = filled_conf.topk(k_remask, largest=False)
                        canvas[b, low_indices] = mask_token

            history.append(canvas.clone())

            # Early stop check
            still_masked = (canvas[:, prompt_len:] == mask_token).any()
            if not still_masked and step > num_steps * 0.8:
                break

        # Final fill
        is_masked = (canvas == mask_token)
        if is_masked.any():
            z = self.core.get_latent_canvas(canvas)
            z = self.core.refine_block(z)
            z = self.apply_global_coherence(z)
            predictions = self.core.decode(z).argmax(dim=-1)
            canvas[is_masked] = predictions[is_masked]
            history.append(canvas.clone())

        return canvas, history


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_global_parameters(model: HybridShimmer) -> dict:
    """Count parameters in core vs global layers."""
    core_params = sum(p.numel() for p in model.core.parameters())
    global_params = sum(p.numel() for p in model.global_layers.parameters())
    coherence_params = sum(p.numel() for p in model.coherence_head.parameters()) if model.coherence_head else 0

    return {
        "core": core_params,
        "global": global_params,
        "coherence": coherence_params,
        "total": core_params + global_params + coherence_params,
        "global_ratio": global_params / (core_params + global_params + coherence_params),
    }
