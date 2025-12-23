"""
Dialectic: Bidirectional Layer Negotiation for Shimmer/LIRA

Layers can:
  - Accept input and proceed (forward)
  - Reject input and send correction back (backtrack)
  - Negotiate until synthesis is reached

Thesis     → Layer N's output
Antithesis → Layer N+1's rejection
Synthesis  → Refined output after negotiation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass

from .layers import RMSNorm, RotaryEmbedding, apply_rotary_pos_emb


@dataclass
class DialecticConfig:
    """Configuration for Dialectic layers."""
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    max_backtracks: int = 3          # Max times any layer can request backtrack
    confidence_threshold: float = 0.7 # Above this = accept, below = negotiate
    dropout: float = 0.0

    # Vocabulary
    vocab_size: int = 10000
    max_seq_len: int = 512

    # Special tokens
    pad_token_id: int = 0
    mask_token_id: int = None  # Set to vocab_size if None

    def __post_init__(self):
        if self.mask_token_id is None:
            self.mask_token_id = self.vocab_size


class NegotiationResult(NamedTuple):
    """Result of layer negotiation."""
    output: torch.Tensor           # The (possibly refined) output
    confidence: torch.Tensor       # Mean confidence for backtrack decision [B, 1]
    correction: torch.Tensor       # Suggested correction for previous layer
    accepted: torch.Tensor         # Whether input was accepted (bool per sample)
    position_confidence: torch.Tensor = None  # Per-position confidence [B, L, 1] for supervision
    position_conf_logits: torch.Tensor = None  # Logits before sigmoid [B, L, 1] for autocast-safe training


class DialecticAttention(nn.Module):
    """
    Attention with confidence estimation.

    Outputs both the attended values AND a confidence score
    indicating how "coherent" the attention pattern is.
    """

    def __init__(self, config: DialecticConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Confidence head: estimates how "good" this attention is
        self.confidence_head = nn.Linear(config.hidden_size, 1, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: Attended values [B, L, D]
            confidence: Per-position confidence [B, L, 1]
        """
        B, L, D = x.shape

        # Project and transpose to [B, H, L, D]
        q = self.q_proj(x).view(B, L, self.config.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.config.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.config.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary(x, seq_len=L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)

        # Estimate confidence from output (return both logits and sigmoid for different uses)
        conf_logits = self.confidence_head(out)  # [B, L, 1] - for autocast-safe training
        confidence = torch.sigmoid(conf_logits)   # [B, L, 1] - for inference/decisions

        return out, confidence, conf_logits


class DialecticFFN(nn.Module):
    """Feed-forward with correction output."""

    def __init__(self, config: DialecticConfig):
        super().__init__()
        hidden_dim = int(config.hidden_size * 2.667)

        self.w1 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, hidden_dim, bias=False)

        # Correction head: suggests what should change
        self.correction_head = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: FFN output [B, L, D]
            correction: Suggested correction vector [B, L, D]
        """
        # SwiGLU
        out = self.w2(F.silu(self.w1(x)) * self.w3(x))

        # Correction is a residual suggestion
        correction = self.correction_head(out)

        return out, correction


class DialecticLayer(nn.Module):
    """
    A single Dialectic layer that can:
    1. Process input normally (accept)
    2. Signal rejection and provide correction (negotiate)

    The layer outputs confidence - if low, the previous layer
    should incorporate the correction and try again.
    """

    def __init__(self, config: DialecticConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = DialecticAttention(config)

        self.norm2 = RMSNorm(config.hidden_size)
        self.ffn = DialecticFFN(config)

        # Gate for incorporating feedback from next layer
        self.feedback_gate = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        feedback: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> NegotiationResult:
        """
        Args:
            x: Input tensor [B, L, D]
            feedback: Correction from next layer (if backtracking) [B, L, D]
            mask: Attention mask

        Returns:
            NegotiationResult with output, confidence, correction, accepted
        """
        # Incorporate feedback if provided (backtracking scenario)
        if feedback is not None:
            # Gate controls how much feedback to incorporate
            gate_input = torch.cat([x, feedback], dim=-1)
            gate = torch.sigmoid(self.feedback_gate(gate_input))
            x = x + gate * feedback

        # Self-attention with confidence
        attn_out, attn_conf, attn_conf_logits = self.attn(self.norm1(x), mask)
        x = x + attn_out

        # FFN with correction
        ffn_out, correction = self.ffn(self.norm2(x))
        x = x + ffn_out

        # Per-position confidence [B, L, 1] and mean confidence [B, 1]
        position_confidence = attn_conf  # [B, L, 1]
        position_conf_logits = attn_conf_logits  # [B, L, 1] - for autocast-safe training
        mean_confidence = attn_conf.mean(dim=1)  # [B, 1]

        # Accept if mean confidence above threshold
        accepted = mean_confidence.squeeze(-1) > self.config.confidence_threshold

        return NegotiationResult(
            output=x,
            confidence=mean_confidence,  # For backtracking decision
            correction=correction,
            accepted=accepted,
            position_confidence=position_confidence,
            position_conf_logits=position_conf_logits,  # For autocast-safe training
        )


class DialecticBlock(nn.Module):
    """
    A block of Dialectic layers with negotiation.

    Layers negotiate in sequence:
    - Layer N processes, outputs confidence
    - If confidence low, Layer N-1 gets correction and retries
    - Process repeats until all layers accept or max_backtracks reached
    """

    def __init__(self, config: DialecticConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            DialecticLayer(config, i) for i in range(config.num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward with negotiation.

        Uses soft gating for batching compatibility:
        - All samples go through same number of steps
        - Soft gates control actual influence

        Returns:
            output: Final negotiated output [B, L, D]
            stats: Dictionary with negotiation statistics
        """
        B, L, D = x.shape
        device = x.device

        stats = {
            'backtracks_per_layer': torch.zeros(self.config.num_layers, device=device),
            'final_confidences': [],
            'total_backtracks': 0,
            # NEW: Track per-layer confidence tensors for supervision
            'layer_confidences': [],  # List of [B, L] tensors
        }

        # Process through layers with potential backtracking
        layer_outputs = [None] * self.config.num_layers
        layer_feedbacks = [None] * self.config.num_layers

        current_input = x

        for backtrack_round in range(self.config.max_backtracks + 1):
            # Forward pass through all layers
            for i, layer in enumerate(self.layers):
                # Get feedback if this isn't the first round
                feedback = layer_feedbacks[i]

                # Process layer
                result = layer(current_input, feedback=feedback, mask=mask)
                layer_outputs[i] = result

                # Soft gate based on confidence (for gradient flow)
                # High confidence = use output as-is
                # Low confidence = will be refined in next round
                conf = result.confidence  # [B, 1]

                # Prepare input for next layer
                current_input = result.output

                # Check if this layer wants to send feedback to previous
                if i > 0 and backtrack_round < self.config.max_backtracks:
                    # Low confidence = send correction back
                    # Expand conf to [B, 1, 1] to broadcast with [B, L, D]
                    send_feedback = (1 - conf).unsqueeze(-1)  # [B, 1, 1]
                    layer_feedbacks[i-1] = send_feedback * result.correction

                    if (conf < self.config.confidence_threshold).any():
                        stats['backtracks_per_layer'][i-1] += 1
                        stats['total_backtracks'] += 1

            # Check if all layers are confident (early exit)
            all_confident = all(
                (r.confidence > self.config.confidence_threshold).all()
                for r in layer_outputs if r is not None
            )
            if all_confident:
                break

            # Reset for next round (start from beginning with feedbacks)
            current_input = x

        # Final output from last layer
        final_output = layer_outputs[-1].output
        stats['final_confidences'] = [r.confidence.mean().item() for r in layer_outputs]

        # NEW: Collect per-position confidence LOGITS for autocast-safe supervision
        # Each layer outputs position_conf_logits [B, L, 1]
        stats['layer_conf_logits'] = [
            r.position_conf_logits.squeeze(-1) if r.position_conf_logits is not None else None
            for r in layer_outputs
        ]  # List of [B, L] tensors (logits, not sigmoid'd)

        return final_output, stats


class DialecticCanvas(nn.Module):
    """
    Full Shimmer model with Dialectic layers.

    Combines:
    - LIRA's iterative refinement (K passes)
    - Dialectic's bidirectional negotiation (within each pass)
    """

    def __init__(self, config: DialecticConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size + 1, config.hidden_size)
        self.embed_scale = config.hidden_size ** 0.5

        # Dialectic block (replaces RefineBlock)
        self.dialectic = DialecticBlock(config)

        # Output heads
        self.norm = RMSNorm(config.hidden_size)
        self.token_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.confidence_head = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_refine_steps: int = 1
    ) -> dict:
        """
        Forward pass with LIRA refinement + Dialectic negotiation.

        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Padding mask [B, L]
            num_refine_steps: K iterations of the full dialectic block

        Returns:
            dict with token_logits, confidence, negotiation_stats
        """
        # Embed
        x = self.embed_tokens(input_ids) * self.embed_scale

        # Convert attention mask for attention
        if attention_mask is not None:
            attn_mask = attention_mask == 0  # True where padded
        else:
            attn_mask = None

        # LIRA-style refinement: apply dialectic block K times
        all_stats = []
        all_layer_conf_logits = []  # Collect logits for autocast-safe training
        for k in range(num_refine_steps):
            x, stats = self.dialectic(x, mask=attn_mask)
            all_stats.append(stats)
            # Collect layer confidence LOGITS for supervision
            if stats.get('layer_conf_logits'):
                all_layer_conf_logits.extend(stats['layer_conf_logits'])

        # Output
        x = self.norm(x)
        token_logits = self.token_head(x)
        confidence = torch.sigmoid(self.confidence_head(x))

        return {
            'token_logits': token_logits,
            'confidence': confidence,
            'negotiation_stats': all_stats,
            'layer_conf_logits': all_layer_conf_logits,  # Logits for autocast-safe training
        }

    def generate_dialectic(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        num_refine_steps: int = 2,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        Generate with Dialectic negotiation.

        Uses iterative unmasking like LIRA, but each step
        involves full dialectic negotiation between layers.
        """
        device = prompt_ids.device
        B = prompt_ids.shape[0]

        # Initialize canvas: prompt + masks
        canvas = torch.cat([
            prompt_ids,
            torch.full((B, max_new_tokens), self.config.mask_token_id, device=device)
        ], dim=1)

        mask_token_id = self.config.mask_token_id

        # Iterative unmasking with dialectic
        for iteration in range(max_new_tokens):
            # Find mask positions
            mask_positions = canvas == mask_token_id
            if not mask_positions.any():
                break

            # Forward with negotiation
            result = self.forward(canvas, num_refine_steps=num_refine_steps)
            logits = result['token_logits']
            confidence = result['confidence']

            # Sample from logits
            logits = logits / temperature
            if top_k > 0:
                topk_logits, topk_indices = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(topk_logits, dim=-1)
                sampled_indices = torch.multinomial(probs.view(-1, top_k), 1).view(B, -1)
                predictions = torch.gather(topk_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)
            else:
                predictions = torch.argmax(logits, dim=-1)

            # Fill highest confidence mask position
            mask_confidence = confidence.squeeze(-1).clone()
            mask_confidence[~mask_positions] = -float('inf')

            best_positions = mask_confidence.argmax(dim=-1)

            for b in range(B):
                pos = best_positions[b].item()
                if mask_positions[b, pos]:
                    canvas[b, pos] = predictions[b, pos]

        return canvas


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick test
if __name__ == "__main__":
    config = DialecticConfig(
        hidden_size=256,
        num_heads=8,
        num_layers=4,
        max_backtracks=2,
        vocab_size=10000
    )

    model = DialecticCanvas(config)
    print(f"Dialectic model: {count_parameters(model):,} parameters")

    # Test forward
    x = torch.randint(0, 10000, (2, 32))
    result = model(x, num_refine_steps=2)

    print(f"Token logits: {result['token_logits'].shape}")
    print(f"Confidence: {result['confidence'].shape}")
    print(f"Negotiation rounds: {len(result['negotiation_stats'])}")

    for i, stats in enumerate(result['negotiation_stats']):
        print(f"  Round {i}: backtracks={stats['total_backtracks']}, "
              f"confidences={[f'{c:.2f}' for c in stats['final_confidences']]}")
