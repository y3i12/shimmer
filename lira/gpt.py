"""
GPT: Autoregressive Transformer Baseline

A standard GPT-style decoder-only transformer for comparison with LIRA.
Uses the same building blocks (RMSNorm, RoPE, SwiGLU) but with causal attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

from .layers import RMSNorm, RotaryEmbedding, SwiGLU, rotate_half, apply_rotary_pos_emb


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    vocab_size: int = 10000
    hidden_size: int = 256
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = 512
    dropout: float = 0.0

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"


class CausalAttention(nn.Module):
    """Multi-head attention with causal mask and RoPE."""

    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

        # Pre-compute causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len),
            persistent=False
        )

    def forward(self, x: torch.Tensor, use_cache: bool = False, past_kv: tuple = None) -> tuple:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary(x, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache for generation
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        # Scaled dot-product attention with causal mask
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        seq_len = k.size(2)
        if seq_len <= self.causal_mask.size(-1):
            mask = self.causal_mask[:, :, :L, :seq_len]
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_size)

        return self.o_proj(out), new_kv


class GPTBlock(nn.Module):
    """Pre-norm transformer block with causal attention."""

    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = CausalAttention(hidden_size, num_heads, max_seq_len, dropout)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size)

    def forward(self, x: torch.Tensor, use_cache: bool = False, past_kv: tuple = None) -> tuple:
        # Attention with residual
        attn_out, new_kv = self.attn(self.attn_norm(x), use_cache, past_kv)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_kv


class GPTModel(nn.Module):
    """
    GPT-style autoregressive language model.

    For fair comparison with LIRA:
    - Same parameter count (given same hidden_size, num_layers, num_heads)
    - Same building blocks (RMSNorm, RoPE, SwiGLU)
    - Only difference: causal vs bidirectional attention
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = math.sqrt(config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList([
            GPTBlock(config.hidden_size, config.num_heads, config.max_seq_len, config.dropout)
            for _ in range(config.num_layers)
        ])

        # Output
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
        use_cache: bool = False,
        past_kv: list = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: [B, L] token IDs
            labels: [B, L] target token IDs (for loss computation)
            use_cache: Whether to return KV cache for generation
            past_kv: Previous KV cache

        Returns:
            dict with 'logits', optionally 'loss' and 'past_kv'
        """
        B, L = input_ids.shape

        # Embed tokens
        x = self.embed(input_ids) * self.embed_scale

        # Process through layers
        new_kv = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past = past_kv[i] if past_kv is not None else None
            x, kv = layer(x, use_cache, layer_past)
            if use_cache:
                new_kv.append(kv)

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}

        if use_cache:
            result["past_kv"] = new_kv

        # Compute loss if labels provided
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV caching.

        Args:
            prompt_ids: [B, L] prompt token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            use_cache: Use KV cache for faster generation

        Returns:
            [B, L + max_new_tokens] generated token IDs
        """
        self.eval()
        device = prompt_ids.device
        generated = prompt_ids.clone()
        past_kv = None

        for _ in range(max_new_tokens):
            # Get input (full sequence or just last token with cache)
            if use_cache and past_kv is not None:
                input_ids = generated[:, -1:]
            else:
                input_ids = generated

            # Forward pass
            output = self(input_ids, use_cache=use_cache, past_kv=past_kv)
            logits = output["logits"][:, -1, :]  # Last position

            if use_cache:
                past_kv = output["past_kv"]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    min_top_k = top_k_vals[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < min_top_k, float("-inf"), logits)

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative prob above threshold
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float("-inf"))

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if max length reached
            if generated.size(1) >= self.config.max_seq_len:
                break

        return generated


def count_gpt_parameters(model: GPTModel) -> int:
    """Count trainable parameters in GPT model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
