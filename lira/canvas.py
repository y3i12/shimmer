"""
Latent Canvas Model for iterative refinement.

Core idea:
- Latent canvas (one embedding per position) gets refined iteratively
- Same RefineBlock applied K times (TRM-style parameter reuse)
- Confidence measurement enables soft crystallization
- Final decode to tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

from .layers import RMSNorm, RefineBlock


@dataclass
class LatentCanvasConfig:
    """Configuration for Latent Canvas Model."""

    # Model dimensions
    vocab_size: int = 50257  # GPT-2 default
    hidden_size: int = 256
    num_heads: int = 8
    num_layers: int = 2  # Layers per RefineBlock
    max_seq_len: int = 128

    # Refinement settings
    num_refine_steps: int = 1  # K: how many times to apply RefineBlock

    # Special tokens
    mask_token_id: int = field(default=-1)  # Set to vocab_size if -1
    pad_token_id: int = 0

    # Training
    dropout: float = 0.0

    def __post_init__(self):
        if self.mask_token_id == -1:
            self.mask_token_id = self.vocab_size


class LatentCanvasModel(nn.Module):
    """
    Latent Canvas Model with iterative refinement.

    Architecture:
        1. Embed tokens → latent canvas
        2. Apply RefineBlock K times (same weights)
        3. Measure confidence at each position
        4. Decode to token logits

    The key insight: refinement happens in latent space,
    and the same network is applied repeatedly.
    """

    def __init__(self, config: LatentCanvasConfig):
        super().__init__()
        self.config = config

        # Token embedding (includes mask token)
        self.embed_tokens = nn.Embedding(
            config.vocab_size + 1,  # +1 for mask token
            config.hidden_size
        )

        # Learnable "noise" embedding for masked positions
        # This gives the model a starting point for refinement
        self.mask_embed = nn.Parameter(torch.randn(config.hidden_size) * 0.02)

        # The refinement block (applied K times)
        self.refine_block = RefineBlock(
            config.hidden_size,
            config.num_heads,
            config.num_layers,
            config.max_seq_len
        )

        # Output heads
        self.token_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.confidence_head = nn.Linear(config.hidden_size, 1, bias=False)

        # Tie embeddings with output (optional but common)
        # self.token_head.weight = self.embed_tokens.weight[:-1]  # Exclude mask token

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_latent_canvas(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert input tokens to latent canvas.
        Masked positions get the learnable mask embedding.
        """
        # Get base embeddings
        z = self.embed_tokens(input_ids)

        # Replace masked positions with learnable mask embedding
        mask_positions = (input_ids == self.config.mask_token_id)
        z[mask_positions] = self.mask_embed

        return z

    def refine(
        self,
        z: torch.Tensor,
        num_steps: Optional[int] = None,
        return_intermediates: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Apply refinement steps to latent canvas.

        Args:
            z: Latent canvas [B, L, D]
            num_steps: Override config.num_refine_steps
            return_intermediates: Return all intermediate states

        Returns:
            Refined latent canvas (and optionally intermediates)
        """
        num_steps = num_steps or self.config.num_refine_steps
        intermediates = []

        for step in range(num_steps):
            z = self.refine_block(z)
            if return_intermediates:
                intermediates.append(z.clone())

        if return_intermediates:
            return z, intermediates
        return z

    def get_confidence(self, z: torch.Tensor) -> torch.Tensor:
        """
        Measure confidence at each position.

        Returns:
            Confidence scores [B, L] in range [0, 1]
        """
        return torch.sigmoid(self.confidence_head(z)).squeeze(-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent canvas to token logits.

        Returns:
            Token logits [B, L, vocab_size]
        """
        return self.token_head(z)

    def forward(
        self,
        input_ids: torch.Tensor,
        num_refine_steps: Optional[int] = None,
        return_confidence: bool = True,
        return_intermediates: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: embed → refine → decode.

        Args:
            input_ids: Input token IDs [B, L] (with masks)
            num_refine_steps: Override default refinement steps
            return_confidence: Also return confidence scores
            return_intermediates: Return intermediate latent states

        Returns:
            Dictionary with:
                - logits: Token logits [B, L, vocab_size]
                - confidence: Confidence scores [B, L] (if requested)
                - intermediates: List of latent states (if requested)
        """
        # Embed to latent canvas
        z = self.get_latent_canvas(input_ids)

        # Refine
        if return_intermediates:
            z, intermediates = self.refine(z, num_refine_steps, return_intermediates=True)
        else:
            z = self.refine(z, num_refine_steps)

        # Decode
        logits = self.decode(z)

        result = {"logits": logits, "latent": z}

        if return_confidence:
            conf_logits = self.confidence_head(z).squeeze(-1)
            result["confidence"] = torch.sigmoid(conf_logits)
            result["conf_logits"] = conf_logits  # Raw logits for training

        if return_intermediates:
            result["intermediates"] = intermediates

        return result

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        gen_length: int,
        num_iterations: int = 10,
        temperature: float = 1.0,
        confidence_threshold: float = 0.9,
        remask_threshold: float = 0.3,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate tokens via iterative refinement.

        This is the core "diffusion-like" generation:
        1. Start with prompt + masks
        2. Refine and decode
        3. Commit high-confidence predictions
        4. Optionally re-mask low-confidence positions
        5. Repeat until convergence

        Args:
            prompt_ids: Prompt token IDs [B, prompt_len]
            gen_length: Number of tokens to generate
            num_iterations: Max refinement iterations
            temperature: Sampling temperature
            confidence_threshold: Commit predictions above this
            remask_threshold: Re-mask predictions below this

        Returns:
            Generated token IDs [B, prompt_len + gen_length]
            List of canvas states for visualization
        """
        B = prompt_ids.size(0)
        device = prompt_ids.device

        # Initialize canvas: prompt + masks
        mask_token = self.config.mask_token_id
        canvas = torch.cat([
            prompt_ids,
            torch.full((B, gen_length), mask_token, device=device, dtype=torch.long)
        ], dim=1)

        # Track which positions are "generation" (not prompt)
        gen_mask = torch.zeros_like(canvas, dtype=torch.bool)
        gen_mask[:, prompt_ids.size(1):] = True

        history = [canvas.clone()]

        for iteration in range(num_iterations):
            # Forward pass
            output = self.forward(canvas, return_confidence=True)
            logits = output["logits"]
            confidence = output["confidence"]

            # Sample predictions
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                predictions = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, -1)
            else:
                predictions = logits.argmax(dim=-1)

            # Get confidence of predictions (softmax probability of chosen token)
            pred_probs = F.softmax(logits, dim=-1)
            pred_confidence = pred_probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)

            # Combine learned confidence with prediction confidence
            combined_confidence = (confidence + pred_confidence) / 2

            # Only consider generation positions that are currently masked
            is_masked = (canvas == mask_token) & gen_mask

            # Commit high-confidence predictions
            commit_mask = is_masked & (combined_confidence > confidence_threshold)
            canvas[commit_mask] = predictions[commit_mask]

            # Re-mask low-confidence positions (that were previously filled)
            was_filled = (canvas != mask_token) & gen_mask
            remask_mask = was_filled & (combined_confidence < remask_threshold)
            canvas[remask_mask] = mask_token

            history.append(canvas.clone())

            # Check for convergence (no masks remaining in generation region)
            if not (canvas[gen_mask.view(B, -1)] == mask_token).any():
                break

        return canvas, history

    @torch.no_grad()
    def generate_topk(
        self,
        prompt_ids: torch.Tensor,
        gen_length: int,
        num_steps: int = 10,
        temperature: float = 0.8,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate using LLaDA-style top-k selection.

        Instead of threshold-based, commit a fixed number of tokens
        per step based on confidence ranking.
        """
        B = prompt_ids.size(0)
        device = prompt_ids.device
        mask_token = self.config.mask_token_id

        # Initialize canvas: prompt + masks
        canvas = torch.cat([
            prompt_ids,
            torch.full((B, gen_length), mask_token, device=device, dtype=torch.long)
        ], dim=1)

        prompt_len = prompt_ids.size(1)
        history = [canvas.clone()]

        # How many tokens to commit per step (distribute evenly)
        tokens_per_step = max(1, gen_length // num_steps)

        for step in range(num_steps):
            # Forward pass
            output = self.forward(canvas)
            logits = output["logits"]

            # Sample predictions
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                predictions = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, -1)
            else:
                predictions = logits.argmax(dim=-1)

            # Get confidence (probability of predicted token)
            pred_probs = F.softmax(logits, dim=-1)
            confidence = pred_probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)

            # Only consider masked positions in generation region
            is_masked = canvas == mask_token
            is_masked[:, :prompt_len] = False  # Don't touch prompt

            # Set confidence to -inf for non-masked positions
            confidence = torch.where(is_masked, confidence, torch.tensor(-float('inf'), device=device))

            # Select top-k most confident masked positions
            num_masked = is_masked.sum(dim=1)
            k = min(tokens_per_step, num_masked.min().item())

            if k > 0:
                for b in range(B):
                    # Get indices of top-k confident positions for this batch
                    batch_conf = confidence[b]
                    _, top_indices = batch_conf.topk(k)
                    canvas[b, top_indices] = predictions[b, top_indices]

            history.append(canvas.clone())

            # Check for convergence
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
        remask_ratio: float = 0.3,
        min_confident_ratio: float = 0.1,
        repetition_penalty: float = 1.2,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate with confidence-based re-masking.

        The key innovation: tokens can be UN-committed if confidence drops.
        This creates diffusion-like iterative refinement where the canvas
        evolves bidirectionally.

        Args:
            prompt_ids: Prompt token IDs [B, prompt_len]
            gen_length: Number of tokens to generate
            num_steps: Max refinement iterations
            temperature: Sampling temperature
            remask_ratio: Fraction of lowest-confidence filled tokens to re-mask each step
            min_confident_ratio: Min fraction to keep per step (prevents all re-masking)
        """
        B = prompt_ids.size(0)
        device = prompt_ids.device
        mask_token = self.config.mask_token_id

        # Initialize canvas: prompt + masks
        canvas = torch.cat([
            prompt_ids,
            torch.full((B, gen_length), mask_token, device=device, dtype=torch.long)
        ], dim=1)

        prompt_len = prompt_ids.size(1)
        history = [canvas.clone()]

        # Track how many tokens to commit/remask per step
        tokens_per_step = max(1, gen_length // (num_steps // 2))

        for step in range(num_steps):
            # Forward pass with confidence
            output = self.forward(canvas, return_confidence=True)
            logits = output["logits"]
            learned_confidence = output["confidence"]  # From trained confidence head

            # Apply repetition penalty to tokens already in canvas
            if repetition_penalty != 1.0:
                for b in range(B):
                    existing_tokens = canvas[b][canvas[b] != mask_token].unique()
                    for token in existing_tokens:
                        logits[b, :, token] /= repetition_penalty

            # Sample predictions
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                predictions = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, -1)
            else:
                predictions = logits.argmax(dim=-1)

            # Get prediction confidence (softmax probability)
            pred_probs = F.softmax(logits, dim=-1)
            pred_confidence = pred_probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)

            # Combine learned confidence with prediction confidence
            combined_confidence = (learned_confidence + pred_confidence) / 2

            # === PHASE 1: Fill some masked positions ===
            is_masked = (canvas == mask_token)
            is_masked[:, :prompt_len] = False  # Protect prompt

            # For masked positions, use combined confidence
            fill_confidence = torch.where(
                is_masked,
                combined_confidence,
                torch.tensor(-float('inf'), device=device)
            )

            # Fill top-k most confident masked positions
            num_masked = is_masked.sum(dim=1)
            k_fill = min(tokens_per_step, num_masked.min().item())

            if k_fill > 0:
                for b in range(B):
                    _, top_indices = fill_confidence[b].topk(k_fill)
                    canvas[b, top_indices] = predictions[b, top_indices]

            # === PHASE 2: Re-mask low-confidence filled positions ===
            is_filled = (canvas != mask_token)
            is_filled[:, :prompt_len] = False  # Protect prompt

            num_filled = is_filled.sum(dim=1)

            if num_filled.min().item() > 0:
                # Calculate how many to potentially re-mask
                k_remask = int(num_filled.min().item() * remask_ratio)
                k_keep = int(num_filled.min().item() * min_confident_ratio)
                k_remask = min(k_remask, num_filled.min().item() - k_keep)

                if k_remask > 0:
                    for b in range(B):
                        # Get confidence of filled positions
                        filled_conf = torch.where(
                            is_filled[b],
                            combined_confidence[b],
                            torch.tensor(float('inf'), device=device)
                        )
                        # Find lowest confidence filled positions
                        _, low_indices = filled_conf.topk(k_remask, largest=False)
                        # Re-mask them
                        canvas[b, low_indices] = mask_token

            history.append(canvas.clone())

            # Check for convergence (all filled and stable)
            still_masked = (canvas[:, prompt_len:] == mask_token).any()
            if not still_masked and step > num_steps // 2:
                # Run a few more refinement steps even when filled
                pass  # Continue refining

            # Early stop if fully filled and past minimum steps
            if not still_masked and step > num_steps * 0.8:
                break

        # Final pass: fill any remaining masks without re-masking
        is_masked = (canvas == mask_token)
        if is_masked.any():
            output = self.forward(canvas)
            predictions = output["logits"].argmax(dim=-1)
            canvas[is_masked] = predictions[is_masked]
            history.append(canvas.clone())

        return canvas, history


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
