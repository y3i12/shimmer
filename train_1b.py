"""
Training script for ~1B parameter Shimmer model on 8GB VRAM.

Memory optimizations:
- 8-bit Adam (bitsandbytes)
- Activation/gradient checkpointing
- Gradient accumulation
- FP16 mixed precision
- Efficient attention (torch.nn.functional.scaled_dot_product_attention)

Target: ~1B params with D=2048, 16 layers
"""

import os
import sys
import argparse
import time
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.utils.checkpoint import checkpoint

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from lira.layers import RMSNorm, RotaryEmbedding, apply_rotary_pos_emb, rotate_half


@dataclass
class Shimmer1BConfig:
    """Configuration for ~1B parameter model."""
    # Model
    vocab_size: int = 50257
    hidden_size: int = 2048
    intermediate_size: int = 5504  # 2.7 * hidden
    num_heads: int = 16
    num_layers: int = 16
    max_seq_len: int = 512

    # Training
    batch_size: int = 2  # Micro batch
    gradient_accumulation_steps: int = 16  # Effective batch = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 50000

    # Memory optimizations
    use_gradient_checkpointing: bool = True
    use_8bit_adam: bool = True

    # Refinement (LIRA)
    num_refine_steps: int = 2

    # Special tokens
    mask_token_id: int = -1  # Set to vocab_size
    pad_token_id: int = 50256

    def __post_init__(self):
        if self.mask_token_id == -1:
            self.mask_token_id = self.vocab_size


class EfficientAttention(nn.Module):
    """Memory-efficient attention using PyTorch's SDPA."""

    def __init__(self, config: Shimmer1BConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = self.rotary(x, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Efficient attention (uses Flash Attention if available)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        return self.o_proj(out)


class EfficientSwiGLU(nn.Module):
    """SwiGLU FFN."""

    def __init__(self, config: Shimmer1BConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, config: Shimmer1BConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = EfficientAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.ffn = EfficientSwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Shimmer1B(nn.Module):
    """
    ~1B parameter Shimmer model with memory optimizations.
    """

    def __init__(self, config: Shimmer1BConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size + 1, config.hidden_size)
        self.mask_embed = nn.Parameter(torch.randn(config.hidden_size) * 0.02)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_size)

        # Output heads
        self.token_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.confidence_head = nn.Linear(config.hidden_size, 1, bias=False)

        # Gradient checkpointing
        self.gradient_checkpointing = config.use_gradient_checkpointing

        self._init_weights()

        # Count params
        self.num_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {self.num_params / 1e9:.2f}B")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        num_refine_steps: int = None
    ) -> dict:
        """Forward pass with optional gradient checkpointing."""
        num_refine_steps = num_refine_steps or self.config.num_refine_steps

        # Embed
        z = self.embed_tokens(input_ids)
        mask_positions = (input_ids == self.config.mask_token_id)
        z[mask_positions] = self.mask_embed

        # Refine K times
        for _ in range(num_refine_steps):
            # Apply transformer layers with optional checkpointing
            for layer in self.layers:
                if self.gradient_checkpointing and self.training:
                    z = checkpoint(layer, z, use_reentrant=False)
                else:
                    z = layer(z)

        z = self.norm(z)

        # Decode
        logits = self.token_head(z)
        conf_logits = self.confidence_head(z).squeeze(-1)

        return {
            "logits": logits,
            "confidence": torch.sigmoid(conf_logits),
            "conf_logits": conf_logits,
        }


def create_optimizer(model: nn.Module, config: Shimmer1BConfig):
    """Create optimizer with optional 8-bit Adam."""

    # Separate weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'norm' in name or 'bias' in name or 'embed' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if config.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
        )
        print("Using 8-bit Adam optimizer")
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
        )
        print("Using standard AdamW optimizer")

    return optimizer


def get_lr(step: int, config: Shimmer1BConfig) -> float:
    """Learning rate schedule with warmup and cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    # Cosine decay
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def load_data(config: Shimmer1BConfig, num_samples: int = 10000):
    """Load TinyStories dataset."""
    from data.dataset import load_tinystories

    print(f"Loading TinyStories ({num_samples} samples)...")

    train_data = load_tinystories(
        split="train",
        num_samples=num_samples,
        max_length=config.max_seq_len,
        mask_ratio=0.3,
    )

    val_data = load_tinystories(
        split="validation",
        num_samples=min(1000, num_samples // 10),
        max_length=config.max_seq_len,
        mask_ratio=0.3,
    )

    # Update vocab size and mask token
    config.vocab_size = train_data.vocab_size
    config.mask_token_id = train_data.mask_token_id

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader


def estimate_memory():
    """Estimate and print memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def train(config: Shimmer1BConfig, num_samples: int = 10000):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load data
    train_loader, val_loader = load_data(config, num_samples)

    # Create model
    print(f"\nCreating Shimmer 1B model...")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Vocab: {config.vocab_size}")
    print(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
    print(f"  8-bit Adam: {config.use_8bit_adam}")

    model = Shimmer1B(config).to(device)
    estimate_memory()

    # Optimizer
    optimizer = create_optimizer(model, config)
    scaler = GradScaler()

    # Training
    print(f"\nStarting training...")
    print(f"  Micro batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Max steps: {config.max_steps}")
    print()

    model.train()
    step = 0
    accumulation_step = 0
    running_loss = 0.0
    best_val_loss = float('inf')

    os.makedirs("checkpoints", exist_ok=True)
    start_time = time.time()

    while step < config.max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask_ratio = batch["mask_ratio"].to(device)

            # Forward with mixed precision
            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(input_ids)
                logits = output["logits"]

                # Loss with mask ratio weighting (LLaDA style)
                mask_positions = (input_ids == config.mask_token_id)

                loss = F.cross_entropy(
                    logits[mask_positions],
                    labels[mask_positions],
                    reduction='none'
                )

                # Weight by inverse mask ratio
                weights = 1.0 / mask_ratio.unsqueeze(1).expand_as(input_ids)[mask_positions]
                loss = (loss * weights).mean()

                # Scale for accumulation
                loss = loss / config.gradient_accumulation_steps

            # Backward
            scaler.scale(loss).backward()
            running_loss += loss.item()
            accumulation_step += 1

            # Update weights
            if accumulation_step >= config.gradient_accumulation_steps:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update LR
                lr = get_lr(step, config)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                step += 1
                accumulation_step = 0

                # Logging
                if step % 10 == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = (step * config.batch_size * config.gradient_accumulation_steps * config.max_seq_len) / elapsed
                    print(f"Step {step:>6} | Loss: {running_loss:.4f} | LR: {lr:.2e} | {tokens_per_sec:.0f} tok/s")
                    estimate_memory()
                    running_loss = 0.0

                # Validation
                if step % 500 == 0:
                    val_loss = validate(model, val_loader, config, device)
                    print(f"  Val Loss: {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            'step': step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'config': config,
                        }, "checkpoints/shimmer_1b_best.pt")
                        print(f"  Saved best checkpoint!")

                    model.train()

                if step >= config.max_steps:
                    break

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    return model


@torch.no_grad()
def validate(model, val_loader, config, device):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            output = model(input_ids)
            logits = output["logits"]

            mask_positions = (input_ids == config.mask_token_id)
            loss = F.cross_entropy(logits[mask_positions], labels[mask_positions])

        total_loss += loss.item()
        num_batches += 1

        if num_batches >= 50:  # Limit validation batches
            break

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train Shimmer 1B")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--no_8bit_adam", action="store_true")
    parser.add_argument("--no_checkpointing", action="store_true")
    args = parser.parse_args()

    config = Shimmer1BConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_steps=args.max_steps,
        use_8bit_adam=not args.no_8bit_adam,
        use_gradient_checkpointing=not args.no_checkpointing,
    )

    train(config, args.num_samples)


if __name__ == "__main__":
    main()
