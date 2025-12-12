"""
Shimmer Training Script

Train a LIRA (Latent Iterative Refinement Architecture) model.

Four phases:
1. Single refine pass - baseline masked reconstruction
2. Multiple refine passes - test iterative refinement hypothesis
3. Variable corruption - LLaDA-style (10-100% masking)
4. Confidence supervision - learn to predict uncertainty
"""

import argparse
import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

from lira import LatentCanvasConfig, LatentCanvasModel, count_parameters
from data import load_tinystories, create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train Latent Canvas Model")

    # Phase selection
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Training phase (1-4)")

    # Data
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)

    # Model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)

    # Phase-specific: refinement steps
    parser.add_argument("--num_refine_steps", type=int, default=1,
                        help="Refinement iterations (Phase 2+ uses >1)")

    # Phase-specific: masking
    parser.add_argument("--min_mask_ratio", type=float, default=0.3,
                        help="Min mask ratio (Phase 3 uses 0.1)")
    parser.add_argument("--max_mask_ratio", type=float, default=0.3,
                        help="Max mask ratio (Phase 3 uses 1.0)")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=500)

    return parser.parse_args()


def setup_phase(args):
    """Configure settings based on phase."""
    phase_configs = {
        1: {  # Baseline: single pass, fixed masking
            "num_refine_steps": 1,
            "min_mask_ratio": 0.3,
            "max_mask_ratio": 0.3,
            "description": "Single refine pass, fixed 30% masking"
        },
        2: {  # TRM hypothesis: multiple passes
            "num_refine_steps": 4,  # Try 4 iterations
            "min_mask_ratio": 0.3,
            "max_mask_ratio": 0.3,
            "description": "Multiple refine passes (K=4), fixed 30% masking"
        },
        3: {  # LLaDA-style: variable corruption
            "num_refine_steps": 4,
            "min_mask_ratio": 0.1,
            "max_mask_ratio": 1.0,
            "description": "Multiple passes (K=4), variable corruption (10-100%)"
        },
        4: {  # Full: iterative refinement with re-masking
            "num_refine_steps": 4,
            "min_mask_ratio": 0.1,
            "max_mask_ratio": 1.0,
            "description": "Full iterative refinement with confidence + re-masking"
        }
    }

    config = phase_configs[args.phase]

    # Only override if not explicitly set by user
    if args.num_refine_steps == 1 and args.phase > 1:
        args.num_refine_steps = config["num_refine_steps"]
    if args.min_mask_ratio == 0.3 and args.phase >= 3:
        args.min_mask_ratio = config["min_mask_ratio"]
    if args.max_mask_ratio == 0.3 and args.phase >= 3:
        args.max_mask_ratio = config["max_mask_ratio"]

    print(f"\n=== Phase {args.phase}: {config['description']} ===")
    print(f"  Refine steps: {args.num_refine_steps}")
    print(f"  Mask ratio: {args.min_mask_ratio:.1%} - {args.max_mask_ratio:.1%}")

    return args


def compute_loss(
    model: LatentCanvasModel,
    batch: dict,
    phase: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Compute loss based on phase.

    Phase 1-3: Standard reconstruction loss
    Phase 4: Reconstruction + confidence loss
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    mask_positions = batch["mask_positions"].to(device)
    mask_ratio = batch["mask_ratio"].to(device)

    # Forward pass
    output = model(input_ids, return_confidence=True)
    logits = output["logits"]
    confidence = output["confidence"]

    # Reconstruction loss (only on masked positions)
    # Weighted by 1/mask_ratio (LLaDA-style: harder = more credit)
    token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none"
    ).view_as(labels)

    # Only count loss on masked positions
    masked_loss = token_loss * mask_positions.float()
    # Weight by inverse mask ratio (per sample)
    weights = (1.0 / mask_ratio.clamp(min=0.1)).unsqueeze(-1)
    weighted_loss = (masked_loss * weights).sum() / mask_positions.float().sum().clamp(min=1)

    result = {
        "loss": weighted_loss,
        "token_loss": weighted_loss.detach(),
    }

    # Phase 4: Add confidence supervision
    if phase == 4:
        # Target: high confidence where prediction matches label
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct = (predictions == labels).float()

        # Confidence should predict correctness
        # Use logits version for autocast safety
        conf_logits = output["conf_logits"]
        conf_loss = F.binary_cross_entropy_with_logits(
            conf_logits[mask_positions],
            correct[mask_positions],
        )
        result["loss"] = result["loss"] + 0.1 * conf_loss
        result["conf_loss"] = conf_loss.detach()

    return result


@torch.no_grad()
def evaluate(
    model: LatentCanvasModel,
    dataloader: torch.utils.data.DataLoader,
    phase: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_masked = 0

    for batch in dataloader:
        result = compute_loss(model, batch, phase, device)
        total_loss += result["loss"].item()

        # Accuracy
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask_positions = batch["mask_positions"].to(device)

        output = model(input_ids)
        predictions = output["logits"].argmax(dim=-1)
        correct = (predictions == labels) & mask_positions
        total_correct += correct.sum().item()
        total_masked += mask_positions.sum().item()

    model.train()
    return {
        "val_loss": total_loss / len(dataloader),
        "val_acc": total_correct / max(total_masked, 1) * 100,
    }


@torch.no_grad()
def generate_samples(
    model: LatentCanvasModel,
    tokenizer,
    num_samples: int = 3,
    prompt: str = "Once upon a time",
    gen_length: int = 50,
    device: torch.device = torch.device("cuda"),
):
    """Generate samples and print them."""
    model.eval()

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_tensor = torch.tensor([prompt_ids], device=device)

    print(f"\nGenerating from: '{prompt}'")
    print("-" * 50)

    for i in range(num_samples):
        # Use top-k generation (more reliable than threshold-based)
        canvas, history = model.generate_topk(
            prompt_tensor,
            gen_length=gen_length,
            num_steps=15,
            temperature=0.8,
        )

        text = tokenizer.decode(canvas[0].cpu().tolist())
        print(f"Sample {i+1} ({len(history)} iterations): {text[:100]}...")

    model.train()


def train(args):
    """Main training loop."""
    torch.manual_seed(args.seed)

    # Setup phase
    args = setup_phase(args)

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    # Load data
    print("\nLoading data...")
    train_tokens, vocab_size = load_tinystories(args.num_samples, "train", args.seed)
    val_tokens, _ = load_tinystories(min(1000, args.num_samples // 5), "validation", args.seed + 1)

    # Create model
    config = LatentCanvasConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        num_refine_steps=args.num_refine_steps,
    )

    model = LatentCanvasModel(config)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create dataloaders
    train_loader = create_dataloader(
        train_tokens,
        mask_token_id=config.mask_token_id,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        min_mask_ratio=args.min_mask_ratio,
        max_mask_ratio=args.max_mask_ratio,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_tokens,
        mask_token_id=config.mask_token_id,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        min_mask_ratio=args.min_mask_ratio,
        max_mask_ratio=args.max_mask_ratio,
        shuffle=False,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    # Mixed precision
    scaler = GradScaler() if args.fp16 else None

    # Training loop
    best_val_loss = float("inf")
    global_step = 0

    print(f"\nStarting training: {args.epochs} epochs, {len(train_loader)} batches/epoch")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if args.fp16:
                with autocast():
                    result = compute_loss(model, batch, args.phase, device)
                scaler.scale(result["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                result = compute_loss(model, batch, args.phase, device)
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            epoch_loss += result["loss"].item()
            global_step += 1

            # Logging
            if global_step % 100 == 0:
                conf_str = f" | Conf Loss: {result.get('conf_loss', 0):.4f}" if 'conf_loss' in result else ""
                print(f"Step {global_step} | Loss: {result['token_loss']:.4f}{conf_str}")

            # Evaluation
            if global_step % args.eval_every == 0:
                val_metrics = evaluate(model, val_loader, args.phase, device)
                print(f"  Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_acc']:.2f}%")

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "args": args,
                        "global_step": global_step,
                        "val_loss": best_val_loss,
                    }, f"{args.checkpoint_dir}/phase{args.phase}_best.pt")
                    print(f"  Saved best checkpoint (val_loss={best_val_loss:.4f})")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, args.phase, device)

        print(f"\nEpoch {epoch+1}/{args.epochs} completed in {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_acc']:.2f}%")

    # Save final checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "args": args,
        "global_step": global_step,
    }, f"{args.checkpoint_dir}/phase{args.phase}_final.pt")

    # Generate samples
    print("\n=== Sample Generations ===")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    generate_samples(model, tokenizer, num_samples=3, device=device)

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
