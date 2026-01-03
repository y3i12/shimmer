"""
Shimmer Training Script

Train LIRA or Dialectic models.

Models:
- lira: Latent Iterative Refinement Architecture (original Shimmer)
- dialectic: Bidirectional layer negotiation with backtracking

Phases (apply to both models):
1. Single refine pass - baseline masked reconstruction
2. Multiple refine passes - test iterative refinement hypothesis
3. Confidence supervision - learn to predict uncertainty

Note: Variable corruption (10-100%) was removed as the model learns implicit
remasking through iterative refinement - explicit variable masking didn't help.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import argparse
import time
import datetime
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from pathlib import Path
from dataclasses import dataclass


def clear_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

from lira import LatentCanvasConfig, LatentCanvasModel, count_parameters
from lira.dialectic import DialecticConfig, DialecticCanvas
from lira.hybrid import HybridConfig, HybridShimmer, count_global_parameters
from lira.gpt import GPTConfig, GPTModel, count_gpt_parameters
from data import load_dataset_by_name, create_dataloader, create_gpt_dataloader, get_default_prompt, list_datasets


@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage in progressive training."""
    stage: int
    num_refine_steps: int  # Default K if not using random_k
    min_mask_ratio: float
    max_mask_ratio: float
    use_confidence_loss: bool
    data_start: float  # 0.0 - 1.0 (fraction of dataset)
    data_end: float
    description: str
    # Stage-aware K ranges for random_k mode
    k_min: int = 1
    k_max: int = 8


# Progressive curriculum: 3 stages with fresh data each
CURRICULUM_STAGES = {
    1: CurriculumStage(
        stage=1,
        num_refine_steps=2,
        min_mask_ratio=0.15,
        max_mask_ratio=0.45,
        use_confidence_loss=True,
        data_start=0.0,
        data_end=0.333,
        description="K=1-2, variable 15-45% masking",
        k_min=1,
        k_max=2,
    ),
    2: CurriculumStage(
        stage=2,
        num_refine_steps=4,
        min_mask_ratio=0.20,
        max_mask_ratio=0.60,
        use_confidence_loss=True,
        data_start=0.333,
        data_end=0.666,
        description="K=1-4, variable 20-60% masking",
        k_min=1,
        k_max=3,
    ),
    3: CurriculumStage(
        stage=3,
        num_refine_steps=8,
        min_mask_ratio=0.30,
        max_mask_ratio=0.70,
        use_confidence_loss=True,
        data_start=0.666,
        data_end=1.0,
        description="K=1-7, variable 30-70% masking",
        k_min=1,
        k_max=4,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Shimmer Models")

    # Model selection
    parser.add_argument("--model", type=str, default="lira", choices=["lira", "dialectic", "hybrid", "gpt"],
                        help="Model architecture: lira (original), dialectic (backtracking), hybrid (global coherence), or gpt (autoregressive baseline)")

    # Phase selection
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                        help="Training phase (1-3)")

    # Data
    parser.add_argument("--dataset", type=str, default="tinystories",
                        choices=["tinystories", "everyday", "bitext", "arena", "blend", "agentic"],
                        help="Dataset: tinystories, everyday (2K), bitext (27K), arena (33K), blend (450K instruction mix)")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=None,
                        help="Sliding window stride. If set, long docs are split into overlapping windows. "
                             "E.g., --stride 256 with --max_seq_len 512 = 50%% overlap. Default: None (truncate)")
    parser.add_argument("--vocab_size", type=int, default=0,
                        help="0=GPT-2 (50k), >0=custom SentencePiece tokenizer")
    parser.add_argument("--batch_size", type=int, default=32)

    # Model architecture (shared)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--token_head_type", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Token head: linear (D×V), mlp (D×D + D×V, adds non-linear feature mixing)")
    parser.add_argument("--confidence_head_type", type=str, default="mlp",
                        choices=["linear", "mlp", "entropy"],
                        help="Confidence head: linear (~1K), mlp (~660K), entropy (~4.2M, uses logits)")
    parser.add_argument("--use_coherence_head", action="store_true",
                        help="Enable ELECTRA-style coherence head (detects replaced tokens)")
    parser.add_argument("--coherence_replace_prob", type=float, default=0.15,
                        help="Probability of replacing tokens for coherence training (default: 0.15)")

    # LIRA-specific
    parser.add_argument("--num_refine_steps", type=int, default=1,
                        help="LIRA refinement iterations (Phase 2+ uses >1)")
    parser.add_argument("--use_progress_embedding", action="store_true",
                        help="Inject iteration progress signal (0→1) into RefineBlock")
    parser.add_argument("--random_k", action="store_true",
                        help="Sample random K per batch (requires --use_progress_embedding for best results)")
    parser.add_argument("--k_min", type=int, default=1,
                        help="Minimum K for random sampling (default: 1)")
    parser.add_argument("--k_max", type=int, default=16,
                        help="Maximum K for random sampling (default: 16)")

    # Hybrid Mamba settings
    parser.add_argument("--hybrid_mode", type=str, default=None,
                        choices=["attention", "mamba", "parallel", "interleaved", "adaptive"],
                        help="Hybrid mode: attention (pure), mamba (pure), parallel (Hymba), interleaved (Nemotron), adaptive (iteration-aware)")
    parser.add_argument("--mamba_ratio", type=float, default=0.75,
                        help="Ratio of Mamba layers for interleaved/adaptive modes")
    parser.add_argument("--state_size", type=int, default=16,
                        help="Mamba SSM state dimension")

    # Dialectic-specific
    parser.add_argument("--max_backtracks", type=int, default=3,
                        help="Max backtrack rounds per dialectic pass")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Dialectic confidence threshold for acceptance")

    # Hybrid-specific (global coherence)
    parser.add_argument("--num_global_layers", type=int, default=12,
                        help="Number of sparse global attention layers")
    parser.add_argument("--global_heads", type=int, default=12,
                        help="Number of attention heads in global layers")
    parser.add_argument("--global_freq_train", type=int, default=1,
                        help="Apply global attention every N refine steps during training")
    parser.add_argument("--global_freq_gen", type=int, default=1,
                        help="Apply global attention every N steps during generation")

    # Phase-specific: masking
    parser.add_argument("--min_mask_ratio", type=float, default=0.2,
                        help="Min mask ratio (default 0.2, Phase 3 uses 0.1)")
    parser.add_argument("--max_mask_ratio", type=float, default=0.6,
                        help="Max mask ratio (default 0.6, Phase 3 uses 1.0)")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Accumulate gradients over N batches (effective batch = batch_size * N)")

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (0, 1, etc.). Use -1 for CPU.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile() for faster training (PyTorch 2.0+)")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode: default (stable), reduce-overhead (CUDA graphs - may conflict with embeddings), max-autotune (slower compile, faster run)")

    # Checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Load model weights from checkpoint (for fine-tuning or resuming)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training: restore optimizer, scheduler, epoch (use with --load_checkpoint)")
    parser.add_argument("--checkpoint_name", type=str, default=None,
                        help="Custom checkpoint name prefix")
    parser.add_argument("--eval_every", type=int, default=180)

    # Progressive curriculum (Option C: single run through all stages)
    parser.add_argument("--progressive", action="store_true",
                        help="Progressive curriculum: single run through all 3 stages with fresh data each")
    parser.add_argument("--stage_epochs", type=int, default=3,
                        help="Epochs per curriculum stage (total epochs = 3 × stage_epochs)")
    parser.add_argument("--start_stage", type=int, default=1, choices=[1, 2, 3],
                        help="Start from this stage (useful for resuming progressive training)")

    return parser.parse_args()


def setup_phase(args):
    """Configure settings based on phase."""
    phase_configs = {
        1: {  # Baseline: single pass, variable masking
            "num_refine_steps": 2,
            "min_mask_ratio": 0.1,
            "max_mask_ratio": 0.3,
            "description": "Single refine pass, variable 20-60% masking"
        },
        2: {  # TRM hypothesis: multiple passes
            "num_refine_steps": 4,  # Try 4 iterations
            "min_mask_ratio": 0.15,
            "max_mask_ratio": 0.45,
            "description": "Multiple refine passes (K=4), variable 20-60% masking"
        },
        3: {  # Full: iterative refinement with confidence
            "num_refine_steps": 8,
            "min_mask_ratio": 0.2,
            "max_mask_ratio": 0.6,
            "description": "Full iterative refinement with confidence supervision"
        }
    }

    config = phase_configs[args.phase]

    # Only override if not explicitly set by user
    if args.num_refine_steps == 1 and args.phase > 1:
        args.num_refine_steps = config["num_refine_steps"]
    # Apply phase-specific mask ratios if using defaults
    if args.min_mask_ratio == 0.2 or args.min_mask_ratio == 0.3:
        args.min_mask_ratio = config["min_mask_ratio"]
    if args.max_mask_ratio == 0.6 or args.max_mask_ratio == 0.3:
        args.max_mask_ratio = config["max_mask_ratio"]

    print(f"\n=== Phase {args.phase}: {config['description']} ===")
    print(f"  Model: {args.model.upper()}")
    print(f"  Dataset: {args.dataset}")
    if args.hybrid_mode:
        print(f"  Hybrid mode: {args.hybrid_mode} (mamba_ratio={args.mamba_ratio})")
    print(f"  Refine steps: {args.num_refine_steps}")
    print(f"  Mask ratio: {args.min_mask_ratio:.1%} - {args.max_mask_ratio:.1%}")
    if args.model == "dialectic":
        print(f"  Max backtracks: {args.max_backtracks}")
        print(f"  Confidence threshold: {args.confidence_threshold}")
    if args.model == "hybrid":
        print(f"  Global layers: {args.num_global_layers}")
        print(f"  Global heads: {args.global_heads}")
        print(f"  Global freq (train/gen): {args.global_freq_train}/{args.global_freq_gen}")

    return args


def create_model(args, vocab_size: int, mask_token_id: int):
    """Create model based on --model flag."""
    if args.model == "gpt":
        config = GPTConfig(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
        )
        model = GPTModel(config)
    elif args.model == "lira":
        config = LatentCanvasConfig(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            num_refine_steps=args.num_refine_steps,
            token_head_type=args.token_head_type,
            confidence_head_type=args.confidence_head_type,
            use_progress_embedding=args.use_progress_embedding,
            use_coherence_head=getattr(args, 'use_coherence_head', False),
            mask_token_id=mask_token_id,
            hybrid_mode=args.hybrid_mode,
            mamba_ratio=args.mamba_ratio,
            state_size=args.state_size,
        )
        model = LatentCanvasModel(config)
    elif args.model == "hybrid":
        config = HybridConfig(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            num_refine_steps=args.num_refine_steps,
            mask_token_id=mask_token_id,
            # Global coherence settings
            num_global_layers=args.num_global_layers,
            global_heads=args.global_heads,
            global_frequency_train=args.global_freq_train,
            global_frequency_gen=args.global_freq_gen,
            use_coherence_loss=True,
        )
        model = HybridShimmer(config)
    else:  # dialectic
        config = DialecticConfig(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            max_backtracks=args.max_backtracks,
            confidence_threshold=args.confidence_threshold,
            mask_token_id=mask_token_id,
        )
        model = DialecticCanvas(config)

    return model, config


def compute_loss(
    model,
    batch: dict,
    phase: int,
    device: torch.device,
    model_type: str = "lira",
    num_refine_steps: int = 1,
    random_k: bool = False,
    k_min: int = 1,
    k_max: int = 16,
    use_coherence: bool = False,
    coherence_replace_prob: float = 0.15,
    vocab_size: int = 10000,
) -> dict[str, torch.Tensor]:
    """
    Compute loss based on phase and model type.

    Phase 1-3: Standard reconstruction loss
    Phase 4: Reconstruction + confidence loss
    GPT: Next-token prediction loss (ignores phase)
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Sample random K if enabled
    if random_k:
        import random
        num_refine_steps = random.randint(k_min, k_max)

    # GPT uses different loss computation
    if model_type == "gpt":
        output = model(input_ids, labels=labels)
        return {
            "loss": output["loss"],
            "token_loss": output["loss"].detach(),
        }

    # LIRA/Dialectic/Hybrid use masked reconstruction
    mask_positions = batch["mask_positions"].to(device)
    mask_ratio = batch["mask_ratio"].to(device)

    # ELECTRA-style coherence training: replace some tokens with random ones
    coherence_labels = None
    if use_coherence and model_type == "lira":
        # Create coherence labels: 1 = original, 0 = replaced
        coherence_labels = torch.ones_like(input_ids, dtype=torch.float)

        # Only replace non-masked positions (we want to detect replaced TOKENS, not masks)
        replaceable = ~mask_positions
        replace_mask = (torch.rand_like(input_ids.float()) < coherence_replace_prob) & replaceable

        # Replace with random tokens
        random_tokens = torch.randint(0, vocab_size, input_ids.shape, device=device)
        input_ids = torch.where(replace_mask, random_tokens, input_ids)

        # Labels: 0 for replaced, 1 for original
        coherence_labels[replace_mask] = 0.0

    # Forward pass (different output keys for each model)
    if model_type == "lira":
        output = model(input_ids, num_refine_steps=num_refine_steps, return_confidence=True, return_coherence=use_coherence)
        logits = output["logits"]
        confidence = output["confidence"]
        conf_logits = output.get("conf_logits")
        coh_logits = output.get("coh_logits")
    elif model_type == "hybrid":
        output = model(input_ids, num_refine_steps=num_refine_steps, return_confidence=True, return_coherence=True, training=True)
        logits = output["logits"]
        confidence = output["confidence"]
        conf_logits = output.get("conf_logits")
        coherence = output.get("coherence")  # Sequence-level coherence
    else:  # dialectic
        output = model(input_ids, num_refine_steps=num_refine_steps)
        logits = output["token_logits"]
        confidence = output["confidence"].squeeze(-1)  # [B, L, 1] -> [B, L]
        conf_logits = None  # Dialectic uses different confidence mechanism

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

    # Coherence loss (ELECTRA-style: detect replaced tokens)
    if use_coherence and model_type == "lira" and coh_logits is not None and coherence_labels is not None:
        # Compute coherence loss on non-masked positions (where we replaced tokens)
        non_masked = ~mask_positions
        if non_masked.any():
            coh_loss = F.binary_cross_entropy_with_logits(
                coh_logits[non_masked],
                coherence_labels[non_masked],
            )
            result["loss"] = result["loss"] + 0.5 * coh_loss  # Weight coherence loss
            result["coh_loss"] = coh_loss.detach()

    # Phase 4: Add confidence supervision
    if phase == 4:
        # Compute targets for confidence supervision
        with torch.no_grad():
            # Binary target (for models that need it)
            predictions = logits.argmax(dim=-1)
            correct = (predictions == labels).float()

            # Soft target: probability assigned to correct token
            # This gives continuous gradient signal instead of binary
            probs = F.softmax(logits, dim=-1)
            soft_target = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        if model_type == "hybrid" and conf_logits is not None:
            # Hybrid: Use logits version for autocast safety + coherence loss
            conf_loss = F.binary_cross_entropy_with_logits(
                conf_logits[mask_positions],
                correct[mask_positions],
            )
            # Add coherence supervision if available
            if coherence is not None:
                # Target: high coherence when accuracy is high
                # Compute per-sample accuracy (handle variable mask counts)
                masked_correct = (correct * mask_positions.float())
                batch_accuracy = masked_correct.sum(dim=1) / mask_positions.float().sum(dim=1).clamp(min=1)
                coherence_loss = F.mse_loss(coherence, batch_accuracy)
                conf_loss = conf_loss + 0.1 * coherence_loss
                result["coherence_loss"] = coherence_loss.detach()
        elif model_type == "lira" and conf_logits is not None:
            # LIRA: MSE with soft targets
            # Confidence should match probability of correct token
            # This gives continuous gradient (not binary) - breaks the plateau!
            pred_confidence = torch.sigmoid(conf_logits[mask_positions])
            conf_loss = F.mse_loss(pred_confidence, soft_target[mask_positions])
        elif model_type == "dialectic" and "layer_conf_logits" in output:
            # Dialectic: Supervise PER-LAYER confidence (the actual backtrack signal!)
            # Each layer should be confident when its contribution leads to correct prediction
            # Using LOGITS for autocast safety
            layer_conf_logits = output["layer_conf_logits"]

            conf_losses = []
            for layer_logits in layer_conf_logits:
                if layer_logits is not None:
                    # layer_logits is [B, L], correct is [B, L]
                    layer_logits_masked = layer_logits[mask_positions]
                    correct_masked = correct[mask_positions]
                    # Use with_logits for autocast safety
                    layer_loss = F.binary_cross_entropy_with_logits(layer_logits_masked, correct_masked)
                    conf_losses.append(layer_loss)

            if conf_losses:
                conf_loss = torch.stack(conf_losses).mean()

                # ADD: Decisiveness loss - push confidence away from 0.5
                # When correct, reward high confidence (incentivize halting)
                # When wrong, reward low confidence (incentivize backtracking)
                # This uses entropy-like regularization: -p*log(p) - (1-p)*log(1-p)
                # Minimizing this pushes toward 0 or 1
                all_conf_logits = torch.cat([l[mask_positions] for l in layer_conf_logits if l is not None])
                all_conf_probs = torch.sigmoid(all_conf_logits)
                # Entropy: high when p=0.5, low when p near 0 or 1
                eps = 1e-7
                entropy = -(all_conf_probs * torch.log(all_conf_probs + eps) +
                           (1 - all_conf_probs) * torch.log(1 - all_conf_probs + eps))
                decisiveness_loss = entropy.mean()  # Minimize entropy = be decisive

                conf_loss = conf_loss + 0.5 * decisiveness_loss  # Weight the decisiveness term
                result["decisiveness_loss"] = decisiveness_loss.detach()
            else:
                conf_loss = torch.tensor(0.0, device=device)
        else:
            # Fallback: final confidence head
            conf_loss = F.binary_cross_entropy(
                confidence[mask_positions].clamp(1e-7, 1-1e-7),
                correct[mask_positions],
            )

        result["loss"] = result["loss"] + 0.1 * conf_loss
        result["conf_loss"] = conf_loss.detach()

    # Dialectic-specific: track negotiation stats
    if model_type == "dialectic" and "negotiation_stats" in output:
        stats = output["negotiation_stats"]
        if stats:
            total_backtracks = sum(s["total_backtracks"] for s in stats)
            result["backtracks"] = torch.tensor(total_backtracks, device=device)

    return result


@torch.no_grad()
def evaluate(
    model,
    dataloader: torch.utils.data.DataLoader,
    phase: int,
    device: torch.device,
    model_type: str = "lira",
    num_refine_steps: int = 1,
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_masked = 0
    total_backtracks = 0
    total_tokens = 0

    for batch in dataloader:
        result = compute_loss(model, batch, phase, device, model_type, num_refine_steps)
        total_loss += result["loss"].item()

        if "backtracks" in result:
            total_backtracks += result["backtracks"].item()

        # Accuracy
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if model_type == "gpt":
            # GPT: next-token prediction accuracy
            output = model(input_ids)
            logits = output["logits"]
            # Shift for comparison
            predictions = logits[:, :-1, :].argmax(dim=-1)
            shifted_labels = labels[:, 1:]
            # Only count non-padding tokens
            valid_mask = shifted_labels != -100
            correct = (predictions == shifted_labels) & valid_mask
            total_correct += correct.sum().item()
            total_tokens += valid_mask.sum().item()
        else:
            # LIRA/Dialectic/Hybrid: masked prediction accuracy
            mask_positions = batch["mask_positions"].to(device)

            if model_type == "lira":
                output = model(input_ids)
                predictions = output["logits"].argmax(dim=-1)
            elif model_type == "hybrid":
                output = model(input_ids, training=False)
                predictions = output["logits"].argmax(dim=-1)
            else:  # dialectic
                output = model(input_ids, num_refine_steps=num_refine_steps)
                predictions = output["token_logits"].argmax(dim=-1)

            correct = (predictions == labels) & mask_positions
            total_correct += correct.sum().item()
            total_masked += mask_positions.sum().item()

    model.train()

    metrics = {
        "val_loss": total_loss / len(dataloader),
    }

    if model_type == "gpt":
        metrics["val_acc"] = total_correct / max(total_tokens, 1) * 100
    else:
        metrics["val_acc"] = total_correct / max(total_masked, 1) * 100

    if model_type == "dialectic":
        metrics["avg_backtracks"] = total_backtracks / len(dataloader)

    return metrics


@torch.no_grad()
def generate_samples(
    model,
    tokenizer,
    num_samples: int = 3,
    prompt: str = "Once upon a time",
    gen_length: int = 50,
    device: torch.device = torch.device("cuda"),
    model_type: str = "lira",
    num_refine_steps: int = 1,
):
    """Generate samples and print them."""
    model.eval()

    # Handle both GPT-2 and custom ShimmerTokenizer
    from lira.tokenizer import ShimmerTokenizer
    if isinstance(tokenizer, ShimmerTokenizer):
        prompt_ids = tokenizer.encode(prompt, add_bos=False, add_eos=False)
    else:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_tensor = torch.tensor([prompt_ids], device=device)

    print(f"\nGenerating from: '{prompt}'")
    print("-" * 50)

    for i in range(num_samples):
        if model_type == "gpt":
            # GPT: autoregressive generation
            output = model.generate(
                prompt_tensor,
                max_new_tokens=gen_length,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
            )
            text = tokenizer.decode(output[0].cpu().tolist())
            iterations = gen_length
        elif model_type == "lira":
            canvas, history = model.generate_topk(
                prompt_tensor,
                gen_length=gen_length,
                num_steps=max(15, gen_length * 2),
                temperature=0.8,
            )
            text = tokenizer.decode(canvas[0].cpu().tolist())
            iterations = len(history)
        elif model_type == "hybrid":
            canvas, history = model.generate_topk(
                prompt_tensor,
                gen_length=gen_length,
                num_steps=max(15, gen_length * 2),
                temperature=0.8,
            )
            text = tokenizer.decode(canvas[0].cpu().tolist())
            iterations = len(history)
        else:  # dialectic
            canvas = model.generate_dialectic(
                prompt_tensor,
                max_new_tokens=gen_length,
                num_refine_steps=num_refine_steps,
                temperature=0.8,
            )
            text = tokenizer.decode(canvas[0].cpu().tolist())
            iterations = gen_length  # Dialectic does one token per iteration

        print(f"Sample {i+1} ({iterations} iters): {text[:120]}...")

    model.train()


def train_progressive(args):
    """
    Progressive curriculum training - single run through all 3 stages.

    Key features:
    - Fresh data for each stage (no repetition across stages)
    - Gradual complexity increase: K=1 → K=4
    - Confidence supervision only in final stage
    - Single checkpoint lineage
    """
    torch.manual_seed(args.seed)

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    total_epochs = args.stage_epochs * 3
    print(f"\n{'='*60}")
    print(f"PROGRESSIVE CURRICULUM TRAINING")
    print(f"{'='*60}")
    print(f"  Model: {args.model.upper()}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Total samples: {args.num_samples}")
    print(f"  Samples per stage: ~{args.num_samples // 3}")
    print(f"  Epochs per stage: {args.stage_epochs}")
    print(f"  Total epochs: {total_epochs}")
    print(f"  Start stage: {args.start_stage}")
    print(f"{'='*60}\n")

    # Gradient accumulation
    accum_steps = args.gradient_accumulation_steps
    effective_batch_size = args.batch_size * accum_steps

    # Load ALL data upfront
    print(f"Loading data from '{args.dataset}'...")
    all_train_tokens, vocab_size, tokenizer = load_dataset_by_name(
        args.dataset, args.num_samples, "train", args.seed,
        vocab_size=args.vocab_size, return_tokenizer=True
    )
    val_tokens, _, _ = load_dataset_by_name(
        args.dataset, min(1000, args.num_samples // 5), "validation",
        args.seed + 1, vocab_size=args.vocab_size, return_tokenizer=True
    )

    # Split data into 3 parts (one per stage) - no overlap = no repetition
    n = len(all_train_tokens)
    stage_data = {
        1: all_train_tokens[0:n//3],
        2: all_train_tokens[n//3:2*n//3],
        3: all_train_tokens[2*n//3:n],
    }

    print(f"\nData split across stages:")
    for stage_num, data in stage_data.items():
        print(f"  Stage {stage_num}: {len(data)} samples")

    # Get mask_token_id from tokenizer
    from lira.tokenizer import ShimmerTokenizer
    if isinstance(tokenizer, ShimmerTokenizer):
        mask_token_id = tokenizer.mask_token_id
    else:
        mask_token_id = vocab_size

    # Create model
    model, config = create_model(args, vocab_size, mask_token_id)
    print(f"\n{args.model.upper()} model parameters: {count_parameters(model):,}")

    if args.model == "hybrid":
        param_info = count_global_parameters(model)
        print(f"  Core (LIRA): {param_info['core']:,}")
        print(f"  Global layers: {param_info['global']:,} ({param_info['global_ratio']:.1%})")
        print(f"  Coherence head: {param_info['coherence']:,}")

    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"\nLoading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            print("  Stripped _orig_mod. prefix from compiled checkpoint")

        try:
            model.load_state_dict(state_dict, strict=False)
            print("  Loaded model weights (strict=False)")
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")

    # Setup device
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
        print(f"\nUsing CPU")
    else:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
        gpu_name = torch.cuda.get_device_name(args.gpu)
        gpu_mem = torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3
        print(f"\nUsing GPU {args.gpu}: {gpu_name} ({gpu_mem:.1f}GB)")

    model = model.to(device)

    # Optional: compile model
    if args.compile and hasattr(torch, 'compile'):
        print(f"\nCompiling model with torch.compile(mode='{args.compile_mode}')...")
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print("  ✓ Model compiled successfully")
        except Exception as e:
            print(f"  ⚠ Compilation failed: {e}")

    # Optimizer (single optimizer for entire training)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler spans all epochs
    total_steps_estimate = sum(
        (len(stage_data[s]) // effective_batch_size) * args.stage_epochs
        for s in range(1, 4)
    ) // args.gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, total_steps_estimate))

    # Mixed precision
    scaler = GradScaler('cuda') if args.fp16 else None

    # Checkpoint naming
    ckpt_prefix = args.checkpoint_name or f"{args.model}_{args.dataset}_progressive"

    # Training state
    best_val_loss = float("inf")
    global_step = 0
    start_time = datetime.datetime.now()

    print(f"\nGradient accumulation: {accum_steps} steps → effective batch size: {effective_batch_size}")

    # ========== PROGRESSIVE TRAINING LOOP ==========
    current_stage = 0
    train_loader = None

    for epoch in range(total_epochs):
        # Determine current stage (1-indexed)
        new_stage = (epoch // args.stage_epochs) + 1

        # Skip stages before start_stage
        if new_stage < args.start_stage:
            print(f"\nSkipping epoch {epoch+1} (stage {new_stage} < start_stage {args.start_stage})")
            continue

        # Stage transition?
        if new_stage != current_stage:
            current_stage = new_stage
            stage_config = CURRICULUM_STAGES[current_stage]

            print(f"\n{'='*60}")
            print(f"STAGE {current_stage}/3: {stage_config.description}")
            print(f"{'='*60}")
            print(f"  Refinement steps (K): {stage_config.num_refine_steps}")
            print(f"  Mask ratio: {stage_config.min_mask_ratio:.0%} - {stage_config.max_mask_ratio:.0%}")
            print(f"  Confidence loss: {'ON' if stage_config.use_confidence_loss else 'OFF'}")
            print(f"  Data samples: {len(stage_data[current_stage])}")
            print(f"  Epochs in this stage: {args.stage_epochs}")
            print(f"{'='*60}\n")

            # Create new dataloader for this stage's data
            train_loader = create_dataloader(
                stage_data[current_stage],
                mask_token_id=config.mask_token_id,
                batch_size=args.batch_size,
                max_seq_len=args.max_seq_len,
                min_mask_ratio=stage_config.min_mask_ratio,
                max_mask_ratio=stage_config.max_mask_ratio,
                shuffle=True,
                stride=args.stride,
            )

            # Validation loader uses same masking as current stage
            val_loader = create_dataloader(
                val_tokens,
                mask_token_id=config.mask_token_id,
                batch_size=args.batch_size,
                max_seq_len=args.max_seq_len,
                min_mask_ratio=stage_config.min_mask_ratio,
                max_mask_ratio=stage_config.max_mask_ratio,
                shuffle=False,
                stride=args.stride,
            )

        # Current stage config
        stage_config = CURRICULUM_STAGES[current_stage]
        phase_for_loss = 4 if stage_config.use_confidence_loss else 1

        epoch_start = time.time()
        epoch_loss = 0
        epoch_batches = 0
        last_log = datetime.datetime.now()
        last_val = datetime.datetime.now()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            is_accumulating = ((batch_idx + 1) % accum_steps != 0) and (batch_idx + 1 < len(train_loader))

            if args.compile and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                torch.compiler.cudagraph_mark_step_begin()

            # Use stage-specific K ranges if random_k enabled
            stage_k_min = stage_config.k_min if args.random_k else args.k_min
            stage_k_max = stage_config.k_max if args.random_k else args.k_max

            if args.fp16:
                with autocast('cuda'):
                    result = compute_loss(
                        model, batch, phase_for_loss, device,
                        args.model, stage_config.num_refine_steps,
                        random_k=args.random_k, k_min=stage_k_min, k_max=stage_k_max,
                        use_coherence=getattr(args, 'use_coherence_head', False),
                        coherence_replace_prob=getattr(args, 'coherence_replace_prob', 0.15),
                        vocab_size=vocab_size,
                    )
                    scaled_loss = result["loss"] / effective_batch_size
                scaler.scale(scaled_loss).backward()

                if not is_accumulating:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                result = compute_loss(
                    model, batch, phase_for_loss, device,
                    args.model, stage_config.num_refine_steps,
                    random_k=args.random_k, k_min=stage_k_min, k_max=stage_k_max,
                    use_coherence=getattr(args, 'use_coherence_head', False),
                    coherence_replace_prob=getattr(args, 'coherence_replace_prob', 0.15),
                    vocab_size=vocab_size,
                )
                scaled_loss = result["loss"] / effective_batch_size
                scaled_loss.backward()

                if not is_accumulating:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

            if not is_accumulating:
                scheduler.step()

            epoch_loss += result["loss"].item()
            epoch_batches += 1
            global_step += args.batch_size

            now = datetime.datetime.now()

            # Periodic logging
            if (now - last_log).total_seconds() > 10:
                last_log = now
                avg_loss = (epoch_loss / epoch_batches if epoch_batches > 0 else 0) / effective_batch_size
                elapsed = (now - start_time).total_seconds()

                log_parts = [
                    f"Stage {current_stage}",
                    f"Epoch {epoch+1}/{total_epochs}",
                    f"Step {global_step}",
                    f"Loss {avg_loss:.4f}",
                    f"LR {scheduler.get_last_lr()[0]:.2e}",
                    f"Elapsed {elapsed:.0f}s",
                ]

                if stage_config.use_confidence_loss and "conf_loss" in result:
                    log_parts.append(f"ConfLoss {result['conf_loss'].item():.4f}")

                if "coh_loss" in result:
                    log_parts.append(f"CohLoss {result['coh_loss'].item():.4f}")

                print(" | ".join(log_parts))

            # Periodic validation
            if (now - last_val).total_seconds() > args.eval_every:
                val_metrics = evaluate(
                    model, val_loader, phase_for_loss, device,
                    args.model, stage_config.num_refine_steps
                )
                last_val = datetime.datetime.now()

                val_loss = val_metrics['val_loss'] / args.batch_size
                train_loss = (epoch_loss / epoch_batches if epoch_batches > 0 else 0) / effective_batch_size

                print(f"\n  Validation: TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | ValAcc {val_metrics['val_acc']:.2f}%\n")

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if scaler else None,
                        "epoch": epoch,
                        "stage": current_stage,
                        "config": config,
                        "args": args,
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "progressive": True,
                    }, f"{args.checkpoint_dir}/{ckpt_prefix}_best.pt")
                    print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f})")

                clear_memory()

        # End of epoch summary
        epoch_time = time.time() - epoch_start
        val_metrics = evaluate(
            model, val_loader, phase_for_loss, device,
            args.model, stage_config.num_refine_steps
        )

        avg_epoch_loss = (epoch_loss // epoch_batches) / effective_batch_size
        val_loss = val_metrics['val_loss'] / effective_batch_size

        print(f"\n[Stage {current_stage}] Epoch {epoch+1}/{total_epochs} completed in {epoch_time:.1f}s")
        print(f"  TrainLoss: {avg_epoch_loss:.4f} | ValLoss: {val_loss:.4f} | ValAcc: {val_metrics['val_acc']:.2f}%")

        # Save stage checkpoint at end of each stage
        stage_epoch = (epoch % args.stage_epochs) + 1
        if stage_epoch == args.stage_epochs:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "epoch": epoch,
                "stage": current_stage,
                "config": config,
                "args": args,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "progressive": True,
            }, f"{args.checkpoint_dir}/{ckpt_prefix}_stage{current_stage}.pt")
            print(f"  ✓ Saved stage {current_stage} checkpoint")

        # Update best if needed
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]

        clear_memory()

    # Final checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": total_epochs - 1,
        "stage": 3,
        "config": config,
        "args": args,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "progressive": True,
    }, f"{args.checkpoint_dir}/{ckpt_prefix}_final.pt")

    # Generate samples
    print("\n=== Sample Generations ===")
    default_prompt = get_default_prompt(args.dataset)
    generate_samples(
        model, tokenizer, num_samples=3, prompt=default_prompt, device=device,
        model_type=args.model, num_refine_steps=4  # Use full refinement for generation
    )

    print(f"\n{'='*60}")
    print(f"PROGRESSIVE TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"  Total epochs: {total_epochs}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved:")
    print(f"    - {ckpt_prefix}_best.pt")
    print(f"    - {ckpt_prefix}_stage1.pt ... {ckpt_prefix}_stage3.pt")
    print(f"    - {ckpt_prefix}_final.pt")
    print(f"{'='*60}\n")


def train(args):
    """Main training loop."""
    torch.manual_seed(args.seed)

    # Setup phase
    args = setup_phase(args)

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    # Load data
    print(f"\nLoading data from '{args.dataset}'...")
    train_tokens, vocab_size, tokenizer = load_dataset_by_name(
        args.dataset, args.num_samples, "train", args.seed,
        vocab_size=args.vocab_size, return_tokenizer=True
    )
    val_tokens, _, _ = load_dataset_by_name(
        args.dataset, min(1000, args.num_samples // 5), "validation",
        args.seed + 1, vocab_size=args.vocab_size, return_tokenizer=True
    )

    # Get mask_token_id from tokenizer (custom uses 4, GPT-2 uses vocab_size)
    from lira.tokenizer import ShimmerTokenizer
    if isinstance(tokenizer, ShimmerTokenizer):
        mask_token_id = tokenizer.mask_token_id  # 4 for custom tokenizer
    else:
        mask_token_id = vocab_size  # GPT-2 style: use vocab_size

    # Create model
    model, config = create_model(args, vocab_size, mask_token_id)
    print(f"\n{args.model.upper()} model parameters: {count_parameters(model):,}")

    # Show hybrid-specific parameter breakdown
    if args.model == "hybrid":
        param_info = count_global_parameters(model)
        print(f"  Core (LIRA): {param_info['core']:,}")
        print(f"  Global layers: {param_info['global']:,} ({param_info['global_ratio']:.1%})")
        print(f"  Coherence head: {param_info['coherence']:,}")

    # Load checkpoint if specified
    checkpoint = None
    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)

        # Handle loading LIRA checkpoint into Dialectic (or vice versa)
        state_dict = checkpoint["model_state_dict"]

        # Handle state dict from compiled model (torch.compile adds _orig_mod. prefix)
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            print("  Stripped _orig_mod. prefix from compiled checkpoint")

        try:
            model.load_state_dict(state_dict, strict=False)
            print("  Loaded model weights (strict=False)")
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")
            print("  Starting with fresh weights")
            checkpoint = None  # Don't try to resume if model load failed

    # Setup device with GPU selection
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
        print(f"\nUsing CPU")
    else:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)  # Set default for tensors created without explicit device
        gpu_name = torch.cuda.get_device_name(args.gpu)
        gpu_mem = torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3
        print(f"\nUsing GPU {args.gpu}: {gpu_name} ({gpu_mem:.1f}GB)")

    model = model.to(device)

    # Optional: compile model for faster training (PyTorch 2.0+)
    if args.compile:
        if hasattr(torch, 'compile'):
            print(f"\nCompiling model with torch.compile(mode='{args.compile_mode}')...")
            print("  First forward pass will be slow (compiling), then faster.")
            try:
                model = torch.compile(model, mode=args.compile_mode)
                print("  ✓ Model compiled successfully")
            except Exception as e:
                print(f"  ⚠ Compilation failed, falling back to eager mode: {e}")
        else:
            print("  ⚠ torch.compile not available (requires PyTorch 2.0+)")

    # Create dataloaders (GPT uses different dataloader)
    if args.model == "gpt":
        train_loader = create_gpt_dataloader(
            train_tokens,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            shuffle=True,
            stride=args.stride,
        )
        val_loader = create_gpt_dataloader(
            val_tokens,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            shuffle=False,
            stride=args.stride,
        )
    else:
        train_loader = create_dataloader(
            train_tokens,
            mask_token_id=config.mask_token_id,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            min_mask_ratio=args.min_mask_ratio,
            max_mask_ratio=args.max_mask_ratio,
            shuffle=True,
            stride=args.stride,
        )
        val_loader = create_dataloader(
            val_tokens,
            mask_token_id=config.mask_token_id,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            min_mask_ratio=args.min_mask_ratio,
            max_mask_ratio=args.max_mask_ratio,
            shuffle=False,
            stride=args.stride,
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler (account for gradient accumulation - fewer optimizer steps)
    optimizer_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, optimizer_steps))

    # Total samples for time estimation (all batches, not just optimizer steps)
    total_samples = len(train_loader) * args.batch_size * args.epochs

    # Mixed precision
    scaler = GradScaler('cuda') if args.fp16 else None

    # Gradient accumulation setup
    accum_steps = args.gradient_accumulation_steps
    effective_batch_size = args.batch_size * accum_steps

    # Resume training state if requested
    start_epoch = 0
    resumed_global_step = 0
    resumed_best_val_loss = float("inf")
    if args.resume and checkpoint is not None:
        print("\nResuming training state from checkpoint...")

        # Restore optimizer state
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("  ✓ Restored optimizer state")
            except Exception as e:
                print(f"  ⚠ Could not restore optimizer state: {e}")

        # Restore scheduler state
        if "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("  ✓ Restored scheduler state")
            except Exception as e:
                print(f"  ⚠ Could not restore scheduler state: {e}")

        # Restore scaler state (for fp16)
        if scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                print("  ✓ Restored scaler state")
            except Exception as e:
                print(f"  ⚠ Could not restore scaler state: {e}")

        # Restore epoch (continue from next epoch)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f"  ✓ Resuming from epoch {start_epoch}")

        # Restore global step
        if "global_step" in checkpoint:
            resumed_global_step = checkpoint["global_step"]
            print(f"  ✓ Resuming from global step {resumed_global_step}")

        # Restore best val loss
        if "best_val_loss" in checkpoint:
            resumed_best_val_loss = checkpoint["best_val_loss"]
            print(f"  ✓ Best val loss so far: {resumed_best_val_loss / effective_batch_size:.4f}")
    elif args.resume and checkpoint is None:
        print("\n⚠ --resume specified but no checkpoint loaded, starting fresh")

    # Checkpoint naming
    if args.checkpoint_name:
        ckpt_prefix = args.checkpoint_name
    else:
        ckpt_prefix = f"{args.model}_{args.dataset}_phase{args.phase}"
        if args.hybrid_mode:
            ckpt_prefix += f"_{args.hybrid_mode}"


    # Training loop
    best_val_loss = resumed_best_val_loss
    global_step = resumed_global_step
    epoch_loss = 0
    start_time = datetime.datetime.now()
    last_log = datetime.datetime.now()
    last_val = datetime.datetime.now()

    if start_epoch > 0:
        print(f"\nResuming training from epoch {start_epoch}: {args.epochs - start_epoch} epochs remaining, {len(train_loader)} batches/epoch")
    else:
        print(f"\nStarting training: {args.epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"  Gradient accumulation: {accum_steps} steps → effective batch size: {effective_batch_size}")

    last_global_step_log = global_step
    last_validation_log = 0
    average_loss_log = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        # epoch_loss = 0
        doc = 0
        optimizer.zero_grad()  # Zero at start of epoch

        for batch_idx, batch in enumerate(train_loader):
            # Determine if this is an accumulation step or optimizer step
            is_accumulating = ((batch_idx + 1) % accum_steps != 0) and (batch_idx + 1 < len(train_loader))

            # Mark step boundary for CUDA graphs (needed for torch.compile with reduce-overhead)
            if args.compile and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                torch.compiler.cudagraph_mark_step_begin()

            if args.fp16:
                with autocast('cuda'):
                    result = compute_loss(
                        model, batch, args.phase, device,
                        args.model, args.num_refine_steps,
                        random_k=args.random_k, k_min=args.k_min, k_max=args.k_max
                    )
                    # Scale loss for accumulation
                    scaled_loss = result["loss"] / accum_steps
                scaler.scale(scaled_loss).backward()

                if not is_accumulating:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                result = compute_loss(
                    model, batch, args.phase, device,
                    args.model, args.num_refine_steps,
                    random_k=args.random_k, k_min=args.k_min, k_max=args.k_max
                )
                # Scale loss for accumulation
                scaled_loss = result["loss"] / effective_batch_size
                scaled_loss.backward()

                if not is_accumulating:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

            # Step scheduler per optimizer step (not per batch)
            if not is_accumulating:
                scheduler.step()
            loss_item = result["loss"].item() / effective_batch_size
            epoch_loss += loss_item
            global_step += args.batch_size
            doc += args.batch_size
            average_loss_log += loss_item

            now = datetime.datetime.now()

            # Logging
            if (now - last_log).total_seconds() > 5:
                last_log = now
                steps_since_last_log = global_step - last_global_step_log
                
                avg_item_loss_log = average_loss_log / steps_since_last_log
                average_loss_log = 0
                train_loss_log = 0
                if global_step > 0:
                    train_loss_log = (epoch_loss / global_step)

                log_parts = [
                    f"Step {global_step} ({steps_since_last_log:+d})",
                    f"Epoch {epoch}",
                    f"Doc {doc}",
                    f"TrainLoss {train_loss_log:.4f}",
                    f"InstAvgLoss {avg_item_loss_log:.4f} (train{(avg_item_loss_log-train_loss_log):+.4f}/last_val{(avg_item_loss_log-last_validation_log):+.4f})",
                ]
                last_global_step_log = global_step

                # Dialectic-specific logging
                if "backtracks" in result:
                    log_parts.append(f"Backtracks {result['backtracks'].item():.0f}")
                if "decisiveness_loss" in result:
                    log_parts.append(f"Decisive {result['decisiveness_loss'].item():.3f}")

                avg_time = 0
                if global_step > 0:
                    avg_time = (now - start_time).total_seconds() / global_step

                log_parts.extend([
                    f"Elapsed {(now - start_time).total_seconds():.0f}s",
                    f"~Remain {(avg_time * (total_samples - global_step)):.0f}s"
                ])

                print(' | '.join(log_parts))

            # Evaluation
            if (now - last_val).total_seconds() > args.eval_every:
                val_metrics = evaluate(
                    model, val_loader, args.phase, device,
                    args.model, args.num_refine_steps
                )
                last_val = datetime.datetime.now() # after evaluation/validation wait eval_every seconds
                last_validation_log = (val_metrics['val_loss'] / effective_batch_size)
                train_loss = 0
                if global_step > 0:
                    train_loss = (epoch_loss / global_step)
                val_log = f"\nTrainLoss {train_loss:.4f} | ValLoss {last_validation_log:.4f}"
                val_log += f" | Gap {((last_validation_log - train_loss) * 100):.2f}%"
                val_log += f" | ValAcc {val_metrics['val_acc']:.2f}%"

                if "avg_backtracks" in val_metrics:
                    val_log += f" | AvgBacktracks {val_metrics['avg_backtracks']:.1f}"

                print(val_log + "\n")

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if scaler else None,
                        "epoch": epoch,
                        "config": config,
                        "args": args,
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "val_loss": best_val_loss / effective_batch_size,
                        "model_type": args.model,
                        "dataset": args.dataset,
                    }, f"{args.checkpoint_dir}/{ckpt_prefix}_best.pt")
                    print(f"  Saved best checkpoint (val_loss={best_val_loss / effective_batch_size:.4f})")

                # Clear memory after evaluation to reduce fragmentation
                clear_memory()

        # Epoch summary
        epoch_time = time.time() - epoch_start
        val_metrics = evaluate(
            model, val_loader, args.phase, device,
            args.model, args.num_refine_steps
        )

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "config": config,
            "args": args,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "val_loss": best_val_loss / effective_batch_size,
            "model_type": args.model,
            "dataset": args.dataset,
        }, f"{args.checkpoint_dir}/{ckpt_prefix}_last.pt")
        
        epoch_summary = f"\nEpoch {epoch+1}/{args.epochs} completed in {epoch_time:.1f}s"
        if global_step > 0 : 
            epoch_summary += f"\n TrainLoss: {(epoch_loss/global_step):.4f}"
        epoch_summary += f" | Val Loss: {val_metrics['val_loss'] / effective_batch_size:.4f}"
        epoch_summary += f" | Val Acc: {val_metrics['val_acc']:.2f}%"

        if "avg_backtracks" in val_metrics:
            epoch_summary += f" | Avg Backtracks: {val_metrics['avg_backtracks']:.1f}"

        print(epoch_summary + "\n")

        # Clear memory at end of each epoch
        clear_memory()

    # Save final checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": args.epochs - 1,
        "config": config,
        "args": args,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_type": args.model,
        "dataset": args.dataset,
    }, f"{args.checkpoint_dir}/{ckpt_prefix}_final.pt")

    # Generate samples with dataset-appropriate prompt
    print("\n=== Sample Generations ===")
    default_prompt = get_default_prompt(args.dataset)
    generate_samples(
        model, tokenizer, num_samples=3, prompt=default_prompt, device=device,
        model_type=args.model, num_refine_steps=args.num_refine_steps
    )

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    args = parse_args()
    if args.progressive:
        train_progressive(args)
    else:
        train(args)
