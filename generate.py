#!/usr/bin/env python3
"""
Shimmer Generation Script

Generate text from LIRA or Dialectic checkpoints.

Usage:
    # Auto-detect model type from checkpoint
    python generate.py --checkpoint checkpoints/lira_phase1_best.pt --prompt "Once upon a time"

    # Explicit model type
    python generate.py --checkpoint checkpoints/dialectic_phase1_best.pt --model dialectic --prompt "Once upon a time"

    # More samples with different settings
    python generate.py --checkpoint checkpoints/phase4_best.pt --num_samples 5 --max_new_tokens 100 --temperature 0.9
"""

import argparse
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lira import LatentCanvasModel, LatentCanvasConfig
from lira.dialectic import DialecticCanvas, DialecticConfig
from data import get_default_prompt, list_datasets


def load_tokenizer(vocab_size: int, tokenizer_path: str = "tokenizers/shimmer_blend_10000.model"):
    """Load the appropriate tokenizer based on vocab_size."""
    if vocab_size == 0 or vocab_size >= 50000:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Loaded GPT-2 tokenizer (vocab_size={tokenizer.vocab_size})")
        return tokenizer, "gpt2"
    else:
        from lira.tokenizer import ShimmerTokenizer
        if not Path(tokenizer_path).exists():
            print(f"ERROR: Custom tokenizer not found at {tokenizer_path}")
            print("You may need to train with the same vocab_size first to create the tokenizer.")
            sys.exit(1)
        tokenizer = ShimmerTokenizer(str(tokenizer_path))
        print(f"Loaded custom tokenizer (vocab_size={vocab_size}) from {tokenizer_path}")
        return tokenizer, "custom"


def encode_prompt(tokenizer, tokenizer_type: str, prompt: str) -> list[int]:
    """Encode prompt using the appropriate tokenizer interface."""
    if tokenizer_type == "gpt2":
        return tokenizer.encode(prompt, add_special_tokens=False)
    else:
        return tokenizer.encode(prompt, add_bos=False, add_eos=False)


def decode_tokens(tokenizer, tokenizer_type: str, token_ids: list[int]) -> str:
    """Decode tokens using the appropriate tokenizer interface."""
    return tokenizer.decode(token_ids)


def detect_model_type(checkpoint: dict) -> str:
    """Detect model type from checkpoint."""
    # Check if explicitly saved
    if "model_type" in checkpoint:
        return checkpoint["model_type"]

    # Check config type
    config = checkpoint.get("config")
    if config is not None:
        if isinstance(config, DialecticConfig):
            return "dialectic"
        elif isinstance(config, LatentCanvasConfig):
            return "lira"
        # Check for dialectic-specific attributes
        if hasattr(config, 'max_backtracks'):
            return "dialectic"

    # Check state dict keys for dialectic-specific layers
    state_dict = checkpoint.get("model_state_dict", {})
    if any("dialectic" in key or "feedback_gate" in key for key in state_dict.keys()):
        return "dialectic"

    # Default to LIRA
    return "lira"


def create_model(config, model_type: str):
    """Create model based on type."""
    if model_type == "dialectic":
        return DialecticCanvas(config)
    else:
        return LatentCanvasModel(config)


@torch.no_grad()
def generate_lira(
    model: LatentCanvasModel,
    tokenizer,
    tokenizer_type: str,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    num_steps: int = None,
    device: str = "cuda",
    verbose: bool = False,
) -> str:
    """Generate text using LIRA model."""
    model.eval()

    prompt_ids = encode_prompt(tokenizer, tokenizer_type, prompt)
    mask_id = model.config.mask_token_id
    input_ids = prompt_ids + [mask_id] * max_new_tokens
    canvas = torch.tensor([input_ids], dtype=torch.long, device=device)

    prompt_len = len(prompt_ids)
    gen_len = max_new_tokens

    if num_steps is None:
        num_steps = gen_len

    tokens_per_step = max(1, gen_len // num_steps)

    if verbose:
        print(f"Prompt length: {prompt_len}, Generation length: {gen_len}")
        print(f"Steps: {num_steps}, Tokens per step: {tokens_per_step}")

    for step in range(num_steps):
        result = model(canvas)
        logits = result["logits"]

        mask_positions = (canvas[0] == mask_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            break

        gen_logits = logits[0, mask_positions] / temperature
        probs = torch.softmax(gen_logits, dim=-1)
        confidence, _ = probs.max(dim=-1)

        k = min(tokens_per_step, len(mask_positions))
        top_k_conf, top_k_idx = confidence.topk(k)

        for idx in top_k_idx:
            pos = mask_positions[idx]
            sampled_token = torch.multinomial(probs[idx], num_samples=1).item()
            canvas[0, pos] = sampled_token

        if verbose:
            filled = gen_len - (canvas[0] == mask_id).sum().item()
            print(f"Step {step+1}/{num_steps}: filled {filled}/{gen_len} tokens")

    generated_ids = canvas[0].cpu().tolist()
    return decode_tokens(tokenizer, tokenizer_type, generated_ids)


@torch.no_grad()
def generate_dialectic(
    model: DialecticCanvas,
    tokenizer,
    tokenizer_type: str,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    num_refine_steps: int = 2,
    device: str = "cuda",
    verbose: bool = False,
) -> tuple[str, dict]:
    """Generate text using Dialectic model with negotiation stats."""
    model.eval()

    prompt_ids = encode_prompt(tokenizer, tokenizer_type, prompt)
    prompt_tensor = torch.tensor([prompt_ids], device=device)

    # Use dialectic's built-in generation
    canvas = model.generate_dialectic(
        prompt_tensor,
        max_new_tokens=max_new_tokens,
        num_refine_steps=num_refine_steps,
        temperature=temperature,
    )

    generated_ids = canvas[0].cpu().tolist()
    generated_text = decode_tokens(tokenizer, tokenizer_type, generated_ids)

    # Get negotiation stats from a forward pass (for verbose mode)
    stats = {}
    if verbose:
        result = model(canvas, num_refine_steps=num_refine_steps)
        if "negotiation_stats" in result:
            all_stats = result["negotiation_stats"]
            total_backtracks = sum(s["total_backtracks"] for s in all_stats)
            avg_confidence = sum(
                sum(s["final_confidences"]) / len(s["final_confidences"])
                for s in all_stats
            ) / len(all_stats) if all_stats else 0
            stats = {
                "total_backtracks": total_backtracks,
                "avg_confidence": avg_confidence,
            }

    return generated_text, stats


def main():
    parser = argparse.ArgumentParser(description="Generate text from Shimmer checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--model", type=str, default=None, choices=["lira", "dialectic"],
                        help="Model type (auto-detected if not specified)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt to generate from (auto from dataset if not specified)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["tinystories", "everyday", "bitext", "arena"],
                        help="Dataset (for default prompt, auto-detected from checkpoint)")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Number of generation steps (LIRA: default=max_new_tokens)")
    parser.add_argument("--num_refine_steps", type=int, default=2,
                        help="Refinement steps per token (Dialectic only)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed generation info")
    parser.add_argument("--tokenizer", type=str, default="tokenizers/shimmer_blend_10000.model",
                        help="Directory containing custom tokenizers")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Detect or use specified model type
    if args.model is None:
        model_type = detect_model_type(checkpoint)
        print(f"Auto-detected model type: {model_type.upper()}")
    else:
        model_type = args.model
        print(f"Using specified model type: {model_type.upper()}")

    # Detect or use specified dataset (for default prompt)
    if args.dataset is None:
        dataset = checkpoint.get("dataset", "tinystories")
        print(f"Auto-detected dataset: {dataset}")
    else:
        dataset = args.dataset
        print(f"Using specified dataset: {dataset}")

    # Set prompt: use provided, or default for dataset
    if args.prompt is None:
        args.prompt = get_default_prompt(dataset)
        print(f"Using default prompt for {dataset}: '{args.prompt[:50]}...'" if len(args.prompt) > 50 else f"Using default prompt for {dataset}: '{args.prompt}'")

    # Get config from checkpoint
    config = checkpoint["config"]
    hybrid_info = ""
    if hasattr(config, 'hybrid_mode') and config.hybrid_mode:
        hybrid_info = f", hybrid={config.hybrid_mode}"
    print(f"Model config: hidden={config.hidden_size}, layers={config.num_layers}, "
          f"vocab={config.vocab_size}, mask_id={config.mask_token_id}{hybrid_info}")

    if model_type == "dialectic" and hasattr(config, 'max_backtracks'):
        print(f"Dialectic config: max_backtracks={config.max_backtracks}, "
              f"confidence_threshold={config.confidence_threshold}")

    # Load tokenizer
    tokenizer, tokenizer_type = load_tokenizer(config.vocab_size, args.tokenizer)

    # Create model and load weights
    model = create_model(config, model_type)

    # Handle state dict from compiled model (torch.compile adds _orig_mod. prefix)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Generate samples
    print(f"\n{'='*60}")
    print(f"Model: {model_type.upper()} | Dataset: {dataset}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Temperature: {args.temperature}, Max tokens: {args.max_new_tokens}")
    if model_type == "dialectic":
        print(f"Refine steps: {args.num_refine_steps}")
    print(f"{'='*60}\n")

    for i in range(args.num_samples):
        print(f"--- Sample {i+1}/{args.num_samples} ---")

        if model_type == "lira":
            generated = generate_lira(
                model=model,
                tokenizer=tokenizer,
                tokenizer_type=tokenizer_type,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_steps=args.num_steps,
                device=args.device,
                verbose=args.verbose,
            )
            print(generated)
        else:  # dialectic
            generated, stats = generate_dialectic(
                model=model,
                tokenizer=tokenizer,
                tokenizer_type=tokenizer_type,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_refine_steps=args.num_refine_steps,
                device=args.device,
                verbose=args.verbose,
            )
            print(generated)
            if args.verbose and stats:
                print(f"  [Backtracks: {stats.get('total_backtracks', 0)}, "
                      f"Avg confidence: {stats.get('avg_confidence', 0):.3f}]")

        print()

    print(f"{'='*60}")
    print("Generation complete!")


if __name__ == "__main__":
    main()
