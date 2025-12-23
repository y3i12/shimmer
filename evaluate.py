#!/usr/bin/env python3
"""
Shimmer Model Evaluation Script

Comprehensive quality control for trained checkpoints:
- Generation quality tests (various prompts)
- Perplexity calculation
- Confidence head analysis
- Multiple generation methods comparison

Usage:
    python evaluate.py --checkpoint checkpoints/shimmer_v1_29M1_384_12_12_10235V_phase_4.pt
    python evaluate.py --checkpoint checkpoints/shimmer_v1_29M1_384_12_12_10235V_phase_4.pt --full
"""

import argparse
import torch
import torch.nn.functional as F
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np

# Optional: semantic similarity for coherence testing
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Note: sentence-transformers not installed. Semantic coherence check disabled.")
    print("      Install with: pip install sentence-transformers")

sys.path.insert(0, str(Path(__file__).parent))

from lira import LatentCanvasModel, LatentCanvasConfig


# ============================================================================
# Semantic Coherence Helper
# ============================================================================

class SemanticCoherenceChecker:
    """Check semantic coherence using sentence embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = None
        self.model_name = model_name

    def _load_model(self):
        if self.model is None and SEMANTIC_AVAILABLE:
            print(f"  Loading semantic model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
        return self.model is not None

    def check_coherence(
        self,
        prompt: str,
        generated: str,
        threshold: float = 0.5
    ) -> tuple[bool, float]:
        """
        Check if generated text is semantically coherent with prompt.

        Uses cosine similarity between:
        - prompt embedding
        - (prompt + generated) embedding

        Higher similarity = generated text flows naturally from prompt.

        Returns:
            (is_coherent, similarity_score)
        """
        if not self._load_model():
            return False, 0.0

        # Encode prompt and full continuation
        prompt_emb = self.model.encode(prompt, convert_to_tensor=True)
        full_emb = self.model.encode(prompt + " " + generated, convert_to_tensor=True)

        # Cosine similarity
        similarity = F.cosine_similarity(
            prompt_emb.unsqueeze(0),
            full_emb.unsqueeze(0)
        ).item()

        return similarity >= threshold, similarity

    def check_topic_relevance(
        self,
        generated: str,
        expected_topics: list[str],
        threshold: float = 0.3
    ) -> tuple[bool, float, list[str]]:
        """
        Check if generated text is semantically related to expected topics.

        Returns:
            (is_relevant, max_similarity, matched_topics)
        """
        if not self._load_model():
            return False, 0.0, []

        gen_emb = self.model.encode(generated, convert_to_tensor=True)

        matched = []
        max_sim = 0.0

        for topic in expected_topics:
            topic_emb = self.model.encode(topic, convert_to_tensor=True)
            sim = F.cosine_similarity(
                gen_emb.unsqueeze(0),
                topic_emb.unsqueeze(0)
            ).item()

            if sim > max_sim:
                max_sim = sim
            if sim >= threshold:
                matched.append((topic, sim))

        # Sort by similarity
        matched.sort(key=lambda x: x[1], reverse=True)
        matched_topics = [t for t, s in matched[:3]]  # Top 3

        return len(matched) > 0, max_sim, matched_topics


# Global instance (lazy loaded)
_semantic_checker = None

def get_semantic_checker() -> SemanticCoherenceChecker:
    global _semantic_checker
    if _semantic_checker is None:
        _semantic_checker = SemanticCoherenceChecker()
    return _semantic_checker


# ============================================================================
# Test Prompts for Story Generation
# ============================================================================

STORY_PROMPTS = [
    "Once upon a time",
    "There was a little girl named",
    "The sun was shining and",
    "One day, a small cat",
    "In a big forest, there lived",
    "The little boy wanted to",
    "A friendly dog named Max",
    "The princess looked at the",
]

CHALLENGE_PROMPTS = [
    # Longer context
    "Once upon a time, there was a little girl who loved to play in the garden. One sunny day, she",
    # Emotional content
    "The little boy was very sad because",
    # Action sequence
    "The brave knight drew his sword and",
    # Dialogue setup
    '"Hello!" said the rabbit. The bear replied',
]

COHERENCE_PROMPTS = [
    # Should continue logically - expected words should appear in multi-token continuation
    ("The cat was hungry so it", ["ate", "food", "fish", "milk", "eat", "dinner", "fed"]),
    ("It was raining outside so they stayed", ["inside", "home", "house", "dry", "warm", "room", "indoors"]),
    ("The sun set and the sky turned", ["dark", "orange", "red", "pink", "night", "black", "purple", "colors"]),
    ("The little girl was happy because she got a new", ["toy", "doll", "dress", "gift", "present", "book", "friend"]),
    ("The dog ran fast and caught the", ["ball", "stick", "frisbee", "toy", "bone", "treat"]),
]


# ============================================================================
# Tokenizer Loading
# ============================================================================

def load_tokenizer(vocab_size: int, tokenizer_dir: str = "tokenizers"):
    """Load the appropriate tokenizer based on vocab_size."""
    if vocab_size == 0 or vocab_size >= 50000:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, "gpt2"
    else:
        from lira.tokenizer import ShimmerTokenizer
        tokenizer_path = Path(tokenizer_dir) / f"shimmer_blend_{vocab_size}.model"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        tokenizer = ShimmerTokenizer(str(tokenizer_path))
        return tokenizer, "custom"


def encode(tokenizer, tokenizer_type: str, text: str) -> list[int]:
    """Encode text."""
    if tokenizer_type == "gpt2":
        return tokenizer.encode(text, add_special_tokens=False)
    else:
        return tokenizer.encode(text, add_bos=False, add_eos=False)


def decode(tokenizer, tokenizer_type: str, ids: list[int]) -> str:
    """Decode token IDs."""
    return tokenizer.decode(ids)


# ============================================================================
# Evaluation Functions
# ============================================================================

@torch.no_grad()
def generate_sample(
    model, tokenizer, tokenizer_type, prompt: str,
    max_tokens: int = 50, temperature: float = 0.8,
    method: str = "topk", device: str = "cuda"
) -> tuple[str, dict]:
    """Generate a single sample with metadata."""
    model.eval()

    prompt_ids = encode(tokenizer, tokenizer_type, prompt)
    prompt_tensor = torch.tensor([prompt_ids], device=device)

    if method == "topk":
        output, history = model.generate_topk(
            prompt_tensor, max_tokens,
            num_steps=int(max_tokens*2),
            temperature=temperature
        )
    elif method == "remasking":
        output, history = model.generate_with_remasking(
            prompt_tensor, max_tokens,
            num_steps=int(max_tokens*2),
            temperature=temperature,
            remask_ratio=0.2,
            min_confident_ratio=0.3
        )
    else:  # default confidence-based
        output, history = model.generate(
            prompt_tensor, max_tokens,
            num_iterations=int(max_tokens*2),
            temperature=temperature
        )

    generated_ids = output[0].cpu().tolist()
    generated_text = decode(tokenizer, tokenizer_type, generated_ids)

    metadata = {
        "iterations": len(history),
        "prompt_len": len(prompt_ids),
        "total_len": len(generated_ids),
    }

    return generated_text, metadata


@torch.no_grad()
def calculate_perplexity(
    model, tokenizer, tokenizer_type,
    texts: list[str], device: str = "cuda"
) -> float:
    """Calculate perplexity on a set of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        ids = encode(tokenizer, tokenizer_type, text)
        if len(ids) < 5:
            continue

        # Create input with some masking (like training)
        input_ids = torch.tensor([ids], device=device)

        # Forward pass
        output = model(input_ids)
        logits = output["logits"]

        # Calculate cross-entropy loss
        # Shift for next-token prediction style
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum'
        )

        total_loss += loss.item()
        total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)

    return perplexity


@torch.no_grad()
def analyze_confidence(
    model, tokenizer, tokenizer_type,
    prompts: list[str], device: str = "cuda"
) -> dict:
    """Analyze confidence head behavior."""
    model.eval()

    all_confidence = []
    mask_confidence = []
    filled_confidence = []

    mask_token = model.config.mask_token_id

    for prompt in prompts:
        ids = encode(tokenizer, tokenizer_type, prompt)
        # Create a mix of filled and masked
        input_ids = ids + [mask_token] * 20
        input_tensor = torch.tensor([input_ids], device=device)

        output = model(input_tensor, return_confidence=True)
        conf = output["confidence"][0].cpu().numpy()

        all_confidence.extend(conf.tolist())
        filled_confidence.extend(conf[:len(ids)].tolist())
        mask_confidence.extend(conf[len(ids):].tolist())

    return {
        "mean_all": float(np.mean(all_confidence)),
        "std_all": float(np.std(all_confidence)),
        "mean_filled": float(np.mean(filled_confidence)),
        "mean_masked": float(np.mean(mask_confidence)),
        "confidence_gap": float(np.mean(filled_confidence) - np.mean(mask_confidence)),
    }


@torch.no_grad()
def test_coherence(
    model, tokenizer, tokenizer_type,
    coherence_tests: list[tuple],
    device: str = "cuda",
    num_gen_tokens: int = 10,
    use_semantic: bool = True,
) -> dict:
    """
    Test logical coherence of continuations using multi-token generation.

    LIRA is trained for parallel multi-mask prediction, not single-token prediction.
    This test generates multiple tokens and checks coherence via:
    1. Keyword matching (expected words appear in continuation)
    2. Semantic similarity (continuation flows naturally from prompt)

    A continuation passes if EITHER check succeeds.
    """
    model.eval()

    # Initialize semantic checker if available
    semantic_checker = None
    if use_semantic and SEMANTIC_AVAILABLE:
        semantic_checker = get_semantic_checker()

    results = []

    for prompt, expected_tokens in coherence_tests:
        prompt_ids = encode(tokenizer, tokenizer_type, prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=device)

        # Generate multi-token continuation (LIRA's strength)
        output, _ = model.generate_topk(
            prompt_tensor,
            gen_length=num_gen_tokens,
            num_steps=num_gen_tokens * 2,
            temperature=0.7,
        )

        # Decode the generated continuation (excluding prompt)
        generated_ids = output[0, len(prompt_ids):].cpu().tolist()
        generated_text = decode(tokenizer, tokenizer_type, generated_ids).lower()

        # === Check 1: Keyword matching ===
        expected_lower = [t.lower() for t in expected_tokens]
        found_words = [w for w in expected_lower if w in generated_text]
        keyword_pass = len(found_words) > 0

        # === Check 2: Semantic coherence ===
        semantic_pass = False
        semantic_score = 0.0
        semantic_topics = []

        if semantic_checker:
            # Check if continuation flows from prompt
            semantic_pass, semantic_score = semantic_checker.check_coherence(
                prompt, generated_text, threshold=0.7
            )

            # Also check topic relevance if keyword failed
            if not keyword_pass:
                topic_pass, _, semantic_topics = semantic_checker.check_topic_relevance(
                    generated_text, expected_tokens, threshold=0.25
                )
                if topic_pass:
                    semantic_pass = True

        # Pass if EITHER check succeeds
        coherent = keyword_pass or semantic_pass

        results.append({
            "prompt": prompt,
            "expected_any": expected_tokens,
            "generated": generated_text.strip()[:60],
            "found_words": found_words,
            "keyword_pass": keyword_pass,
            "semantic_pass": semantic_pass,
            "semantic_score": round(semantic_score, 3),
            "semantic_topics": semantic_topics,
            "coherent": coherent,
        })

    coherence_rate = sum(1 for r in results if r["coherent"]) / len(results)
    keyword_rate = sum(1 for r in results if r["keyword_pass"]) / len(results)
    semantic_rate = sum(1 for r in results if r["semantic_pass"]) / len(results)

    return {
        "coherence_rate": coherence_rate,
        "keyword_rate": keyword_rate,
        "semantic_rate": semantic_rate,
        "semantic_available": semantic_checker is not None,
        "details": results,
    }


def test_diversity(
    model, tokenizer, tokenizer_type,
    prompts: list[str],
    num_generations: int = 5,
    max_tokens: int = 50,
    device: str = "cuda",
) -> dict:
    """
    Test generation diversity - are outputs varied or repetitive?

    Generates multiple continuations for each prompt and measures
    how different they are from each other.
    """
    model.eval()

    all_diversities = []
    prompt_results = []

    for prompt in prompts:
        prompt_ids = encode(tokenizer, tokenizer_type, prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=device)

        generations = []
        for _ in range(num_generations):
            output, _ = model.generate_topk(
                prompt_tensor,
                gen_length=max_tokens,
                num_steps=max_tokens * 2,
                temperature=0.9,  # Higher temp for diversity
            )
            gen_ids = output[0, len(prompt_ids):].cpu().tolist()
            gen_text = decode(tokenizer, tokenizer_type, gen_ids).lower()
            generations.append(gen_text)

        # Measure diversity: unique n-grams across all generations
        all_words = []
        all_bigrams = []
        all_trigrams = []

        for gen in generations:
            words = gen.split()
            all_words.extend(words)
            all_bigrams.extend([f"{words[i]} {words[i+1]}" for i in range(len(words)-1)])
            all_trigrams.extend([f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)])

        # Diversity = unique / total (higher = more diverse)
        word_diversity = len(set(all_words)) / len(all_words) if all_words else 0
        bigram_diversity = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0
        trigram_diversity = len(set(all_trigrams)) / len(all_trigrams) if all_trigrams else 0

        # Self-BLEU approximation: how similar are generations to each other?
        # Lower = more diverse
        unique_gens = len(set(generations))
        self_similarity = 1 - (unique_gens / num_generations)

        prompt_results.append({
            "prompt": prompt[:30],
            "word_diversity": word_diversity,
            "bigram_diversity": bigram_diversity,
            "trigram_diversity": trigram_diversity,
            "unique_generations": unique_gens,
            "self_similarity": self_similarity,
        })

        all_diversities.append(bigram_diversity)

    return {
        "avg_word_diversity": np.mean([r["word_diversity"] for r in prompt_results]),
        "avg_bigram_diversity": np.mean([r["bigram_diversity"] for r in prompt_results]),
        "avg_trigram_diversity": np.mean([r["trigram_diversity"] for r in prompt_results]),
        "avg_self_similarity": np.mean([r["self_similarity"] for r in prompt_results]),
        "details": prompt_results,
    }


@torch.no_grad()
def test_perplexity(
    model, tokenizer, tokenizer_type,
    num_samples: int = 100,
    mask_ratio: float = 0.3,
    device: str = "cuda",
) -> dict:
    """
    Calculate MASKED RECONSTRUCTION perplexity on TinyStories validation data.

    Unlike autoregressive perplexity, this measures how well LIRA
    reconstructs randomly masked tokens - matching its training objective.

    Lower perplexity = better masked language modeling.
    """
    model.eval()

    # Load TinyStories validation
    try:
        from datasets import load_dataset
        dataset = load_dataset("roneneldan/TinyStories", split="validation")
        texts = [dataset[i]["text"] for i in range(min(num_samples, len(dataset)))]
    except Exception as e:
        return {"error": f"Could not load TinyStories: {e}"}

    mask_token = model.config.mask_token_id
    total_loss = 0.0
    total_masked_tokens = 0

    for text in texts:
        ids = encode(tokenizer, tokenizer_type, text)
        if len(ids) < 10 or len(ids) > 256:
            continue

        # Create masked input (LIRA's actual task)
        input_ids = torch.tensor([ids], device=device)
        seq_len = input_ids.size(1)

        # Random mask positions (like training)
        mask_positions = torch.rand(1, seq_len, device=device) < mask_ratio
        # Don't mask first/last tokens for stability
        mask_positions[:, 0] = False
        mask_positions[:, -1] = False

        # Apply mask
        masked_input = input_ids.clone()
        masked_input[mask_positions] = mask_token

        # Forward pass
        output = model(masked_input)
        logits = output["logits"]

        # Only measure loss on masked positions
        masked_logits = logits[mask_positions]  # [num_masked, vocab]
        masked_labels = input_ids[mask_positions]  # [num_masked]

        if masked_logits.size(0) > 0:
            loss = F.cross_entropy(
                masked_logits,
                masked_labels,
                reduction='sum'
            )
            total_loss += loss.item()
            total_masked_tokens += masked_logits.size(0)

    avg_loss = total_loss / total_masked_tokens if total_masked_tokens > 0 else float('inf')
    perplexity = np.exp(min(avg_loss, 100))  # Cap to avoid overflow

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "mask_ratio": mask_ratio,
        "masked_tokens_evaluated": total_masked_tokens,
        "samples_evaluated": len(texts),
        "note": "Masked reconstruction perplexity (appropriate for LIRA)",
    }


@torch.no_grad()
def test_length_stress(
    model, tokenizer, tokenizer_type,
    prompt: str = "Once upon a time",
    lengths: list[int] = [50, 100, 200],
    device: str = "cuda",
) -> dict:
    """
    Test generation quality at different lengths.

    Checks if quality degrades with longer generation.
    """
    model.eval()

    prompt_ids = encode(tokenizer, tokenizer_type, prompt)
    prompt_tensor = torch.tensor([prompt_ids], device=device)

    results = []

    for length in lengths:
        import time
        start = time.time()

        output, history = model.generate_topk(
            prompt_tensor,
            gen_length=length,
            num_steps=length * 2,
            temperature=0.8,
        )

        elapsed = time.time() - start
        tokens_per_sec = length / elapsed

        gen_ids = output[0, len(prompt_ids):].cpu().tolist()
        gen_text = decode(tokenizer, tokenizer_type, gen_ids)

        # Analyze quality
        words = gen_text.lower().split()

        # Repetition at end (sign of degradation)
        if len(words) >= 10:
            last_10 = words[-10:]
            unique_last_10 = len(set(last_10))
            end_diversity = unique_last_10 / 10
        else:
            end_diversity = 1.0

        # Bigram repetition
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_rep = 1 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0

        results.append({
            "length": length,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "iterations": len(history),
            "end_diversity": round(end_diversity, 3),
            "bigram_repetition": round(bigram_rep, 3),
            "sample": gen_text[:100] + "..." if len(gen_text) > 100 else gen_text,
        })

    return {
        "results": results,
        "quality_degrades": results[-1]["end_diversity"] < results[0]["end_diversity"] - 0.2,
    }


def test_repetition(generated_texts: list[str]) -> dict:
    """Analyze repetition patterns in generated text."""
    results = []

    for text in generated_texts:
        words = text.lower().split()
        if len(words) < 5:
            continue

        # Count consecutive repetitions
        consecutive_reps = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                consecutive_reps += 1

        # Count bigram repetitions
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        unique_bigrams = set(bigrams)
        bigram_rep_rate = 1 - (len(unique_bigrams) / len(bigrams)) if bigrams else 0

        # Count trigram repetitions
        if len(words) >= 3:
            trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
            unique_trigrams = set(trigrams)
            trigram_rep_rate = 1 - (len(unique_trigrams) / len(trigrams)) if trigrams else 0
        else:
            trigram_rep_rate = 0

        results.append({
            "consecutive_reps": consecutive_reps,
            "bigram_rep_rate": bigram_rep_rate,
            "trigram_rep_rate": trigram_rep_rate,
            "word_count": len(words),
        })

    if not results:
        return {"error": "No valid texts to analyze"}

    return {
        "avg_consecutive_reps": np.mean([r["consecutive_reps"] for r in results]),
        "avg_bigram_rep_rate": np.mean([r["bigram_rep_rate"] for r in results]),
        "avg_trigram_rep_rate": np.mean([r["trigram_rep_rate"] for r in results]),
        "samples_analyzed": len(results),
    }


# ============================================================================
# Main Evaluation
# ============================================================================

def run_evaluation(args):
    """Run full evaluation suite."""
    print(f"\n{'='*70}")
    print("SHIMMER MODEL EVALUATION")
    print(f"{'='*70}\n")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    config = checkpoint["config"]
    print(f"\nModel Config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Mask token ID: {config.mask_token_id}")

    # Load model
    model = LatentCanvasModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Load tokenizer
    tokenizer, tokenizer_type = load_tokenizer(config.vocab_size, args.tokenizer_dir)
    print(f"  Tokenizer: {tokenizer_type} (vocab={config.vocab_size})")

    results = {"config": {
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "vocab_size": config.vocab_size,
        "parameters": param_count,
    }}

    # ========================================================================
    # Test 1: Basic Generation Quality
    # ========================================================================
    print(f"\n{'='*70}")
    print("TEST 1: Basic Generation Quality")
    print(f"{'='*70}\n")

    generations = []
    for prompt in STORY_PROMPTS[:args.num_prompts]:
        print(f"Prompt: '{prompt}'")
        print("-" * 50)

        for temp in [0.7, 0.9]:
            text, meta = generate_sample(
                model, tokenizer, tokenizer_type, prompt,
                max_tokens=args.max_tokens,
                temperature=temp,
                method="topk",
                device=args.device
            )
            generations.append(text)
            print(f"  [T={temp}] {text[:150]}...")
            print(f"           (iters={meta['iterations']})")
        print()

    results["generations"] = generations[:10]  # Save first 10

    # ========================================================================
    # Test 2: Repetition Analysis
    # ========================================================================
    print(f"\n{'='*70}")
    print("TEST 2: Repetition Analysis")
    print(f"{'='*70}\n")

    rep_analysis = test_repetition(generations)
    print(f"  Avg consecutive repetitions: {rep_analysis.get('avg_consecutive_reps', 'N/A'):.2f}")
    print(f"  Avg bigram repetition rate: {rep_analysis.get('avg_bigram_rep_rate', 'N/A'):.2%}")
    print(f"  Avg trigram repetition rate: {rep_analysis.get('avg_trigram_rep_rate', 'N/A'):.2%}")

    results["repetition"] = rep_analysis

    # ========================================================================
    # Test 3: Confidence Head Analysis
    # ========================================================================
    print(f"\n{'='*70}")
    print("TEST 3: Confidence Head Analysis")
    print(f"{'='*70}\n")

    conf_analysis = analyze_confidence(
        model, tokenizer, tokenizer_type,
        STORY_PROMPTS[:4], device=args.device
    )
    print(f"  Mean confidence (all): {conf_analysis['mean_all']:.4f}")
    print(f"  Mean confidence (filled tokens): {conf_analysis['mean_filled']:.4f}")
    print(f"  Mean confidence (masked tokens): {conf_analysis['mean_masked']:.4f}")
    print(f"  Confidence gap (filled - masked): {conf_analysis['confidence_gap']:.4f}")

    # Good model should have higher confidence on filled than masked
    if conf_analysis['confidence_gap'] > 0.1:
        print("  [GOOD] Model distinguishes filled from masked positions")
    else:
        print("  [WARN] Low confidence differentiation")

    results["confidence"] = conf_analysis

    # ========================================================================
    # Test 4: Coherence Testing
    # ========================================================================
    print(f"\n{'='*70}")
    print("TEST 4: Logical Coherence")
    print(f"{'='*70}\n")

    coherence = test_coherence(
        model, tokenizer, tokenizer_type,
        COHERENCE_PROMPTS, device=args.device
    )
    print(f"  Combined Coherence: {coherence['coherence_rate']:.1%}")
    print(f"    - Keyword match:  {coherence['keyword_rate']:.1%}")
    if coherence['semantic_available']:
        print(f"    - Semantic match: {coherence['semantic_rate']:.1%}")
    print(f"  (Multi-token generation + semantic similarity)\n")

    for detail in coherence['details']:
        status = "✓ PASS" if detail['coherent'] else "✗ FAIL"
        print(f"  [{status}] '{detail['prompt']}...'")
        print(f"           Generated: \"{detail['generated']}...\"")

        # Show what matched
        reasons = []
        if detail['keyword_pass']:
            reasons.append(f"keywords: {detail['found_words']}")
        if detail['semantic_pass']:
            if detail['semantic_topics']:
                reasons.append(f"semantic topics: {detail['semantic_topics']}")
            else:
                reasons.append(f"semantic flow: {detail['semantic_score']:.2f}")

        if reasons:
            print(f"           Matched: {', '.join(reasons)}")
        else:
            print(f"           Expected any of: {detail['expected_any'][:5]}...")

    results["coherence"] = {
        "rate": coherence['coherence_rate'],
        "details": coherence['details'],
    }

    # ========================================================================
    # Test 5: Generation Methods Comparison (if --full)
    # ========================================================================
    if args.full:
        print(f"\n{'='*70}")
        print("TEST 5: Generation Methods Comparison")
        print(f"{'='*70}\n")

        test_prompt = "Once upon a time, there was"

        for method in ["topk", "remasking"]:
            print(f"\nMethod: {method}")
            print("-" * 40)
            for _ in range(3):
                text, meta = generate_sample(
                    model, tokenizer, tokenizer_type, test_prompt,
                    max_tokens=40, temperature=0.8,
                    method=method, device=args.device
                )
                print(f"  {text[:120]}...")

    # ========================================================================
    # Test 6: Challenge Prompts (if --full)
    # ========================================================================
    if args.full:
        print(f"\n{'='*70}")
        print("TEST 6: Challenge Prompts")
        print(f"{'='*70}\n")

        for prompt in CHALLENGE_PROMPTS:
            print(f"Prompt: '{prompt[:50]}...'")
            text, meta = generate_sample(
                model, tokenizer, tokenizer_type, prompt,
                max_tokens=30, temperature=0.8,
                method="topk", device=args.device
            )
            print(f"Output: {text}")
            print()

    # ========================================================================
    # Test 7: Perplexity on Validation Data (if --full)
    # ========================================================================
    ppl_result = None
    if args.full:
        print(f"\n{'='*70}")
        print("TEST 7: Perplexity (TinyStories Validation)")
        print(f"{'='*70}\n")

        ppl_result = test_perplexity(
            model, tokenizer, tokenizer_type,
            num_samples=100, device=args.device
        )

        if "error" in ppl_result:
            print(f"  Error: {ppl_result['error']}")
        else:
            print(f"  Perplexity: {ppl_result['perplexity']:.2f}")
            print(f"  Avg Loss: {ppl_result['avg_loss']:.4f}")
            print(f"  Masked tokens evaluated: {ppl_result['masked_tokens_evaluated']:,}")
            print(f"  (Mask ratio: {ppl_result['mask_ratio']:.0%})")

            if ppl_result['perplexity'] < 50:
                print("  [GOOD] Low perplexity indicates good language modeling")
            elif ppl_result['perplexity'] < 100:
                print("  [OK] Moderate perplexity")
            else:
                print("  [WARN] High perplexity")

        results["perplexity"] = ppl_result

    # ========================================================================
    # Test 8: Generation Diversity (if --full)
    # ========================================================================
    diversity_result = None
    if args.full:
        print(f"\n{'='*70}")
        print("TEST 8: Generation Diversity")
        print(f"{'='*70}\n")

        diversity_result = test_diversity(
            model, tokenizer, tokenizer_type,
            STORY_PROMPTS[:3],  # Use first 3 prompts
            num_generations=5,
            max_tokens=40,
            device=args.device
        )

        print(f"  Word diversity:   {diversity_result['avg_word_diversity']:.1%}")
        print(f"  Bigram diversity: {diversity_result['avg_bigram_diversity']:.1%}")
        print(f"  Trigram diversity: {diversity_result['avg_trigram_diversity']:.1%}")
        print(f"  Self-similarity:  {diversity_result['avg_self_similarity']:.1%}")

        if diversity_result['avg_bigram_diversity'] > 0.7:
            print("  [GOOD] High diversity - generations are varied")
        elif diversity_result['avg_bigram_diversity'] > 0.5:
            print("  [OK] Moderate diversity")
        else:
            print("  [WARN] Low diversity - generations are repetitive")

        results["diversity"] = diversity_result

    # ========================================================================
    # Test 9: Length Stress Test (if --full)
    # ========================================================================
    length_result = None
    if args.full:
        print(f"\n{'='*70}")
        print("TEST 9: Length Stress Test")
        print(f"{'='*70}\n")

        length_result = test_length_stress(
            model, tokenizer, tokenizer_type,
            prompt="Once upon a time",
            lengths=[50, 100, 200],
            device=args.device
        )

        for r in length_result['results']:
            status = "✓" if r['end_diversity'] > 0.5 else "⚠"
            print(f"  [{status}] {r['length']:3d} tokens: {r['tokens_per_sec']:5.1f} tok/s, "
                  f"end_div={r['end_diversity']:.2f}, rep={r['bigram_repetition']:.2f}")

        if length_result['quality_degrades']:
            print("\n  [WARN] Quality degrades with length")
        else:
            print("\n  [GOOD] Quality maintained across lengths")

        results["length_stress"] = length_result

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}\n")

    print(f"Model: {param_count:,} parameters")
    print(f"Coherence Rate: {coherence['coherence_rate']:.1%}")
    print(f"Confidence Gap: {conf_analysis['confidence_gap']:.4f}")
    print(f"Bigram Repetition: {rep_analysis.get('avg_bigram_rep_rate', 0):.1%}")

    if ppl_result and "perplexity" in ppl_result:
        print(f"Perplexity: {ppl_result['perplexity']:.2f}")
    if diversity_result:
        print(f"Diversity: {diversity_result['avg_bigram_diversity']:.1%}")

    # Overall quality assessment
    quality_score = 0
    if coherence['coherence_rate'] >= 0.6:
        quality_score += 1
    if conf_analysis['confidence_gap'] > 0.05:
        quality_score += 1
    if rep_analysis.get('avg_bigram_rep_rate', 1) < 0.3:
        quality_score += 1

    quality_labels = ["POOR", "FAIR", "GOOD", "EXCELLENT"]
    print(f"\nOverall Quality: {quality_labels[quality_score]} ({quality_score}/3)")

    results["summary"] = {
        "quality_score": quality_score,
        "quality_label": quality_labels[quality_score],
    }

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    print(f"\n{'='*70}")
    print("Evaluation complete!")
    print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Shimmer checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizers",
                        help="Tokenizer directory")
    parser.add_argument("--num_prompts", type=int, default=4,
                        help="Number of prompts to test")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="Max tokens per generation")
    parser.add_argument("--full", action="store_true",
                        help="Run full evaluation suite")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    run_evaluation(args)


if __name__ == "__main__":
    main()
