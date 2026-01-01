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
from lira.gpt import GPTModel, GPTConfig


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
# Dataset-Specific Evaluation Presets
# ============================================================================

EVAL_PRESETS = {
    "tinystories": {
        "name": "TinyStories",
        "hf_dataset": "roneneldan/TinyStories",
        "hf_split": "validation",
        "text_field": "text",
        "tokenizer_pattern": "shimmer_tinystories_{vocab_size}.model",
        "prompts": [
            "Once upon a time",
            "There was a little girl named",
            "The sun was shining and",
            "One day, a small cat",
            "In a big forest, there lived",
            "The little boy wanted to",
            "A friendly dog named Max",
            "The princess looked at the",
        ],
        "challenge_prompts": [
            "Once upon a time, there was a little girl who loved to play in the garden. One sunny day, she",
            "The little boy was very sad because",
            "The brave knight drew his sword and",
            '"Hello!" said the rabbit. The bear replied',
        ],
        "coherence_tests": [
            ("The cat was hungry so it", ["ate", "food", "fish", "milk", "eat", "dinner", "fed"]),
            ("It was raining outside so they stayed", ["inside", "home", "house", "dry", "warm", "room", "indoors"]),
            ("The sun set and the sky turned", ["dark", "orange", "red", "pink", "night", "black", "purple", "colors"]),
            ("The little girl was happy because she got a new", ["toy", "doll", "dress", "gift", "present", "book", "friend"]),
            ("The dog ran fast and caught the", ["ball", "stick", "frisbee", "toy", "bone", "treat"]),
        ],
    },
    "agentic": {
        "name": "Agentic/Instruction",
        "hf_dataset": "teknium/OpenHermes-2.5",
        "hf_split": "train",
        "text_field": "conversations",  # Needs special handling
        "tokenizer_pattern": "shimmer_agentic_{vocab_size}.model",
        "prompts": [
            "Write a function that",
            "Explain how to",
            "What is the difference between",
            "How would you implement",
            "Can you help me understand",
            "Please provide a step-by-step",
            "The best approach to solve this",
            "Here is a solution for",
        ],
        "challenge_prompts": [
            "Write a Python function that takes a list of numbers and returns the sum of all even numbers. The function should",
            "Explain the concept of recursion in programming, including",
            "What are the key differences between REST and GraphQL APIs? First,",
            "How would you optimize a database query that is running slowly? Start by",
        ],
        "coherence_tests": [
            ("To sort a list in Python, you can use", ["sort", "sorted", "list", "function", "method", "ascending"]),
            ("The main difference between a list and a tuple is", ["mutable", "immutable", "change", "modify", "fixed"]),
            ("When debugging code, you should first", ["check", "print", "log", "test", "error", "trace", "breakpoint"]),
            ("A good API design should include", ["documentation", "endpoints", "authentication", "versioning", "REST"]),
            ("To handle errors in Python, use", ["try", "except", "catch", "exception", "error", "raise"]),
        ],
    },
    "code": {
        "name": "Code Generation",
        "hf_dataset": "bigcode/starcoderdata",
        "hf_split": "train",
        "text_field": "content",
        "tokenizer_pattern": "shimmer_code_{vocab_size}.model",
        "prompts": [
            "def ",
            "class ",
            "import ",
            "function ",
            "# TODO:",
            "async def ",
            "public static void",
            "const ",
        ],
        "challenge_prompts": [
            "def calculate_fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n    ",
            "class UserAuthentication:\n    def __init__(self):\n        ",
            "# Implement a binary search algorithm\ndef binary_search(arr, target):\n    ",
            "async def fetch_data(url):\n    \"\"\"Fetch data from URL with retry logic.\"\"\"\n    ",
        ],
        "coherence_tests": [
            ("def add(a, b):\n    return", ["a + b", "sum", "a+b", "result"]),
            ("if x > 0:\n    print", ["x", "positive", "(", "result"]),
            ("for i in range(10):\n    ", ["print", "i", "sum", "append", "result"]),
            ("try:\n    result = divide(a, b)\nexcept", ["ZeroDivisionError", "Exception", "Error", "except"]),
            ("import os\npath = os.path.", ["join", "exists", "dirname", "basename"]),
        ],
    },
    "chat": {
        "name": "Chat/Conversation",
        "hf_dataset": "lmsys/lmsys-chat-1m",
        "hf_split": "train",
        "text_field": "conversation",
        "tokenizer_pattern": "shimmer_chat_{vocab_size}.model",
        "prompts": [
            "Hello! How can I",
            "Thank you for your",
            "I understand your concern about",
            "That's a great question!",
            "Let me explain",
            "Based on what you said,",
            "I'd be happy to help",
            "Here's what I think:",
        ],
        "challenge_prompts": [
            "User: What's the weather like today?\nAssistant: I don't have access to real-time weather data, but",
            "User: Can you help me write a poem?\nAssistant: Of course! Let me",
            "User: I'm feeling stressed about work.\nAssistant: I understand how",
            "User: Explain quantum computing simply.\nAssistant: Quantum computing is",
        ],
        "coherence_tests": [
            ("Thank you for asking! I'd be happy to", ["help", "assist", "explain", "answer", "provide"]),
            ("I apologize, but I cannot", ["help", "assist", "provide", "access", "do"]),
            ("That's an interesting question. The answer is", ["that", "yes", "no", "it", "depends"]),
            ("Based on your description, I recommend", ["you", "that", "trying", "using", "the"]),
            ("Let me break this down into", ["steps", "parts", "sections", "simple", "points"]),
        ],
    },
    "general": {
        "name": "General Text",
        "hf_dataset": None,  # No specific dataset
        "hf_split": None,
        "text_field": None,
        "tokenizer_pattern": "shimmer_{vocab_size}.model",
        "prompts": [
            "The ",
            "In the ",
            "It was ",
            "There ",
            "When ",
            "After ",
            "Before ",
            "As ",
        ],
        "challenge_prompts": [
            "The most important thing to remember about this topic is that",
            "In conclusion, we can see that the evidence suggests",
            "There are several key factors to consider when",
            "When analyzing this situation, it becomes clear that",
        ],
        "coherence_tests": [
            ("The capital of France is", ["Paris", "city", "located", "known"]),
            ("Water freezes at", ["zero", "0", "degrees", "temperature", "cold"]),
            ("The sun rises in the", ["east", "morning", "sky", "horizon"]),
            ("Birds can fly because they have", ["wings", "feathers", "light", "hollow"]),
            ("To make a sandwich, first you need", ["bread", "ingredients", "slice", "spread"]),
        ],
    },
}

# Default preset (for backward compatibility)
DEFAULT_PRESET = "tinystories"


def get_eval_preset(name: str) -> dict:
    """Get evaluation preset by name."""
    if name not in EVAL_PRESETS:
        available = ", ".join(EVAL_PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return EVAL_PRESETS[name]


# Backward compatibility aliases
STORY_PROMPTS = EVAL_PRESETS["tinystories"]["prompts"]
CHALLENGE_PROMPTS = EVAL_PRESETS["tinystories"]["challenge_prompts"]
COHERENCE_PROMPTS = EVAL_PRESETS["tinystories"]["coherence_tests"]


# ============================================================================
# Tokenizer Loading
# ============================================================================

def load_tokenizer(vocab_size: int, tokenizer_dir: str = "tokenizers",
                   preset: str = DEFAULT_PRESET, tokenizer_path: str = None):
    """Load the appropriate tokenizer based on vocab_size and preset."""
    if vocab_size == 0 or vocab_size >= 50000:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, "gpt2"
    else:
        from lira.tokenizer import ShimmerTokenizer

        # Try explicit path first
        if tokenizer_path:
            tok_path = Path(tokenizer_path)
            if tok_path.exists():
                tokenizer = ShimmerTokenizer(str(tok_path))
                return tokenizer, "custom"

        # Try preset pattern
        eval_preset = get_eval_preset(preset)
        pattern = eval_preset["tokenizer_pattern"]
        tok_path = Path(tokenizer_dir) / pattern.format(vocab_size=vocab_size)

        if tok_path.exists():
            tokenizer = ShimmerTokenizer(str(tok_path))
            return tokenizer, "custom"

        # Fallback: try any matching tokenizer
        for fallback_pattern in [
            f"shimmer_*_{vocab_size}.model",
            f"shimmer_{vocab_size}.model",
            f"*_{vocab_size}.model",
        ]:
            matches = list(Path(tokenizer_dir).glob(fallback_pattern))
            if matches:
                tokenizer = ShimmerTokenizer(str(matches[0]))
                print(f"  Note: Using fallback tokenizer: {matches[0].name}")
                return tokenizer, "custom"

        raise FileNotFoundError(
            f"Tokenizer not found. Tried: {tok_path}\n"
            f"  Use --tokenizer_path to specify explicit path"
        )


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
    method: str = "topk", device: str = "cuda",
    model_type: str = "lira"
) -> tuple[str, dict]:
    """Generate a single sample with metadata."""
    model.eval()

    prompt_ids = encode(tokenizer, tokenizer_type, prompt)
    prompt_tensor = torch.tensor([prompt_ids], device=device)

    if model_type == "gpt":
        # GPT: autoregressive generation
        output = model.generate(
            prompt_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
        )
        generated_ids = output[0].cpu().tolist()
        generated_text = decode(tokenizer, tokenizer_type, generated_ids)
        metadata = {
            "iterations": max_tokens,
            "prompt_len": len(prompt_ids),
            "total_len": len(generated_ids),
        }
    else:
        # LIRA: masked generation
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
    prompts: list[str], device: str = "cuda",
    model_type: str = "lira"
) -> dict:
    """Analyze confidence head behavior (LIRA only)."""
    model.eval()

    # GPT doesn't have confidence head
    if model_type == "gpt":
        return {
            "mean_all": 0.0,
            "std_all": 0.0,
            "mean_filled": 0.0,
            "mean_masked": 0.0,
            "confidence_gap": 0.0,
            "note": "GPT has no confidence head",
        }

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
    model_type: str = "lira",
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

        # Generate multi-token continuation
        if model_type == "gpt":
            output = model.generate(
                prompt_tensor,
                max_new_tokens=num_gen_tokens,
                temperature=0.7,
                top_k=50,
            )
        else:
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
    model_type: str = "lira",
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
            if model_type == "gpt":
                output = model.generate(
                    prompt_tensor,
                    max_new_tokens=max_tokens,
                    temperature=0.9,
                    top_k=50,
                )
            else:
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
    model_type: str = "lira",
    preset: str = DEFAULT_PRESET,
) -> dict:
    """
    Calculate perplexity on validation data from the specified preset.

    For LIRA: Masked reconstruction perplexity
    For GPT: Next-token prediction perplexity (standard)
    """
    model.eval()

    eval_preset = get_eval_preset(preset)

    # Load validation dataset based on preset
    try:
        from datasets import load_dataset

        if eval_preset["hf_dataset"] is None:
            return {"error": f"No validation dataset for preset '{preset}'", "skipped": True}

        dataset = load_dataset(
            eval_preset["hf_dataset"],
            split=eval_preset["hf_split"],
            trust_remote_code=True
        )

        # Extract text based on field type
        text_field = eval_preset["text_field"]
        texts = []

        for i in range(min(num_samples, len(dataset))):
            item = dataset[i]

            if text_field == "conversations":
                # Handle conversation format (OpenHermes style)
                if isinstance(item.get("conversations"), list):
                    conv_text = " ".join(
                        msg.get("value", "") for msg in item["conversations"]
                        if isinstance(msg, dict)
                    )
                    texts.append(conv_text)
            elif text_field == "conversation":
                # Handle lmsys chat format
                if isinstance(item.get("conversation"), list):
                    conv_text = " ".join(
                        str(msg) for msg in item["conversation"]
                    )
                    texts.append(conv_text)
            elif text_field in item:
                texts.append(item[text_field])

        if not texts:
            return {"error": f"No valid texts extracted from {eval_preset['name']}", "skipped": True}

    except Exception as e:
        return {"error": f"Could not load {eval_preset['name']}: {e}"}

    total_loss = 0.0
    total_tokens = 0

    if model_type == "gpt":
        # GPT: Next-token prediction perplexity
        for text in texts:
            ids = encode(tokenizer, tokenizer_type, text)
            if len(ids) < 10 or len(ids) > 256:
                continue

            input_ids = torch.tensor([ids], device=device)

            # Forward pass with labels
            output = model(input_ids, labels=input_ids)
            loss = output["loss"]

            # Count valid tokens (excluding padding)
            num_tokens = len(ids) - 1  # -1 because of shift
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(min(avg_loss, 100))

        return {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "tokens_evaluated": total_tokens,
            "samples_evaluated": len(texts),
            "note": "Next-token prediction perplexity (standard GPT)",
        }
    else:
        # LIRA: Masked reconstruction perplexity
        mask_token = model.config.mask_token_id

        for text in texts:
            ids = encode(tokenizer, tokenizer_type, text)
            if len(ids) < 10 or len(ids) > 256:
                continue

            input_ids = torch.tensor([ids], device=device)
            seq_len = input_ids.size(1)

            # Random mask positions (like training)
            mask_positions = torch.rand(1, seq_len, device=device) < mask_ratio
            mask_positions[:, 0] = False
            mask_positions[:, -1] = False

            masked_input = input_ids.clone()
            masked_input[mask_positions] = mask_token

            output = model(masked_input)
            logits = output["logits"]

            masked_logits = logits[mask_positions]
            masked_labels = input_ids[mask_positions]

            if masked_logits.size(0) > 0:
                loss = F.cross_entropy(masked_logits, masked_labels, reduction='sum')
                total_loss += loss.item()
                total_tokens += masked_logits.size(0)

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(min(avg_loss, 100))

        return {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "mask_ratio": mask_ratio,
            "masked_tokens_evaluated": total_tokens,
            "samples_evaluated": len(texts),
            "note": "Masked reconstruction perplexity (LIRA)",
        }


@torch.no_grad()
def test_length_stress(
    model, tokenizer, tokenizer_type,
    prompt: str = "Once upon a time",
    lengths: list[int] = [50, 100, 200],
    device: str = "cuda",
    model_type: str = "lira",
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

        if model_type == "gpt":
            output = model.generate(
                prompt_tensor,
                max_new_tokens=length,
                temperature=0.8,
                top_k=50,
            )
            history = list(range(length))  # Placeholder for iterations
        else:
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

    # Load preset
    preset = args.preset
    eval_preset = get_eval_preset(preset)
    print(f"Evaluation Preset: {eval_preset['name']}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    config = checkpoint["config"]

    # Detect model type from config or checkpoint
    model_type = checkpoint.get("model_type", None)
    if model_type is None:
        # Infer from config type
        if isinstance(config, GPTConfig):
            model_type = "gpt"
        else:
            model_type = "lira"

    print(f"\nModel Type: {model_type.upper()}")
    print(f"Model Config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Vocab size: {config.vocab_size}")
    if model_type != "gpt":
        print(f"  Mask token ID: {config.mask_token_id}")

    # Load model based on type
    if model_type == "gpt":
        model = GPTModel(config)
    else:
        model = LatentCanvasModel(config)

    # Handle state dict from compiled model (torch.compile adds _orig_mod. prefix)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        print("  Stripped _orig_mod. prefix from compiled checkpoint")

    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Load tokenizer
    tokenizer, tokenizer_type = load_tokenizer(
        config.vocab_size, args.tokenizer_dir,
        preset=preset, tokenizer_path=args.tokenizer_path
    )
    print(f"  Tokenizer: {tokenizer_type} (vocab={config.vocab_size})")

    results = {"config": {
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "vocab_size": config.vocab_size,
        "parameters": param_count,
        "model_type": model_type,
        "preset": preset,
    }}

    # Get prompts from preset
    test_prompts = eval_preset["prompts"]
    challenge_prompts = eval_preset["challenge_prompts"]
    coherence_tests = eval_preset["coherence_tests"]

    # ========================================================================
    # Test 1: Basic Generation Quality
    # ========================================================================
    print(f"\n{'='*70}")
    print("TEST 1: Basic Generation Quality")
    print(f"{'='*70}\n")

    generations = []
    for prompt in test_prompts[:args.num_prompts]:
        print(f"Prompt: '{prompt}'")
        print("-" * 50)

        for temp in [0.7, 0.9]:
            text, meta = generate_sample(
                model, tokenizer, tokenizer_type, prompt,
                max_tokens=args.max_tokens,
                temperature=temp,
                method="topk",
                device=args.device,
                model_type=model_type
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
        test_prompts[:4], device=args.device,
        model_type=model_type
    )

    if model_type == "gpt":
        print("  [N/A] GPT has no confidence head")
    else:
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
        coherence_tests, device=args.device,
        model_type=model_type
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

        # GPT only has one generation method
        methods = ["autoregressive"] if model_type == "gpt" else ["topk", "remasking"]

        for method in methods:
            print(f"\nMethod: {method}")
            print("-" * 40)
            for _ in range(3):
                text, meta = generate_sample(
                    model, tokenizer, tokenizer_type, test_prompt,
                    max_tokens=40, temperature=0.8,
                    method=method, device=args.device,
                    model_type=model_type
                )
                print(f"  {text[:120]}...")

    # ========================================================================
    # Test 6: Challenge Prompts (if --full)
    # ========================================================================
    if args.full:
        print(f"\n{'='*70}")
        print("TEST 6: Challenge Prompts")
        print(f"{'='*70}\n")

        for prompt in challenge_prompts:
            print(f"Prompt: '{prompt[:50]}...'")
            text, meta = generate_sample(
                model, tokenizer, tokenizer_type, prompt,
                max_tokens=30, temperature=0.8,
                method="topk", device=args.device,
                model_type=model_type
            )
            print(f"Output: {text}")
            print()

    # ========================================================================
    # Test 7: Perplexity on Validation Data (if --full)
    # ========================================================================
    ppl_result = None
    if args.full:
        print(f"\n{'='*70}")
        print(f"TEST 7: Perplexity ({eval_preset['name']} Validation)")
        print(f"{'='*70}\n")

        ppl_result = test_perplexity(
            model, tokenizer, tokenizer_type,
            num_samples=100, device=args.device,
            model_type=model_type,
            preset=preset
        )

        if "error" in ppl_result:
            print(f"  Error: {ppl_result['error']}")
        else:
            print(f"  Perplexity: {ppl_result['perplexity']:.2f}")
            print(f"  Avg Loss: {ppl_result['avg_loss']:.4f}")
            if model_type == "gpt":
                print(f"  Tokens evaluated: {ppl_result['tokens_evaluated']:,}")
            else:
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
            test_prompts[:3],  # Use first 3 prompts
            num_generations=5,
            max_tokens=40,
            device=args.device,
            model_type=model_type
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
            device=args.device,
            model_type=model_type
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
    parser.add_argument("--preset", type=str, default=DEFAULT_PRESET,
                        choices=list(EVAL_PRESETS.keys()),
                        help=f"Evaluation preset (default: {DEFAULT_PRESET})")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (0, 1, etc.). Use -1 for CPU.")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizers",
                        help="Tokenizer directory")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Explicit tokenizer path (overrides preset pattern)")
    parser.add_argument("--num_prompts", type=int, default=4,
                        help="Number of prompts to test")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="Max tokens per generation")
    parser.add_argument("--full", action="store_true",
                        help="Run full evaluation suite")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    # Setup device
    if args.device == "cpu" or args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
        print(f"\nUsing CPU")
    else:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
        gpu_name = torch.cuda.get_device_name(args.gpu)
        gpu_mem = torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3
        print(f"\nUsing GPU {args.gpu}: {gpu_name} ({gpu_mem:.1f}GB)")

    run_evaluation(args)


if __name__ == "__main__":
    main()
