# LIRA Architecture
**Latent Iterative Refinement Architecture**

---

## Overview

LIRA is a language model architecture that generates text through iterative refinement rather than autoregressive token-by-token prediction. It combines ideas from:

- **Masked Language Models** (BERT) - bidirectional context
- **Diffusion Models** (LLaDA) - iterative denoising
- **Recursive Reasoning** (TRM) - same network applied multiple times

---

## Core Principles

### 1. Latent Canvas

Instead of generating tokens sequentially, LIRA operates on a **canvas** of latent representations:

```
Input tokens → Embed → [z₀, z₁, z₂, ..., zₙ] → Refine → Decode → Output tokens
                              ↑
                        Latent Canvas
```

Each position has a continuous representation that gets refined iteratively.

### 2. Iterative Refinement

The same `RefineBlock` is applied K times:

```python
for k in range(K):
    z = refine_block(z)  # Same weights each time
```

This is inspired by TinyRecursiveModels: **depth through iteration, not layer count**.

### 3. Bidirectional Attention

Unlike autoregressive models (causal masking), LIRA uses full bidirectional attention:

```
Autoregressive:  [a] [b] [c] [?]  → can only see a, b, c
LIRA:            [a] [?] [c] [d]  → sees everything, predicts [?]
```

This enables parallel prediction and global coherence.

### 4. Confidence-Guided Generation

The model learns to predict its own uncertainty:

```python
logits, confidence = model(canvas)
# confidence[i] ∈ [0, 1] = how certain about position i
```

High-confidence tokens crystallize first; low-confidence tokens may be reconsidered.

---

## Components

### RefineBlock

The core computation unit, applied iteratively:

```
┌─────────────────────────────────────┐
│            RefineBlock              │
├─────────────────────────────────────┤
│  Input: z [B, L, D]                 │
│                                     │
│  for layer in layers:               │
│      z = z + Attention(RMSNorm(z))  │
│      z = z + SwiGLU(RMSNorm(z))     │
│                                     │
│  Output: RMSNorm(z)                 │
└─────────────────────────────────────┘
```

**Design choices:**
- Pre-norm (RMSNorm before each sublayer) - stable training
- SwiGLU activation - better than ReLU/GELU
- No causal mask - full bidirectional attention
- RoPE positional encoding - handles variable lengths

### LatentCanvasModel

The full model wrapping RefineBlock:

```
┌─────────────────────────────────────────────────────┐
│              LatentCanvasModel                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  embed_tokens: Embedding(vocab_size + 1, hidden)    │
│  mask_embed: Parameter(hidden)  # learnable         │
│                                                     │
│  refine_block: RefineBlock(...)                     │
│                                                     │
│  token_head: Linear(hidden, vocab_size)             │
│  confidence_head: Linear(hidden, 1)                 │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Forward pass:**
```python
def forward(input_ids, num_refine_steps=K):
    # 1. Embed tokens to latent space
    z = embed_tokens(input_ids)
    z[mask_positions] = mask_embed  # Learnable mask embedding

    # 2. Iterative refinement
    for k in range(num_refine_steps):
        z = refine_block(z)

    # 3. Decode to predictions
    logits = token_head(z)
    confidence = sigmoid(confidence_head(z))

    return logits, confidence
```

---

## Training

### Loss Function

**Masked reconstruction with variable corruption (LLaDA-style):**

```python
# Sample corruption ratio per batch
t ~ Uniform(0.1, 1.0)

# Mask t% of tokens
corrupted = mask_tokens(input, ratio=t)

# Predict original tokens
logits = model(corrupted)

# Loss weighted by difficulty
loss = CrossEntropy(logits[masked], targets[masked]) / t
```

The `1/t` weighting gives more credit for harder reconstructions.

### Phase 4: Confidence Supervision

```python
# Target: was the prediction correct?
correct = (argmax(logits) == targets).float()

# Confidence should predict correctness
conf_loss = BCE(confidence[masked], correct[masked])

total_loss = token_loss + 0.1 * conf_loss
```

---

## Generation

### Top-K Selection (Default)

```python
def generate_topk(prompt, gen_length, num_steps):
    canvas = [prompt..., MASK, MASK, ..., MASK]
    tokens_per_step = gen_length // num_steps

    for step in range(num_steps):
        logits = model(canvas)
        predictions = sample(logits, temperature)
        confidence = softmax(logits).max(dim=-1)

        # Fill top-k most confident masked positions
        top_k = select_topk_masked(confidence, k=tokens_per_step)
        canvas[top_k] = predictions[top_k]

    return canvas
```

### With Re-masking (Experimental)

```python
def generate_with_remasking(prompt, gen_length, num_steps):
    canvas = [prompt..., MASK, MASK, ..., MASK]

    for step in range(num_steps):
        logits, confidence = model(canvas)
        predictions = sample(logits, temperature)

        # Fill high-confidence masked positions
        fill_mask = is_masked & (confidence > threshold)
        canvas[fill_mask] = predictions[fill_mask]

        # Re-mask low-confidence filled positions (the key innovation)
        remask = is_filled & (confidence < remask_threshold)
        canvas[remask] = MASK

    return canvas
```

This enables "changing your mind" - tokens can oscillate between filled and masked until the model converges on a confident solution.

---

## Hyperparameters

### Model Size (Toy)

| Parameter | Value |
|-----------|-------|
| hidden_size | 256 |
| num_heads | 8 |
| num_layers | 2 (per RefineBlock) |
| num_refine_steps | 4 |
| vocab_size | 50,257 (GPT-2) |
| **Total params** | ~13-27M |

### Training

| Parameter | Value |
|-----------|-------|
| batch_size | 32 |
| learning_rate | 1e-4 |
| optimizer | AdamW |
| scheduler | CosineAnnealing |
| grad_clip | 1.0 |
| precision | FP16 |

### Generation

| Parameter | Recommended |
|-----------|-------------|
| temperature | 0.8-0.9 |
| num_steps | 10-20 |
| tokens_per_step | gen_length / num_steps |

---

## Comparison to Other Architectures

| Aspect | GPT (Autoregressive) | BERT (Masked LM) | LLaDA (Diffusion) | LIRA (This) |
|--------|---------------------|------------------|-------------------|-------------|
| Attention | Causal | Bidirectional | Bidirectional | Bidirectional |
| Generation | Sequential | N/A | Parallel iterative | Parallel iterative |
| Refinement | None | None | Confidence-based | Confidence + re-mask |
| Network reuse | No | No | No | Yes (K iterations) |
| Training | Next token | Masked tokens | Variable masking | Variable masking |

---

## Theoretical Foundations

### Why Iterative Refinement?

1. **Global coherence**: Each pass sees the full context, enabling long-range dependencies
2. **Uncertainty resolution**: Early passes establish structure, later passes refine details
3. **Parameter efficiency**: Same network reused K times = K× effective depth

### Why Bidirectional?

Autoregressive models suffer from:
- **Exposure bias**: Training sees ground truth, inference sees own predictions
- **Error accumulation**: Early mistakes propagate
- **Left-to-right bias**: Can't revise earlier tokens

Bidirectional attention avoids these by predicting all positions with full context.

### Why Variable Corruption?

Fixed masking ratio (e.g., 15% in BERT) trains the model for one specific task. Variable corruption (10-100%) trains the model to handle **any** level of uncertainty, essential for generation where the canvas evolves from fully masked to fully filled.

---

## Limitations & Future Work

### Current Limitations

1. **Scale**: 13-27M params is tiny; benefits may amplify at scale
2. **Re-masking instability**: Confidence head needs better calibration
3. **Speed**: K forward passes per generation step (vs 1 for autoregressive)
4. **Training data**: Only tested on TinyStories

### Future Directions

1. **Hierarchical latents**: z_H (structure) + z_L (details), like TRM
2. **Learned halting**: ACT to decide K dynamically per sequence
3. **Hybrid generation**: Autoregressive for speed, refinement for quality
4. **Instruction tuning**: Adapt for chat/reasoning tasks

---

## References

- **LLaDA**: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
- **TRM**: [Tiny Recursive Models](https://arxiv.org/abs/2510.04871)
- **BERT**: [Bidirectional Encoder Representations](https://arxiv.org/abs/1810.04805)
- **RoPE**: [Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- **SwiGLU**: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

---

*"Tokens shimmer until they crystallize."* 💜
