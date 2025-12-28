# Shimmer 💜

**A LIRA Implementation**
*(Latent Iterative Refinement Architecture)*

Shimmer is an experimental language model that generates text through iterative refinement of a latent canvas, inspired by diffusion models and recursive reasoning architectures.

For the moment being, Shimmer has the "Toy LLM" aura and doesn't do much. At the moment, the pre-trained models only synthesize stories based on TinyStories.

The validation for the model working was to train 2 equivalent models, one using LIRA and another using GPT, both with ~12M params. The results were striking - GPT collapsed into severe repetition while LIRA produced coherent stories. See [LIRA 12M evaluation](eval_results/eval_results_lira_12M.txt) and [GPT 12M evaluation](eval_results/eval_results_gpt_12M.txt).

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/y3i12/shimmer.git
cd shimmer
pip install -r requirements.txt

# Train a small model locally
python train.py --progressive \
    --model lira --dataset tinystories \
    --num_samples 100000 --vocab_size 10000 \
    --hidden_size 288 --num_layers 6 --num_heads 6 \
    --batch_size 64 --stage_epochs 3 \
    --device cuda --fp16

# Evaluate
python evaluate.py --checkpoint checkpoints/lira_*_final.pt --full

# Generate text
python generate.py --checkpoint checkpoints/lira_*_final.pt \
    --tokenizer tokenizers/shimmer_*.model \
    --prompt "Once upon a time"
```

---

## Core Idea

Unlike autoregressive models (GPT, LLaMA) that generate left-to-right, Shimmer:

1. **Starts with a canvas** of mask tokens
2. **Refines iteratively** using the same network multiple times
3. **Crystallizes tokens** based on confidence
4. **Can reconsider** - low-confidence tokens may be re-masked

```
[MASK][MASK][MASK][MASK][MASK]
         ↓ refine
[MASK] cat [MASK][MASK][MASK]
         ↓ refine
The cat [MASK][MASK] sleeping
         ↓ refine
The cat was [MASK] sleeping
         ↓ refine
The cat was peacefully sleeping
```

---

## Architecture (LIRA)

```
┌──────────────────────────────────────┐
│         LATENT CANVAS                │
│  [z₀] [z₁] [z₂] ... [zₙ]             │
│         ↓                            │
│    REFINE BLOCK (applied K times)    │
│    - Bidirectional attention         │
│    - Same weights each iteration     │
│         ↓                            │
│    DECODE → tokens + confidence      │
└──────────────────────────────────────┘
```

**Key components:**
- `RefineBlock`: Transformer layers applied iteratively (TRM-inspired)
- `LatentCanvasModel`: Embeds tokens → refines → decodes
- Confidence head: Learns to predict prediction correctness

---

## Training Stages

| Stages | What it tests | Key setting |
|-------|---------------|-------------|
| **1** | Baseline single-pass | K=1, 30% masking |
| **2** | Multiple iterations help? | K=4, 30% masking |
| **3** | Variable corruption | K=4, 10-100% masking |
| **4** | Confidence supervision | + confidence loss |

The training process creates splits of samples for each stage and trains epochs with the splits.

```bash
# 50M tinystories training
python train.py --progressive \
    --model lira \
    --dataset tinystories \
    --num_samples 2000000 \
    --vocab_size 10000 \
    --hidden_size 512 \
    --num_layers 10 \
    --num_heads 8 \
    --max_seq_len 256 \
    --batch_size 64 \
    --stage_epochs 3 \
    --lr 2e-4 \
    --device cuda --fp16 \
    --gpu 1 \
    --checkpoint_name lira_50m_progressive
```

---

## Evaluation Summary (50M parameter toy model)

**Model:** 42,359,808 parameters
**Coherence Rate:** 100.0%
**Confidence Gap:** 0.7126
**Bigram Repetition:** 7.7%
**Perplexity:** 1.64
**Diversity:** 65.9%

_See full results: [LIRA 50M](eval_results/eval_results_lira_50M.txt), [LIRA 12M](eval_results/eval_results_lira_12M.txt), [GPT 12M](eval_results/eval_results_gpt_12M.txt)._

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

1. **Scale**: Validated up to 50M params; benefits seem to amplify at scale
2. **Re-masking instability**: Confidence head needs better calibration at small scale
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

## Name Origin

> *Shimmer*: Tokens shimmer between states—uncertain, fluid—until they crystallize into text.

_ ... but not everything that shimmers is gold ... _

---

## License

MIT

---

*Built with curiosity and 無* 🖤 
