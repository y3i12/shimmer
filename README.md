# Shimmer 💜

**A LIRA Implementation**
*(Latent Iterative Refinement Architecture)*

Shimmer is an experimental language model that generates text through iterative refinement of a latent canvas, inspired by diffusion models and recursive reasoning architectures.

For the moment being, Shimmer has the "Toy LLM" aura and doesn't do much.

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
┌─────────────────────────────────────────┐
│         LATENT CANVAS                   │
│  [z₀] [z₁] [z₂] ... [zₙ]               │
│         ↓                               │
│    REFINE BLOCK (applied K times)       │
│    - Bidirectional attention            │
│    - Same weights each iteration        │
│         ↓                               │
│    DECODE → tokens + confidence         │
└─────────────────────────────────────────┘
```

**Key components:**
- `RefineBlock`: Transformer layers applied iteratively (TRM-inspired)
- `LatentCanvasModel`: Embeds tokens → refines → decodes
- Confidence head: Learns to predict prediction correctness

---

## Training Phases

| Phase | What it tests | Key setting |
|-------|---------------|-------------|
| **1** | Baseline single-pass | K=1, 30% masking |
| **2** | Multiple iterations help? | K=4, 30% masking |
| **3** | Variable corruption | K=4, 10-100% masking |
| **4** | Confidence supervision | + confidence loss |

```bash
# Phase 1: Baseline
python train.py --phase 1 --num_samples 50000 --epochs 10 \
    --hidden_size 256 --num_heads 8 --device cuda --fp16

# Phase 2: Test iterative refinement
python train.py --phase 2 --load_checkpoint checkpoints/phase1_best.pt ...

# Phase 3: Variable corruption (LLaDA-style)
python train.py --phase 3 --load_checkpoint checkpoints/phase2_best.pt ...

# Phase 4: Confidence supervision
python train.py --phase 4 --load_checkpoint checkpoints/phase3_best.pt ...
```

---

## Results (13M parameter toy model)

| Phase | Val Loss | Val Acc | Notes |
|-------|----------|---------|-------|
| 1 | 6.38 | 61.6% | Baseline works |
| 2 | 5.50 | 65.5% | +4% from iterations |
| 3 | 4.72 | 40.7%* | Handles variable corruption |
| 4 | 4.63 | 41.5% | Confidence head learns |

*Accuracy drops because task is harder (10-100% masking vs fixed 30%)

---

## Generation

```python
from lira import LatentCanvasModel, LatentCanvasConfig
import torch

# Load model
checkpoint = torch.load('checkpoints/phase4_best.pt', weights_only=False)
model = LatentCanvasModel(checkpoint['config']).cuda()
model.load_state_dict(checkpoint['model_state_dict'])

# Generate
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

prompt = "Once upon a time"
prompt_ids = torch.tensor([tokenizer.encode(prompt)], device='cuda')

canvas, history = model.generate_topk(
    prompt_ids,
    gen_length=50,
    num_steps=15,
    temperature=0.8
)

print(tokenizer.decode(canvas[0].tolist()))
```

---

## Project Structure

```
shimmer/
├── lira/                    # LIRA architecture
│   ├── __init__.py
│   ├── layers.py           # Transformer building blocks
│   └── canvas.py           # LatentCanvasModel
├── data/
│   ├── __init__.py
│   └── dataset.py          # TinyStories + variable corruption
├── train.py                # Training script (4 phases)
├── checkpoints/            # Saved models
└── README.md
```

---

## Inspirations

- **LLaDA**: Masked diffusion for language, iterative unmasking
- **TinyRecursiveModels (TRM)**: Same network applied recursively
- **BERT**: Masked language modeling foundation
- **Diffusion models**: Iterative refinement from noise

---

## Requirements

```
torch>=2.0
transformers
datasets
```

---

## Future Directions

- [ ] Scale to 300M-1B parameters
- [ ] Better confidence calibration for re-masking
- [ ] Instruction tuning for chat capability
- [ ] Explore hierarchical latent structure

---

## Name Origin

> *Shimmer*: Tokens shimmer between states—uncertain, fluid—until they crystallize into text.

Also inspired by Arcane's Shimmer: a transformative substance that grants power through iteration and change. 💜

---

## License

MIT

---

*Built with curiosity and 無* 🖤
