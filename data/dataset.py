"""
Dataset for Latent Canvas Model training.

Key feature: Variable corruption ratio (LLaDA-style)
- Each sample has random masking ratio t ~ U[0, 1]
- Loss weighted by 1/t (harder = more credit)
- Model learns to handle ANY corruption level
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import random


class LatentCanvasDataset(Dataset):
    """
    Dataset with LLaDA-style variable corruption.

    For each sample:
    1. Sample corruption ratio t ~ U[eps, 1]
    2. Mask t% of tokens
    3. Return (corrupted, original, mask_ratio)
    """

    def __init__(
        self,
        token_ids: list[list[int]],
        mask_token_id: int,
        max_seq_len: int = 128,
        min_mask_ratio: float = 0.1,
        max_mask_ratio: float = 1.0,
        pad_token_id: int = 0,
    ):
        self.token_ids = token_ids
        self.mask_token_id = mask_token_id
        self.max_seq_len = max_seq_len
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.token_ids[idx]

        # Truncate or pad
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        seq_len = len(tokens)

        # Pad if needed
        if seq_len < self.max_seq_len:
            padding = [self.pad_token_id] * (self.max_seq_len - seq_len)
            tokens = tokens + padding

        original = torch.tensor(tokens, dtype=torch.long)

        # Sample corruption ratio (LLaDA-style)
        mask_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)

        # Create corruption mask (only for non-padding positions)
        num_to_mask = int(seq_len * mask_ratio)
        num_to_mask = max(1, num_to_mask)  # At least one mask

        # Random positions to mask (within actual sequence, not padding)
        mask_positions = torch.zeros(self.max_seq_len, dtype=torch.bool)
        if seq_len > 0:
            indices = torch.randperm(seq_len)[:num_to_mask]
            mask_positions[indices] = True

        # Create corrupted input
        corrupted = original.clone()
        corrupted[mask_positions] = self.mask_token_id

        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        attention_mask[:seq_len] = True

        return {
            "input_ids": corrupted,
            "labels": original,
            "mask_positions": mask_positions,
            "attention_mask": attention_mask,
            "mask_ratio": torch.tensor(mask_ratio, dtype=torch.float),
        }


def load_tinystories(
    num_samples: int,
    split: str = "train",
    seed: int = 42
) -> tuple[list[list[int]], int]:
    """
    Load TinyStories dataset with GPT-2 tokenizer.

    Returns:
        token_ids: List of tokenized stories
        vocab_size: Vocabulary size
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Loading TinyStories ({split})...")
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    print("Loading tokenizer (gpt2)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Collect samples
    random.seed(seed)
    stories = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        stories.append(example["text"])

    # Tokenize
    print(f"Tokenizing {len(stories)} stories...")
    token_ids = []
    for story in stories:
        tokens = tokenizer.encode(story, add_special_tokens=False)
        if len(tokens) > 10:  # Skip very short stories
            token_ids.append(tokens)

    print(f"Loaded {len(token_ids)} stories, vocab_size={tokenizer.vocab_size}")
    return token_ids, tokenizer.vocab_size


def create_dataloader(
    token_ids: list[list[int]],
    mask_token_id: int,
    batch_size: int = 32,
    max_seq_len: int = 128,
    min_mask_ratio: float = 0.1,
    max_mask_ratio: float = 1.0,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for training."""
    dataset = LatentCanvasDataset(
        token_ids=token_ids,
        mask_token_id=mask_token_id,
        max_seq_len=max_seq_len,
        min_mask_ratio=min_mask_ratio,
        max_mask_ratio=max_mask_ratio,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
