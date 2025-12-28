"""
Dataset for Latent Canvas Model training.

Key feature: Variable corruption ratio (LLaDA-style)
- Each sample has random masking ratio t ~ U[0, 1]
- Loss weighted by 1/t (harder = more credit)
- Model learns to handle ANY corruption level
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# HuggingFace authentication for gated datasets
# Set HF_TOKEN env var or run: huggingface-cli login
HF_TOKEN = os.environ.get("HF_TOKEN", None)

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import random


class GPTDataset(Dataset):
    """
    Dataset for GPT-style autoregressive language modeling.

    For each sample:
    - input_ids: token sequence
    - labels: shifted by 1 (next token prediction)
    """

    def __init__(
        self,
        token_ids: list[list[int]],
        max_seq_len: int = 128,
        pad_token_id: int = 0,
    ):
        self.token_ids = token_ids
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.token_ids[idx]

        # Truncate if needed
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        seq_len = len(tokens)

        # Pad if needed
        if seq_len < self.max_seq_len:
            padding = [self.pad_token_id] * (self.max_seq_len - seq_len)
            tokens = tokens + padding

        input_ids = torch.tensor(tokens, dtype=torch.long)

        # Labels: same as input (loss computed with shift inside model)
        # Use -100 for padding positions to ignore in loss
        labels = input_ids.clone()
        labels[seq_len:] = -100  # Ignore padding in loss

        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        attention_mask[:seq_len] = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


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


def _get_tokenizer(vocab_size: int, texts: list[str], tokenizer_cache_dir: str = "tokenizers", dataset_name: str = "unknown"):
    """Get tokenizer based on vocab_size setting."""
    if vocab_size > 0:
        # Custom SentencePiece tokenizer
        from lira.tokenizer import get_or_train_tokenizer
        tokenizer = get_or_train_tokenizer(
            texts=texts,
            vocab_size=vocab_size,
            cache_dir=tokenizer_cache_dir,
            dataset_name=dataset_name,
        )
        final_vocab_size = tokenizer.vocab_size
    else:
        # GPT-2 tokenizer
        from transformers import AutoTokenizer
        print("Loading tokenizer (gpt2)...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        final_vocab_size = tokenizer.vocab_size
    return tokenizer, final_vocab_size


def _tokenize_texts(texts: list[str], tokenizer, vocab_size: int, min_length: int = 10) -> list[list[int]]:
    """Tokenize texts with progress logging."""
    num_texts = len(texts)
    print(f"Tokenizing {num_texts} texts...")
    token_ids = []
    done = 0
    for text in texts:
        tokens = tokenizer.encode(text) if vocab_size > 0 else tokenizer.encode(text, add_special_tokens=False)
        done += 1

        if done % 1000 == 0:
            print(f"    {done}/{num_texts} - {(done / float(num_texts) * 100):.2f}%")

        if len(tokens) > min_length:
            token_ids.append(tokens)
    return token_ids


def load_tinystories(
    num_samples: int,
    split: str = "train",
    seed: int = 42,
    vocab_size: int = 0,  # 0 = use GPT-2, >0 = use custom tokenizer
    tokenizer_cache_dir: str = "tokenizers",
    return_tokenizer: bool = False,
) -> tuple[list[list[int]], int] | tuple[list[list[int]], int, any]:
    """
    Load TinyStories dataset with GPT-2 or custom tokenizer.

    Args:
        num_samples: Number of samples to load
        split: Dataset split ("train" or "validation")
        seed: Random seed
        vocab_size: 0 for GPT-2, >0 for custom SentencePiece tokenizer
        tokenizer_cache_dir: Directory to cache custom tokenizer
        return_tokenizer: If True, also return the tokenizer object

    Returns:
        token_ids: List of tokenized stories
        vocab_size: Vocabulary size
        tokenizer: (optional) The tokenizer object if return_tokenizer=True
    """
    from datasets import load_dataset

    print(f"Loading TinyStories ({split})...")
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True, token=HF_TOKEN)

    # Collect samples
    random.seed(seed)
    stories = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        stories.append(example["text"])

    # Get tokenizer and tokenize
    tokenizer, final_vocab_size = _get_tokenizer(vocab_size, stories, tokenizer_cache_dir, dataset_name="tinystories")
    token_ids = _tokenize_texts(stories, tokenizer, vocab_size)

    print(f"Loaded {len(token_ids)} stories, vocab_size={final_vocab_size}")
    if return_tokenizer:
        return token_ids, final_vocab_size, tokenizer
    return token_ids, final_vocab_size


def load_everyday_conversations(
    num_samples: int,
    split: str = "train",
    seed: int = 42,
    vocab_size: int = 0,
    tokenizer_cache_dir: str = "tokenizers",
    return_tokenizer: bool = False,
) -> tuple[list[list[int]], int] | tuple[list[list[int]], int, any]:
    """
    Load HuggingFaceTB/everyday-conversations-llama3.1-2k dataset.

    Small dataset (~2K samples) of everyday conversations.
    Format: Multi-turn conversations with user/assistant roles.
    """
    from datasets import load_dataset

    print(f"Loading everyday-conversations ({split})...")
    # This dataset has 'train_sft' and 'test_sft' splits
    hf_split = "train_sft" if split == "train" else "test_sft"
    dataset = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split=hf_split)

    random.seed(seed)

    # Convert conversations to text
    texts = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        # Format multi-turn conversation
        conversation = example.get("messages", example.get("conversations", []))
        text_parts = []
        for turn in conversation:
            role = turn.get("role", turn.get("from", "user"))
            content = turn.get("content", turn.get("value", ""))
            text_parts.append(f"{role}: {content}")
        texts.append("\n".join(text_parts))

    tokenizer, final_vocab_size = _get_tokenizer(vocab_size, texts, tokenizer_cache_dir, dataset_name="everyday")
    token_ids = _tokenize_texts(texts, tokenizer, vocab_size, min_length=5)

    print(f"Loaded {len(token_ids)} conversations, vocab_size={final_vocab_size}")
    if return_tokenizer:
        return token_ids, final_vocab_size, tokenizer
    return token_ids, final_vocab_size


def load_bitext_customer_support(
    num_samples: int,
    split: str = "train",
    seed: int = 42,
    vocab_size: int = 0,
    tokenizer_cache_dir: str = "tokenizers",
    return_tokenizer: bool = False,
) -> tuple[list[list[int]], int] | tuple[list[list[int]], int, any]:
    """
    Load Bitext customer support chatbot training dataset.

    Format: instruction/response pairs for customer service.
    ~27K samples, 3.57M tokens total.
    """
    from datasets import load_dataset

    print(f"Loading bitext-customer-support ({split})...")
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")

    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Split 90/10 for train/val
    split_idx = int(len(indices) * 0.9)
    if split == "validation":
        indices = indices[split_idx:]
    else:
        indices = indices[:split_idx]

    # Limit to num_samples
    indices = indices[:num_samples]

    texts = []
    for idx in indices:
        example = dataset[idx]
        instruction = example.get("instruction", "")
        response = example.get("response", "")
        # Format as conversation
        text = f"User: {instruction}\nAssistant: {response}"
        texts.append(text)

    tokenizer, final_vocab_size = _get_tokenizer(vocab_size, texts, tokenizer_cache_dir, dataset_name="bitext")
    token_ids = _tokenize_texts(texts, tokenizer, vocab_size, min_length=5)

    print(f"Loaded {len(token_ids)} conversations, vocab_size={final_vocab_size}")
    if return_tokenizer:
        return token_ids, final_vocab_size, tokenizer
    return token_ids, final_vocab_size


def load_chatbot_arena(
    num_samples: int,
    split: str = "train",
    seed: int = 42,
    vocab_size: int = 0,
    tokenizer_cache_dir: str = "tokenizers",
    return_tokenizer: bool = False,
) -> tuple[list[list[int]], int] | tuple[list[list[int]], int, any]:
    """
    Load LMSYS chatbot arena conversations dataset.

    33K real conversations with human preferences.
    Format: Multi-turn conversations from Chatbot Arena.
    """
    from datasets import load_dataset

    print(f"Loading chatbot-arena ({split})...")
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")

    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Split 90/10 for train/val
    split_idx = int(len(indices) * 0.9)
    if split == "validation":
        indices = indices[split_idx:]
    else:
        indices = indices[:split_idx]

    indices = indices[:num_samples]

    texts = []
    for idx in indices:
        example = dataset[idx]
        # Use conversation_a (first model's conversation)
        conversation = example.get("conversation_a", [])
        text_parts = []
        for turn in conversation:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            text_parts.append(f"{role}: {content}")
        if text_parts:
            texts.append("\n".join(text_parts))

    tokenizer, final_vocab_size = _get_tokenizer(vocab_size, texts, tokenizer_cache_dir, dataset_name="arena")
    token_ids = _tokenize_texts(texts, tokenizer, vocab_size, min_length=5)

    print(f"Loaded {len(token_ids)} conversations, vocab_size={final_vocab_size}")
    if return_tokenizer:
        return token_ids, final_vocab_size, tokenizer
    return token_ids, final_vocab_size


def load_shimmer_blend(
    num_samples: int,
    split: str = "train",
    seed: int = 42,
    vocab_size: int = 0,
    tokenizer_cache_dir: str = "tokenizers",
    return_tokenizer: bool = False,
) -> tuple[list[list[int]], int] | tuple[list[list[int]], int, any]:
    """
    Load Shimmer Blend - a curated mix of high-quality instruction datasets.

    Blend composition (proportions of num_samples):
    - SlimOrca-Dedup (40%): General instruction following
    - Orca-Math (20%): Math reasoning
    - UltraChat (25%): Multi-turn dialogue
    - CodeFeedback (15%): Code understanding

    Total: ~450k samples available, scales with num_samples.
    """
    from datasets import load_dataset

    random.seed(seed)

    # Calculate samples per source (proportions)
    n_slimorca = int(num_samples * 0.40)
    n_math = int(num_samples * 0.20)
    n_ultrachat = int(num_samples * 0.25)
    n_code = num_samples - n_slimorca - n_math - n_ultrachat  # remainder (~15%)

    all_texts = []

    # --- 1. SlimOrca-Dedup (40%) - General instruction following ---
    print(f"Loading SlimOrca-Dedup ({n_slimorca} samples)...")
    try:
        slimorca = load_dataset("Open-Orca/SlimOrca-Dedup", split="train", streaming=True, token=HF_TOKEN)
        count = 0
        for example in slimorca:
            if count >= n_slimorca:
                break
            conversations = example.get("conversations", [])
            text_parts = []
            for turn in conversations:
                role = turn.get("from", "user")
                content = turn.get("value", "")
                if role == "system":
                    text_parts.append(f"System: {content}")
                elif role == "human":
                    text_parts.append(f"User: {content}")
                elif role == "gpt":
                    text_parts.append(f"Assistant: {content}")
            if text_parts:
                all_texts.append("\n".join(text_parts))
                count += 1
        print(f"  Loaded {count} SlimOrca samples")
    except Exception as e:
        print(f"  Warning: Could not load SlimOrca-Dedup: {e}")

    # --- 2. Orca-Math (20%) - Math reasoning ---
    print(f"Loading Orca-Math ({n_math} samples)...")
    try:
        orca_math = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True, token=HF_TOKEN)
        count = 0
        for example in orca_math:
            if count >= n_math:
                break
            question = example.get("question", "")
            answer = example.get("answer", "")
            if question and answer:
                text = f"User: {question}\nAssistant: {answer}"
                all_texts.append(text)
                count += 1
        print(f"  Loaded {count} Orca-Math samples")
    except Exception as e:
        print(f"  Warning: Could not load Orca-Math: {e}")

    # --- 3. UltraChat (25%) - Multi-turn dialogue ---
    print(f"Loading UltraChat ({n_ultrachat} samples)...")
    try:
        ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True, token=HF_TOKEN)
        count = 0
        for example in ultrachat:
            if count >= n_ultrachat:
                break
            messages = example.get("messages", [])
            text_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    text_parts.append(f"User: {content}")
                elif role == "assistant":
                    text_parts.append(f"Assistant: {content}")
            if text_parts:
                all_texts.append("\n".join(text_parts))
                count += 1
        print(f"  Loaded {count} UltraChat samples")
    except Exception as e:
        print(f"  Warning: Could not load UltraChat: {e}")

    # --- 4. CodeFeedback (15%) - Code understanding ---
    print(f"Loading CodeFeedback ({n_code} samples)...")
    try:
        codefeedback = load_dataset(
            "m-a-p/CodeFeedback-Filtered-Instruction",
            split="train",
            streaming=True
        )
        count = 0
        for example in codefeedback:
            if count >= n_code:
                break
            query = example.get("query", "")
            answer = example.get("answer", "")
            if query and answer:
                text = f"User: {query}\nAssistant: {answer}"
                all_texts.append(text)
                count += 1
        print(f"  Loaded {count} CodeFeedback samples")
    except Exception as e:
        print(f"  Warning: Could not load CodeFeedback: {e}")

    # Shuffle the blend
    random.shuffle(all_texts)

    # Split for validation if needed
    if split == "validation":
        # Use last 10% for validation
        split_idx = int(len(all_texts) * 0.9)
        all_texts = all_texts[split_idx:]
    else:
        # Use first 90% for training
        split_idx = int(len(all_texts) * 0.9)
        all_texts = all_texts[:split_idx]

    print(f"Total blend size: {len(all_texts)} samples")

    # Tokenize
    tokenizer, final_vocab_size = _get_tokenizer(vocab_size, all_texts, tokenizer_cache_dir, dataset_name="blend")
    token_ids = _tokenize_texts(all_texts, tokenizer, vocab_size, min_length=5)

    print(f"Loaded {len(token_ids)} blended samples, vocab_size={final_vocab_size}")
    if return_tokenizer:
        return token_ids, final_vocab_size, tokenizer
    return token_ids, final_vocab_size


def load_agentic_blend(
    num_samples: int,
    split: str = "train",
    seed: int = 42,
    vocab_size: int = 0,
    tokenizer_cache_dir: str = "tokenizers",
    return_tokenizer: bool = False,
) -> tuple[list[list[int]], int] | tuple[list[list[int]], int, any]:
    """
    Load Agentic Blend - high-quality datasets for 512M+ models.

    Blend composition (proportions of num_samples):
    - Nemotron-v2 (30%): General + reasoning + math + code
    - OpenMathInstruct-2 (22%): Math reasoning (Llama-405B generated)
    - OpenHermes-2.5 (15%): Proven instruction following
    - Ling-Coder-SFT (15%): Code in 20 languages
    - smoltalk (12%): Diverse conversations
    - xlam-function-calling (6%): Agentic/tool use

    Designed for 512M+ parameter models with ~6M samples available.
    """
    from datasets import load_dataset

    random.seed(seed)

    # Calculate samples per source
    n_nemotron = int(num_samples * 0.30)
    n_math = int(num_samples * 0.22)
    n_hermes = int(num_samples * 0.15)
    n_code = int(num_samples * 0.15)
    n_smol = int(num_samples * 0.12)
    n_function = num_samples - n_nemotron - n_math - n_hermes - n_code - n_smol  # ~6%

    all_texts = []

    # --- 1. Nemotron-Post-Training-v2 (30%) - Multi-domain ---
    print(f"Loading Nemotron-v2 ({n_nemotron} samples)...")
    try:
        nemotron = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2", split="train", streaming=True, token=HF_TOKEN)
        count = 0
        for example in nemotron:
            if count >= n_nemotron:
                break
            # Handle different conversation formats
            if "conversations" in example:
                convs = example["conversations"]
                text_parts = []
                for turn in convs:
                    role = turn.get("role", turn.get("from", "user"))
                    content = turn.get("content", turn.get("value", ""))
                    if role in ["system"]:
                        text_parts.append(f"System: {content}")
                    elif role in ["user", "human"]:
                        text_parts.append(f"User: {content}")
                    elif role in ["assistant", "gpt"]:
                        text_parts.append(f"Assistant: {content}")
                if text_parts:
                    all_texts.append("\n".join(text_parts))
                    count += 1
            elif "input" in example and "output" in example:
                text = f"User: {example['input']}\nAssistant: {example['output']}"
                all_texts.append(text)
                count += 1
        print(f"  Loaded {count} Nemotron samples")
    except Exception as e:
        print(f"  Warning: Could not load Nemotron-v2: {e}")

    # --- 2. OpenMathInstruct-2 (22%) - Math reasoning ---
    print(f"Loading OpenMathInstruct-2 ({n_math} samples)...")
    try:
        mathinstruct = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True, token=HF_TOKEN)
        count = 0
        for example in mathinstruct:
            if count >= n_math:
                break
            problem = example.get("problem", example.get("question", ""))
            solution = example.get("generated_solution", example.get("solution", example.get("answer", "")))
            if problem and solution:
                text = f"User: {problem}\nAssistant: {solution}"
                all_texts.append(text)
                count += 1
        print(f"  Loaded {count} OpenMathInstruct samples")
    except Exception as e:
        print(f"  Warning: Could not load OpenMathInstruct-2: {e}")

    # --- 3. OpenHermes-2.5 (15%) - Instruction following ---
    print(f"Loading OpenHermes-2.5 ({n_hermes} samples)...")
    try:
        hermes = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True, token=HF_TOKEN)
        count = 0
        for example in hermes:
            if count >= n_hermes:
                break
            conversations = example.get("conversations", [])
            text_parts = []
            for turn in conversations:
                role = turn.get("from", "user")
                content = turn.get("value", "")
                if role == "system":
                    text_parts.append(f"System: {content}")
                elif role == "human":
                    text_parts.append(f"User: {content}")
                elif role == "gpt":
                    text_parts.append(f"Assistant: {content}")
            if text_parts:
                all_texts.append("\n".join(text_parts))
                count += 1
        print(f"  Loaded {count} OpenHermes samples")
    except Exception as e:
        print(f"  Warning: Could not load OpenHermes-2.5: {e}")

    # --- 4. Ling-Coder-SFT (15%) - Code ---
    print(f"Loading Ling-Coder-SFT ({n_code} samples)...")
    try:
        lingcoder = load_dataset("inclusiveai/Ling-Coder-SFT", split="train", streaming=True, token=HF_TOKEN)
        count = 0
        for example in lingcoder:
            if count >= n_code:
                break
            # Try different field names
            if "conversations" in example:
                convs = example["conversations"]
                text_parts = []
                for turn in convs:
                    role = turn.get("role", turn.get("from", "user"))
                    content = turn.get("content", turn.get("value", ""))
                    if role in ["user", "human"]:
                        text_parts.append(f"User: {content}")
                    elif role in ["assistant", "gpt"]:
                        text_parts.append(f"Assistant: {content}")
                if text_parts:
                    all_texts.append("\n".join(text_parts))
                    count += 1
            elif "instruction" in example:
                instr = example.get("instruction", "")
                inp = example.get("input", "")
                out = example.get("output", "")
                if instr and out:
                    query = f"{instr}\n{inp}" if inp else instr
                    text = f"User: {query}\nAssistant: {out}"
                    all_texts.append(text)
                    count += 1
        print(f"  Loaded {count} Ling-Coder samples")
    except Exception as e:
        print(f"  Warning: Could not load Ling-Coder-SFT: {e}")

    # --- 5. smoltalk (12%) - Diverse conversations ---
    print(f"Loading smoltalk ({n_smol} samples)...")
    try:
        smoltalk = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True, token=HF_TOKEN)
        count = 0
        for example in smoltalk:
            if count >= n_smol:
                break
            messages = example.get("messages", [])
            text_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    text_parts.append(f"System: {content}")
                elif role == "user":
                    text_parts.append(f"User: {content}")
                elif role == "assistant":
                    text_parts.append(f"Assistant: {content}")
            if text_parts:
                all_texts.append("\n".join(text_parts))
                count += 1
        print(f"  Loaded {count} smoltalk samples")
    except Exception as e:
        print(f"  Warning: Could not load smoltalk: {e}")

    # --- 6. xlam-function-calling (6%) - Agentic/tool use ---
    print(f"Loading xlam-function-calling ({n_function} samples)...")
    try:
        xlam = load_dataset("Salesforce/xlam-function-calling-60k", split="train", streaming=True, token=HF_TOKEN)
        count = 0
        for example in xlam:
            if count >= n_function:
                break
            query = example.get("query", example.get("instruction", ""))
            tools = example.get("tools", "")
            answers = example.get("answers", example.get("output", ""))
            if query and answers:
                # Format with tools context if available
                if tools:
                    text = f"System: Available tools: {tools}\nUser: {query}\nAssistant: {answers}"
                else:
                    text = f"User: {query}\nAssistant: {answers}"
                all_texts.append(text)
                count += 1
        print(f"  Loaded {count} xlam function-calling samples")
    except Exception as e:
        print(f"  Warning: Could not load xlam-function-calling: {e}")

    # Shuffle the blend
    random.shuffle(all_texts)

    # Split for validation if needed
    if split == "validation":
        split_idx = int(len(all_texts) * 0.9)
        all_texts = all_texts[split_idx:]
    else:
        split_idx = int(len(all_texts) * 0.9)
        all_texts = all_texts[:split_idx]

    print(f"Total agentic blend size: {len(all_texts)} samples")

    # Tokenize
    tokenizer, final_vocab_size = _get_tokenizer(vocab_size, all_texts, tokenizer_cache_dir, dataset_name="agentic")
    token_ids = _tokenize_texts(all_texts, tokenizer, vocab_size, min_length=5)

    print(f"Loaded {len(token_ids)} agentic samples, vocab_size={final_vocab_size}")
    if return_tokenizer:
        return token_ids, final_vocab_size, tokenizer
    return token_ids, final_vocab_size


# Dataset registry
DATASETS = {
    "tinystories": {
        "loader": load_tinystories,
        "description": "TinyStories - simple children's stories (2.1M samples)",
        "default_prompt": "Once upon a time",
    },
    "everyday": {
        "loader": load_everyday_conversations,
        "description": "Everyday conversations - small chatbot dataset (~2K samples)",
        "default_prompt": "User: Hello, how are you?\nAssistant:",
    },
    "bitext": {
        "loader": load_bitext_customer_support,
        "description": "Bitext customer support - instruction/response pairs (~27K samples)",
        "default_prompt": "User: I need help with my order\nAssistant:",
    },
    "arena": {
        "loader": load_chatbot_arena,
        "description": "Chatbot Arena - real conversations with preferences (33K samples)",
        "default_prompt": "user: What is the capital of France?\nassistant:",
    },
    "blend": {
        "loader": load_shimmer_blend,
        "description": "Shimmer Blend - SlimOrca(40%) + OrcaMath(20%) + UltraChat(25%) + Code(15%)",
        "default_prompt": "User: Explain how neural networks learn.\nAssistant:",
    },
    "agentic": {
        "loader": load_agentic_blend,
        "description": "Agentic Blend - Nemotron(30%) + MathInstruct(22%) + Hermes(15%) + Code(15%) + Smol(12%) + Tools(6%)",
        "default_prompt": "User: Write a Python function to calculate fibonacci numbers.\nAssistant:",
    },
}


def load_dataset_by_name(
    name: str,
    num_samples: int,
    split: str = "train",
    seed: int = 42,
    vocab_size: int = 0,
    tokenizer_cache_dir: str = "tokenizers",
    return_tokenizer: bool = False,
):
    """
    Load dataset by name.

    Available datasets:
    - tinystories: TinyStories children's stories
    - everyday: Everyday conversations (small, ~2K)
    - bitext: Bitext customer support (~27K)
    - arena: Chatbot Arena conversations (33K)
    """
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    loader = DATASETS[name]["loader"]
    return loader(
        num_samples=num_samples,
        split=split,
        seed=seed,
        vocab_size=vocab_size,
        tokenizer_cache_dir=tokenizer_cache_dir,
        return_tokenizer=return_tokenizer,
    )


def get_default_prompt(dataset_name: str) -> str:
    """Get the default generation prompt for a dataset."""
    if dataset_name in DATASETS:
        return DATASETS[dataset_name]["default_prompt"]
    return "Once upon a time"


def list_datasets() -> dict:
    """List all available datasets with descriptions."""
    return {name: info["description"] for name, info in DATASETS.items()}


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
    """Create a DataLoader for LIRA/masked LM training."""
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


def create_gpt_dataloader(
    token_ids: list[list[int]],
    batch_size: int = 32,
    max_seq_len: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for GPT/autoregressive training."""
    dataset = GPTDataset(
        token_ids=token_ids,
        max_seq_len=max_seq_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
