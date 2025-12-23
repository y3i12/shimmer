"""
Custom SentencePiece tokenizer for Shimmer.

Features:
- Train on TinyStories with configurable vocab size
- Cache trained model to disk for reuse
- Drop-in replacement for GPT-2 tokenizer
"""

import os
from pathlib import Path
from typing import Optional
import tempfile

import sentencepiece as spm


class ShimmerTokenizer:
    """
    SentencePiece BPE tokenizer with small vocabulary.

    Saves/loads from disk automatically.
    """

    # Special token IDs
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    MASK_ID = 4  # Reserved for LIRA masking

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 10000,
    ):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.sp = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    @property
    def mask_token_id(self) -> int:
        return self.MASK_ID

    @property
    def pad_token_id(self) -> int:
        return self.PAD_ID

    @property
    def bos_token_id(self) -> int:
        return self.BOS_ID

    @property
    def eos_token_id(self) -> int:
        return self.EOS_ID

    def train(
        self,
        texts: list[str],
        model_prefix: str = "shimmer_tokenizer",
        vocab_size: Optional[int] = None,
    ) -> str:
        """
        Train tokenizer on texts and save model.

        Args:
            texts: List of training texts
            model_prefix: Prefix for saved model files
            vocab_size: Override default vocab size

        Returns:
            Path to saved model file
        """
        vocab_size = vocab_size or self.vocab_size

        # Write texts to temp file for sentencepiece
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for text in texts:
                f.write(text.strip() + '\n')
            temp_path = f.name

        try:
            # Train sentencepiece model
            # Reserve IDs 0-4 for special tokens
            spm.SentencePieceTrainer.train(
                input=temp_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size - 5,  # Reserve 5 special tokens
                model_type='bpe',
                character_coverage=1.0,
                pad_id=self.PAD_ID,
                unk_id=self.UNK_ID,
                bos_id=self.BOS_ID,
                eos_id=self.EOS_ID,
                user_defined_symbols=['[MASK]'],  # Add mask token
                normalization_rule_name='identity',  # Preserve whitespace
            )

            model_file = f"{model_prefix}.model"
            self.load(model_file)
            self.model_path = model_file

            print(f"Tokenizer trained: vocab_size={self.vocab_size}, saved to {model_file}")
            return model_file

        finally:
            os.unlink(temp_path)

    def load(self, model_path: str):
        """Load trained model from disk."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.model_path = model_path
        print(f"Tokenizer loaded: vocab_size={self.vocab_size} from {model_path}")

    def save(self, path: str):
        """Copy model to new location."""
        if self.model_path and os.path.exists(self.model_path):
            import shutil
            shutil.copy(self.model_path, path)
            # Also copy vocab file if exists
            vocab_path = self.model_path.replace('.model', '.vocab')
            if os.path.exists(vocab_path):
                shutil.copy(vocab_path, path.replace('.model', '.vocab'))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to token IDs."""
        if self.sp is None:
            raise ValueError("Tokenizer not trained/loaded. Call train() or load() first.")

        ids = self.sp.encode(text)

        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]

        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        if self.sp is None:
            raise ValueError("Tokenizer not trained/loaded.")

        if skip_special:
            ids = [i for i in ids if i not in (self.PAD_ID, self.BOS_ID, self.EOS_ID, self.MASK_ID)]

        return self.sp.decode(ids)

    def __call__(self, text: str, **kwargs) -> dict:
        """HuggingFace-style interface."""
        ids = self.encode(text)
        return {"input_ids": ids}


def get_or_train_tokenizer(
    texts: Optional[list[str]] = None,
    vocab_size: int = 10000,
    cache_dir: str = "tokenizers",
    dataset_name: str = "unknown",
    force_retrain: bool = False,
) -> ShimmerTokenizer:
    """
    Get cached tokenizer or train new one.

    Args:
        texts: Training texts (required if no cache exists)
        vocab_size: Vocabulary size (target, actual may differ)
        cache_dir: Directory to cache tokenizer
        dataset_name: Name of the dataset (used in filename)
        force_retrain: Force retraining even if cache exists

    Returns:
        ShimmerTokenizer instance

    Naming convention: shimmer_{dataset}_{actual_vocab}.model
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Look for existing tokenizer with this dataset and approximate vocab size
    # Pattern: shimmer_{dataset}_{actual_vocab}.model
    existing_models = list(Path(cache_dir).glob(f"shimmer_{dataset_name}_*.model"))

    if existing_models and not force_retrain:
        # Find one closest to requested vocab size
        best_match = None
        best_diff = float('inf')
        for model_path in existing_models:
            try:
                # Extract actual vocab from filename: shimmer_blend_9995.model -> 9995
                actual_vocab = int(model_path.stem.split('_')[-1])
                diff = abs(actual_vocab - vocab_size)
                if diff < best_diff:
                    best_diff = diff
                    best_match = model_path
            except (ValueError, IndexError):
                continue

        if best_match and best_diff < vocab_size * 0.1:  # Within 10% of target
            print(f"Loading cached tokenizer from {best_match}")
            return ShimmerTokenizer(model_path=str(best_match), vocab_size=vocab_size)

    # Train new tokenizer
    if texts is None:
        raise ValueError("No cached tokenizer found and no texts provided for training")

    print(f"Training new tokenizer with vocab_size={vocab_size} on {len(texts)} texts...")

    tokenizer = ShimmerTokenizer(vocab_size=vocab_size)
    # Use temp prefix, will rename after training
    temp_prefix = os.path.join(cache_dir, f"shimmer_{dataset_name}_temp")
    tokenizer.train(texts, model_prefix=temp_prefix, vocab_size=vocab_size)

    # Rename with actual vocab size
    actual_vocab = tokenizer.vocab_size
    final_model = os.path.join(cache_dir, f"shimmer_{dataset_name}_{actual_vocab}.model")
    final_vocab = os.path.join(cache_dir, f"shimmer_{dataset_name}_{actual_vocab}.vocab")

    temp_model = f"{temp_prefix}.model"
    temp_vocab = f"{temp_prefix}.vocab"

    # Move files to final names
    if os.path.exists(temp_model):
        os.rename(temp_model, final_model)
        tokenizer.model_path = final_model
    if os.path.exists(temp_vocab):
        os.rename(temp_vocab, final_vocab)

    print(f"Tokenizer saved: {final_model} (actual vocab: {actual_vocab})")

    return tokenizer
