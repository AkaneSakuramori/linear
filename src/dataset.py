"""Tiny deterministic synthetic corpus + causal-LM window batching.

The corpus is a seeded pseudo-random stream of words and punctuation, so:
  * it is fully reproducible given (seed, target size),
  * it never touches the network/disk for large downloads,
  * it is small enough to be memorized by a tiny transformer (loss drops).

The token-level corpus is cached to `data/train.txt` / `data/val.txt` and the
tokenizer to `data/tokenizer.json` so a run can be repeated or resumed.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import torch

from src.tokenizer import CharTokenizer

WORD_VOCAB = [
    "alpha", "beta", "gamma", "delta", "the", "quick", "brown", "fox",
    "jumps", "over", "lazy", "dog", "cat", "sat", "on", "mat",
    "run", "walks", "sings", "thinks", "code", "learns",
]


def generate_corpus_text(seed: int, n_chars_target: int) -> str:
    """Stream of words/punctuation of at least `n_chars_target` characters."""
    rng = random.Random(seed)
    words: List[str] = []
    n = 0
    while n < n_chars_target:
        if words and rng.random() < 0.10:
            words.append(".")  # sentence-ish boundary (no trailing space yet)
            n += 1
        words.append(" ")
        n += 1
        w = WORD_VOCAB[rng.randrange(len(WORD_VOCAB))]
        words.append(w)
        n += len(w)
    return "".join(words)


def build_corpora(data_dir: str,
                  train_chars: int,
                  val_chars: int,
                  train_seed: int,
                  val_seed: int,
                  force: bool = False) -> Dict:
    """Generate (or load cached) tokenized train/val corpora.

    Returns dict with keys: train_ids, val_ids (LongTensors of token ids),
    tokenizer, n_train_tokens, n_val_tokens, vocab_size, data_dir.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.txt")
    val_path = os.path.join(data_dir, "val.txt")
    tok_path = os.path.join(data_dir, "tokenizer.json")

    if not force and os.path.exists(train_path) and os.path.exists(val_path):
        with open(train_path, "r", encoding="utf-8") as f:
            train_text = f.read()
        with open(val_path, "r", encoding="utf-8") as f:
            val_text = f.read()
    else:
        train_text = generate_corpus_text(train_seed, train_chars)
        val_text = generate_corpus_text(val_seed, val_chars)
        with open(train_path, "w", encoding="utf-8") as f:
            f.write(train_text)
        with open(val_path, "w", encoding="utf-8") as f:
            f.write(val_text)

    if os.path.exists(tok_path):
        tokenizer = CharTokenizer.load(tok_path)
    else:
        tokenizer = CharTokenizer(train_text)
        tokenizer.save(tok_path)

    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "tokenizer": tokenizer,
        "n_train_tokens": int(train_ids.numel()),
        "n_val_tokens": int(val_ids.numel()),
        "vocab_size": tokenizer.vocab_size,
        "data_dir": data_dir,
    }


# ---------------------------------------------------------------------------
# Window batching
# ---------------------------------------------------------------------------
def _starts_for(global_index: int, seed: int, count: int, max_start: int) -> List[int]:
    """Deterministic pseudo-random window offsets for a given flow step.

    Only depends on (global_index, seed), so training can be resumed exactly.
    """
    rng = random.Random(seed * 0x9E3779B1 + global_index * 0x85EBCA6B)
    return [rng.randrange(0, max_start + 1) for _ in range(count)]


def make_window_batch(ids: torch.Tensor, context_length: int, global_index: int,
                      seed: int, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sampled windows of `count` sequences: (x, targets) of shape (count, ctx)."""
    max_start = int(ids.numel()) - context_length
    if max_start < 1:
        raise ValueError("corpus shorter than context_length")
    starts = torch.tensor(
        _starts_for(global_index, seed, count, max_start), dtype=torch.long)
    pos = starts[:, None] + torch.arange(context_length, dtype=torch.long)
    x = ids[pos]
    y = ids[pos + 1]
    return x, y


def get_eval_starts(n_tokens: int, context_length: int,
                    max_windows: int) -> List[int]:
    """Deterministic stride-sampled start positions covering the val region."""
    n_tokens = max(1, int(n_tokens))
    total_windows = max(1, n_tokens - context_length)
    if max_windows <= 0:
        return list(range(total_windows))
    stride = max(1, (total_windows + max_windows - 1) // max_windows)
    return list(range(0, total_windows, stride))