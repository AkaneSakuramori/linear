"""Second synthetic corpus (generalization test D: unseen corpus).

The same character set / tokenizer as corpus A is reused so the model
architecture (and therefore layer segments) is identical, but the WORD-level
distribution differs sharply (high-frequency English-like words vs the
Greek-alphabet word list of corpus A), shifting the character statistics. This
gives a distribution shift that is cheap and CPU-friendly.
"""
from __future__ import annotations

import os
import random
from typing import Dict

import torch

WORD_VOCAB_B = [
    "the", "and", "of", "to", "in", "a", "is", "that", "it", "for",
    "was", "on", "are", "as", "with", "his", "they", "at", "be", "this",
    "have", "from", "or", "one", "had", "by", "word", "but", "not", "what",
]


def generate_corpus_b_text(seed: int, n_chars_target: int) -> str:
    rng = random.Random(seed)
    words = []
    n = 0
    while n < n_chars_target:
        if words and rng.random() < 0.18:   # different punctuation rate vs corpus A (0.10)
            words.append(".")
            n += 1
        words.append(" ")
        n += 1
        w = WORD_VOCAB_B[rng.randrange(len(WORD_VOCAB_B))]
        words.append(w)
        n += len(w)
    return "".join(words)


def build_corpus_b(data_dir: str, tokenizer_a, train_chars: int, val_chars: int,
                   train_seed: int, val_seed: int, force: bool = False) -> Dict:
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train_b.txt")
    val_path = os.path.join(data_dir, "val_b.txt")
    if not force and os.path.exists(train_path) and os.path.exists(val_path):
        with open(train_path, "r", encoding="utf-8") as f:
            train_text = f.read()
        with open(val_path, "r", encoding="utf-8") as f:
            val_text = f.read()
    else:
        train_text = generate_corpus_b_text(train_seed, train_chars)
        val_text = generate_corpus_b_text(val_seed, val_chars)
        with open(train_path, "w", encoding="utf-8") as f:
            f.write(train_text)
        with open(val_path, "w", encoding="utf-8") as f:
            f.write(val_text)

    # ensure the second corpus uses only the tokenizer's charset (same vocab)
    allowed = set(tokenizer_a.itos.values())
    assert set(train_text) <= allowed and set(val_text) <= allowed, \
        "corpus B uses chars outside corpus A's vocabulary"

    train_ids = torch.tensor(tokenizer_a.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer_a.encode(val_text), dtype=torch.long)
    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "tokenizer": tokenizer_a,
        "n_train_tokens": int(train_ids.numel()),
        "n_val_tokens": int(val_ids.numel()),
        "vocab_size": tokenizer_a.vocab_size,
        "data_dir": data_dir,
        "corpus": "B",
    }
