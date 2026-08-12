"""Simple deterministic character-level tokenizer.

Maps single characters (plus a small set of special tokens) to integer ids.
Cast to str/int explicitly where needed; vocab is stored in a deterministic
sorted order so building is reproducible across processes.
"""
from __future__ import annotations

import json
import os
from typing import List

SPECIAL_TOKENS: List[str] = ["<pad>", "<bos>", "<eos>", "<unk>"]


class CharTokenizer:
    def __init__(self, text: str, specials: List[str] | None = None):
        specials = specials or SPECIAL_TOKENS
        chars = sorted(set(text))
        self.specials = list(specials)
        self.chars = sorted(c for c in chars if c not in set(self.specials))
        self.itos = {i: ch for i, ch in enumerate(self.specials + self.chars)}
        self.stoi = {ch: i for i, ch in self.itos.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i in self.itos:
                out.append(self.itos[i])
        return "".join(out)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"specials": self.specials, "chars": self.chars}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.specials = list(data["specials"])
        tok.chars = list(data["chars"])
        tok.itos = {i: ch for i, ch in enumerate(tok.specials + tok.chars)}
        tok.stoi = {ch: i for i, ch in tok.itos.items()}
        return tok