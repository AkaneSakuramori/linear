import tempfile

import pytest
import torch

from src.dataset import build_corpora, generate_corpus_text
from src.model import ModelConfig, Transformer
from src.tokenizer import CharTokenizer

DEFAULT_SEED = 7


@pytest.fixture
def text() -> str:
    return generate_corpus_text(111, 3000)


@pytest.fixture
def tokenizer(text: str) -> CharTokenizer:
    return CharTokenizer(text)


@pytest.fixture(scope="session")
def tiny_corpora():
    d = tempfile.mkdtemp(prefix="dl_test_")
    return build_corpora(d, 8000, 2000, 5, 6)


@pytest.fixture
def tiny_cfg(tiny_corpora) -> ModelConfig:
    return ModelConfig(
        vocab_size=tiny_corpora["vocab_size"],
        n_layer=1, d_model=64, n_head=4,
        context_length=16, ffn_mult=2, dropout=0.0,
    )


@pytest.fixture
def tiny_model(tiny_cfg) -> Transformer:
    torch.manual_seed(DEFAULT_SEED)
    return Transformer(tiny_cfg)


@pytest.fixture
def toy_batch(tiny_corpora, tiny_cfg):
    from src.dataset import make_window_batch
    return make_window_batch(tiny_corpora["train_ids"], tiny_cfg.context_length,
                             0, DEFAULT_SEED, count=8)