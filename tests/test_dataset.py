import torch

from src.dataset import (build_corpora, generate_corpus_text, get_eval_starts,
                         make_window_batch)
from src.tokenizer import CharTokenizer


def test_corpus_deterministic():
    a, b = generate_corpus_text(99, 5000), generate_corpus_text(99, 5000)
    assert a == b
    assert len(a) >= 5000


def test_same_length_different_seed():
    a, b = generate_corpus_text(1, 5000), generate_corpus_text(2, 5000)
    assert a != b


def test_build_corpora_shapes(tiny_corpora):
    assert tiny_corpora["train_ids"].dtype == torch.long
    assert tiny_corpora["val_ids"].dtype == torch.long
    assert tiny_corpora["n_train_tokens"] == tiny_corpora["train_ids"].numel()
    assert tiny_corpora["n_val_tokens"] == tiny_corpora["val_ids"].numel()
    assert tiny_corpora["vocab_size"] == len(tiny_corpora["tokenizer"].itos)


def test_build_corpora_caches(tmp_path):
    out1 = build_corpora(str(tmp_path), 4000, 1000, 5, 6)
    out2 = build_corpora(str(tmp_path), 4000, 1000, 5, 6)
    assert torch.equal(out1["train_ids"], out2["train_ids"])
    assert torch.equal(out1["val_ids"], out2["val_ids"])


def test_tokenizer_loaded_from_cache(tiny_corpora):
    assert isinstance(tiny_corpora["tokenizer"], CharTokenizer)
    assert tiny_corpora["vocab_size"] == tiny_corpora["tokenizer"].vocab_size


def test_window_batch_shapes_and_offsets(tiny_corpora):
    ctx = 16
    x, y = make_window_batch(tiny_corpora["train_ids"], ctx, 7, seed=1, count=8)
    assert x.shape == (8, ctx) and y.shape == (8, ctx)
    assert torch.equal(x[:, 1:], y[:, :-1])     # shifted-by-one target alignment
    assert (x >= 0).all() and (x < tiny_corpora["vocab_size"]).all()
    assert (y >= 0).all() and (y < tiny_corpora["vocab_size"]).all()


def test_window_batch_deterministic(tiny_corpora):
    a = make_window_batch(tiny_corpora["train_ids"], 16, 7, seed=3, count=8)
    b = make_window_batch(tiny_corpora["train_ids"], 16, 7, seed=3, count=8)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_window_batch_no_boundary_overflow():
    # regression: max_start must leave room for the +1 target shift so that
    # ids[pos + 1] never indexes past the corpus (latent Phase-1/2 bug).
    ids = torch.arange(17)  # 17 tokens, ctx 16 -> only valid start is 0
    x, y = make_window_batch(ids, 16, 0, seed=1, count=2)
    assert x.max() <= 15 and y.max() <= 16
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_window_batch_resume_consistent(tiny_corpora):
    # mirror of the train loop: per-micro-step index only depends on step/seed
    ctx = 16
    idx = 3 * 2 + 1  # step 3, micro-batch 1
    xa = make_window_batch(tiny_corpora["train_ids"], ctx, idx, seed=11, count=4)[0]
    xb = make_window_batch(tiny_corpora["train_ids"], ctx, idx, seed=11, count=4)[0]
    assert torch.equal(xa, xb)


def test_eval_starts_bounds_and_order():
    starts = get_eval_starts(n_tokens=1000, context_length=16, max_windows=64)
    assert len(starts) <= 64
    assert starts == sorted(starts)
    assert all(0 <= s <= 1000 - 16 for s in starts)


def test_build_corpora_respects_force(tmp_path):
    import os

    p = str(tmp_path)
    build_corpora(p, 4000, 1000, 5, 6)
    size_bytes = os.path.getsize(os.path.join(p, "train.txt"))
    build_corpora(p, 4000, 1000, 5, 6, force=True)
    assert os.path.exists(os.path.join(p, "train.txt"))
    assert os.path.getsize(os.path.join(p, "train.txt")) == size_bytes