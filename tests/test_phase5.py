"""Phase-5 tests.

Required categories: 1 meta-train/val/test separation, 2 unseen-seed eval,
3 unseen-corpus eval, 4 no-future-information, 5 structured generator dims,
6 low-rank reconstruction, 7 objective calculation, 8 oracle-improvement metric,
9 update cosine similarity, 10 checkpoint save/load, 11 deterministic eval,
12 compute accounting, 13 CPU resource limits.
"""
import tempfile

import torch

from src.dataset import build_corpora
from src.utils import Config, configure_threads
from src.phase5.config import Phase5Config
from src.phase5.features import feature_dim
from src.phase5 import evaluate as ev
from src.phase5.generator import StructuredGenerator, train_generator
from src.phase5.run import build_examples, save_gen, load_gen

SMALL_STEPS = 10
K = 2
H = 6
RECORD_STEPS = [2, 6, 10]
K_MAX = 6


def _tiny_setup():
    cfg = Config()
    cfg.d_model = 64
    cfg.n_layer = 1
    cfg.context_length = 32
    cfg.ffn_mult = 2
    cfg.corpus_train_chars = 4000
    cfg.corpus_val_chars = 1500
    cfg.batch_size = 4
    cfg.n_steps = SMALL_STEPS
    cfg.eval_max_windows = 64
    cfg.data_dir = tempfile.mkdtemp(prefix="dlp5_")
    corpora = build_corpora(cfg.data_dir, cfg.corpus_train_chars,
                            cfg.corpus_val_chars, cfg.corpus_seed, cfg.corpus_val_seed)
    return cfg, corpora


def _tiny_traj(seed=10):
    cfg, corpora = _tiny_setup()
    from src.phase3.trajectory import generate_trajectory
    return generate_trajectory(cfg, corpora, seed, SMALL_STEPS, RECORD_STEPS,
                               K_MAX, eval_max_windows=64)


def _segs(rec):
    return [{"name": n, "out": int(t.shape[0]),
             "in": int(t.shape[1]) if t.ndim == 2 else int(t.numel()),
             "ndim": 2 if t.ndim == 2 else 1}
            for n, t in sorted(rec["w_states"][0].items())]


def _tiny_gen(rec, rank=2, steps=20):
    cfg, corpora = _tiny_setup()
    p5 = Phase5Config()
    p5.max_steps = SMALL_STEPS
    p5.k_max = K_MAX
    segs = _segs(rec)
    X, idx, targets = build_examples({10: rec}, p5, cfg, K, H, segs)
    g, hist, _ = train_generator(X, idx, targets, segs, rank,
                                 feature_dim("full", K), hidden=8, latent_dim=8,
                                 layer_emb_dim=4, m_basis=8, steps=steps,
                                 lr=1e-2, weight_decay=0.0, batch_size=4,
                                 seed=0, objective="recon")
    return g, segs


# 1. meta-train/val/test separation
def test_meta_split_disjoint():
    tr = {s: _tiny_traj(s) for s in (10, 11)}
    va = {s: _tiny_traj(s) for s in (20,)}
    te = {s: _tiny_traj(s) for s in (30,)}
    assert set(tr) & set(va) == set(tr) & set(te) == set()
    cfg, _ = _tiny_setup()
    p5 = Phase5Config(); p5.max_steps = SMALL_STEPS; p5.k_max = K_MAX
    segs = _segs(_tiny_traj(10))
    Xt, it, tt = build_examples(tr, p5, cfg, K, H, segs)
    Xv, iv, tv = build_examples(va, p5, cfg, K, H, segs)
    assert Xt.shape[0] == 2 * len(segs) and Xv.shape[0] == len(segs)


# 2. unseen-seed evaluation
def test_unseen_seed_eval():
    rec = _tiny_traj(30)  # unseen seed
    cfg, corpora = _tiny_setup()
    from src.phase3.evaluate import Phase3Eval
    eh = Phase3Eval(cfg, corpora)
    g, segs = _tiny_gen(_tiny_traj(10))
    r = ev.learned_result(g, rec, K, H, 2, segs, eh, "full", cfg.lr)
    assert r["quality"]["val_loss"] > 0
    assert "cos_pred_target" in r


# 3. unseen-corpus evaluation
def test_unseen_corpus_eval():
    cfg, _ = _tiny_setup()
    from src.dataset import WORD_VOCAB
    from src.tokenizer import CharTokenizer
    # build corpus-A tokenizer over a text guaranteed to cover the full alphabet
    text_a = (" ".join(WORD_VOCAB) + " .") * 30
    tok_a = CharTokenizer(text_a)
    from src.phase5.corpus2 import build_corpus_b
    b_dir = tempfile.mkdtemp(prefix="dlp5b_")
    corpora_b = build_corpus_b(b_dir, tok_a, 4000, 1500, 5, 6)
    assert corpora_b["vocab_size"] == tok_a.vocab_size
    cfg.data_dir = tempfile.mkdtemp(prefix="dlp5b_")
    from src.phase3.trajectory import generate_trajectory
    rec_b = generate_trajectory(cfg, corpora_b, 70, SMALL_STEPS, RECORD_STEPS,
                                K_MAX, eval_max_windows=64)
    from src.phase3.evaluate import Phase3Eval
    eh_b = Phase3Eval(cfg, corpora_b)
    g, segs = _tiny_gen(_tiny_traj(10))
    r = ev.learned_result(g, rec_b, K, H, 2, segs, eh_b, "full", cfg.lr)
    assert r["quality"]["val_loss"] > 0


# 4. no-future-information enforcement
def test_no_future_info():
    import copy
    rec = _tiny_traj()
    g, segs = _tiny_gen(rec, rank=2)
    rec2 = copy.deepcopy(rec)
    for s in range(K + 1, K_MAX + 1):
        for sg in segs:
            rec2["grad_states"][s][sg["name"]] = rec2["grad_states"][s][sg["name"]] * 1e6
    rec2["w_states"][H] = {n: t * 0.0 for n, t in rec2["w_states"][H].items()}
    from src.phase5.generator import generate_deltas
    d1 = generate_deltas(g, rec, K, 2, segs, "full", 3e-4)
    d2 = generate_deltas(g, rec2, K, 2, segs, "full", 3e-4)
    for n in d1:
        assert torch.allclose(d1[n], d2[n], atol=1e-9)


# 5. structured generator dims
def test_generator_dims():
    rec = _tiny_traj()
    g, segs = _tiny_gen(rec, rank=2)
    layers = [s["name"] for s in segs]
    feats = torch.stack([torch.randn(feature_dim("full", K)) for _ in layers])
    idx = torch.arange(len(layers), dtype=torch.long)
    z, coefs = g(feats, idx)
    assert z.shape[1] == g.latent_dim
    assert coefs.shape[1] == 2 * g.m_basis * 2
    ds = g.deltas_for(coefs, idx)
    for i, s in enumerate(segs):
        if s["ndim"] == 2:
            assert ds[i].shape == (s["out"], s["in"])
        else:
            assert ds[i].shape == (s["in"],)


# 6. low-rank reconstruction
def test_lowrank_reconstruction():
    rec = _tiny_traj()
    g, segs = _tiny_gen(rec, rank=2)
    two_d = [s for s in segs if s["ndim"] == 2][0]
    m, r = g.m_basis, 2
    coefs = torch.randn(2 * m * r)
    ds = g.delta_for_one(coefs, segs.index(two_d))
    assert torch.linalg.matrix_rank(ds) <= r


# 7. objective calculation
def test_objective_calculation():
    rec = _tiny_traj()
    g, segs = _tiny_gen(rec, rank=2)
    from src.phase5.generator import recon_loss
    layers = [s["name"] for s in segs]
    feats = torch.stack([torch.zeros(feature_dim("full", K)) for _ in layers])
    idx = torch.arange(len(layers), dtype=torch.long)
    _, coefs = g(feats, idx)
    ds = g.deltas_for(coefs, idx)
    loss = recon_loss(ds, ds)  # perfect reconstruction -> ~0
    assert loss.item() < 1e-6


# 8. oracle-improvement metric
def test_pct_oracle_recovered():
    assert ev.pct_oracle_recovered(2.0, 1.6, 1.4) == pytest_approx(66.666, 0.01)
    assert ev.pct_oracle_recovered(2.0, 1.4, 1.4) == 100.0
    assert ev.pct_oracle_recovered(2.0, 2.0, 1.4) == 0.0


# 9. update cosine similarity
def test_update_cosine():
    rec = _tiny_traj()
    cfg, corpora = _tiny_setup()
    from src.phase3.evaluate import Phase3Eval
    eh = Phase3Eval(cfg, corpora)
    g, segs = _tiny_gen(rec, rank=2)
    r = ev.learned_result(g, rec, K, H, 2, segs, eh, "full", cfg.lr)
    assert -1.0 <= r["cos_pred_target"] <= 1.0
    assert -1.0 <= r["cos_gradmean_target"] <= 1.0


# 10. checkpoint save/load
def test_checkpoint_save_load(tmp_path):
    rec = _tiny_traj()
    g, segs = _tiny_gen(rec, rank=2, steps=10)
    p5 = Phase5Config()
    path = str(tmp_path / "g.pt")
    save_gen(g, path, p5, 2, feature_dim("full", K), "combined", 0.1,
             history=[{"step": 0}])
    g2, hist = load_gen(path, p5, segs, 2, feature_dim("full", K))
    assert len(hist) == 1
    from src.phase5.generator import generate_deltas
    d1 = generate_deltas(g, rec, K, 2, segs, "full", 3e-4)
    d2 = generate_deltas(g2, rec, K, 2, segs, "full", 3e-4)
    for n in d1:
        assert torch.allclose(d1[n], d2[n], atol=1e-6)


# 11. deterministic evaluation
def test_deterministic_eval():
    rec = _tiny_traj()
    cfg, corpora = _tiny_setup()
    from src.phase3.evaluate import Phase3Eval
    eh = Phase3Eval(cfg, corpora)
    m1 = eh.eval_state(rec["w_states"][H])
    m2 = eh.eval_state(rec["w_states"][H])
    assert m1["val_loss"] == m2["val_loss"]


# 12. compute accounting
def test_compute_accounting():
    rec = _tiny_traj()
    segs = _segs(rec)
    c = ev.generator_compute(rec, K, H, 2, 20, 8, 8, 4, 32, len(segs), 1000, segs)
    assert c["observation_fwd_bwd"] == K
    assert c["steps_saved_vs_conventional"] == H - K
    assert c["direct_total_flops"] > 0
    assert c["conventional_total_flops"] > c["direct_total_flops"]
    assert c["uv_generation_flops"] > 0


# 13. CPU resource limits
def test_cpu_resource_limits():
    configure_threads(4)
    assert torch.get_num_threads() == 4


def pytest_approx(v, rel):
    import pytest
    return pytest.approx(v, rel=rel)
