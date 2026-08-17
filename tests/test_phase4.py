"""Phase-4 tests.

Required categories: 1 low-rank update construction, 2 generated U/V dims,
3 parameter update correctness, 4 oracle SVD reconstruction, 5 reconstruction
loss, 6 no-future-information, 7 meta-train/test separation, 8 checkpoint
save/load, 9 deterministic evaluation, 10 compute accounting, 11 baseline
comparison, 12 CPU resource limits.
"""
import tempfile

import torch

from src.dataset import build_corpora
from src.utils import Config, configure_threads
from src.phase4.config import Phase4Config
from src.phase4.features import build_features, feature_dim
from src.phase4.operator import (UpdateOperator, build_segments, delta_from_row,
                                 offsets_of, recon_loss, total_gen,
                                 train_operator)
from src.phase4.oracle import delta_target, explained_energy, oracle_deltas
from src.phase4.evaluate import (conventional_result, learned_direct_compute,
                                 learned_result, no_update_result,
                                 operator_train_flops)
from src.phase4.run import build_examples, save_operator, load_operator

SMALL_STEPS = 10
SMALL_K = 2
SMALL_H = 6
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
    cfg.data_dir = tempfile.mkdtemp(prefix="dlp4_")
    corpora = build_corpora(cfg.data_dir, cfg.corpus_train_chars,
                            cfg.corpus_val_chars, cfg.corpus_seed, cfg.corpus_val_seed)
    return cfg, corpora


def _tiny_traj(seed=10):
    cfg, corpora = _tiny_setup()
    from src.phase3.trajectory import generate_trajectory
    return generate_trajectory(cfg, corpora, seed, SMALL_STEPS, RECORD_STEPS,
                               K_MAX, eval_max_windows=64)


def _tiny_operator(rec, rank=2, steps=20, seed=0):
    cfg, corpora = _tiny_setup()
    p4 = Phase4Config()
    p4.max_steps = SMALL_STEPS
    p4.k_max = K_MAX
    segs = build_segments(rec["w_states"][0])
    X, idx, targets = build_examples({10: rec}, p4, cfg, SMALL_K, SMALL_H, segs)
    op, _ = train_operator(X, idx, targets, segs, rank,
                           feature_dim("full", SMALL_K), hidden=8,
                           layer_emb_dim=4, steps=steps, lr=1e-2,
                           batch_size=4, seed=seed)
    return op, segs


# ---------------------------------------------------------------------------
# 1. low-rank update construction
# ---------------------------------------------------------------------------
def test_lowrank_update_construction():
    rec = _tiny_traj()
    segs = build_segments(rec["w_states"][0])
    two_d = [s for s in segs if s["ndim"] == 2][0]
    out, in_ = two_d["out"], two_d["in"]
    r = 2
    U = torch.randn(out, r)
    V = torch.randn(in_, r)
    delta = U @ V.t()
    assert delta.shape == (out, in_)
    assert torch.linalg.matrix_rank(delta) <= r


# ---------------------------------------------------------------------------
# 2. generated U/V dimensions
# ---------------------------------------------------------------------------
def test_generated_uv_dims():
    rec = _tiny_traj()
    op, segs = _tiny_operator(rec, rank=2)
    rank = 2
    offs = offsets_of(segs, rank)
    layers = [s["name"] for s in segs]
    feats = torch.stack([build_features(rec, SMALL_K, l, "full", 3e-4)
                         for l in layers])
    idx = torch.arange(len(layers), dtype=torch.long)
    full = op(feats, idx)
    assert full.shape == (len(layers), total_gen(segs, rank))
    for i, s in enumerate(segs):
        delta = delta_from_row(full[i], i, segs, offs, rank)
        if s["ndim"] == 2:
            assert delta.shape == (s["out"], s["in"])
        else:
            assert delta.shape == (s["in"],)


# ---------------------------------------------------------------------------
# 3. parameter update correctness
# ---------------------------------------------------------------------------
def test_parameter_update_correctness():
    rec = _tiny_traj()
    from src.phase4.operator import generate_deltas
    op, segs = _tiny_operator(rec, rank=2)
    deltas = generate_deltas(op, rec, SMALL_K, 2, segs, "full", 3e-4)
    w_pred = {name: rec["w_states"][SMALL_K][name] + deltas[name]
              for name in deltas}
    for name in deltas:
        assert torch.allclose(w_pred[name],
                              rec["w_states"][SMALL_K][name] + deltas[name])


# ---------------------------------------------------------------------------
# 4. oracle SVD reconstruction
# ---------------------------------------------------------------------------
def test_oracle_svd_reconstruction():
    rec = _tiny_traj()
    segs = build_segments(rec["w_states"][0])
    delta = delta_target(rec, SMALL_K, SMALL_H, [s["name"] for s in segs])
    r2 = oracle_deltas(delta, segs, 2)
    r4 = oracle_deltas(delta, segs, 4)
    assert explained_energy(delta, r4, segs) >= explained_energy(delta, r2, segs)
    # 1-D layers reconstructed exactly
    for s in segs:
        if s["ndim"] == 1:
            assert torch.allclose(r2[s["name"]], delta[s["name"]])


def test_oracle_svd_exact_for_rank():
    # synthetic rank-2 matrix: rank-2 SVD reconstructs exactly
    A = torch.randn(20, 10)
    B = torch.randn(10, 8)
    M = A[:, :2] @ B[:2, :]
    u, sv, v = torch.linalg.svd(M, full_matrices=False)
    recon = (u[:, :2] * sv[:2]) @ v[:2, :]
    assert torch.allclose(recon, M, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. reconstruction loss
# ---------------------------------------------------------------------------
def test_recon_loss():
    rec = _tiny_traj()
    op, segs = _tiny_operator(rec, rank=2)
    rank = 2
    offs = offsets_of(segs, rank)
    layers = [s["name"] for s in segs]
    feats = torch.stack([build_features(rec, SMALL_K, l, "full", 3e-4)
                         for l in layers])
    idx = torch.arange(len(layers), dtype=torch.long)
    full = op(feats, idx)
    targets = [delta_from_row(full[i], i, segs, offs, rank) for i in range(len(layers))]
    loss = recon_loss(full, idx, segs, offs, rank, targets)
    assert loss.item() < 1e-4


# ---------------------------------------------------------------------------
# 6. no-future-information enforcement
# ---------------------------------------------------------------------------
def test_no_future_information():
    import copy
    rec = _tiny_traj()
    op, segs = _tiny_operator(rec, rank=2)
    rec2 = copy.deepcopy(rec)
    for s in range(SMALL_K + 1, K_MAX + 1):
        for l in segs:
            rec2["grad_states"][s][l["name"]] = \
                rec2["grad_states"][s][l["name"]] * 1e6
    from src.phase4.operator import generate_deltas
    d1 = generate_deltas(op, rec, SMALL_K, 2, segs, "full", 3e-4)
    d2 = generate_deltas(op, rec2, SMALL_K, 2, segs, "full", 3e-4)
    for name in d1:
        assert torch.allclose(d1[name], d2[name], atol=1e-9), \
            "operator output must not depend on steps > K"


# ---------------------------------------------------------------------------
# 7. meta-train/test separation
# ---------------------------------------------------------------------------
def test_meta_split_disjoint():
    cfg, corpora = _tiny_setup()
    p4 = Phase4Config()
    p4.max_steps = SMALL_STEPS
    p4.k_max = K_MAX
    segs = build_segments(_tiny_traj(10)["w_states"][0])
    train_recs = {s: _tiny_traj(s) for s in (10, 11)}
    val_recs = {s: _tiny_traj(s) for s in (20,)}
    assert set(train_recs) & set(val_recs) == set()
    Xt, it, tt = build_examples(train_recs, p4, cfg, SMALL_K, SMALL_H, segs)
    Xv, iv, tv = build_examples(val_recs, p4, cfg, SMALL_K, SMALL_H, segs)
    assert Xt.shape[0] == 2 * len(segs)
    assert Xv.shape[0] == len(segs)
    assert Xt.shape[1] == feature_dim("full", SMALL_K)


# ---------------------------------------------------------------------------
# 8. checkpoint save/load
# ---------------------------------------------------------------------------
def test_checkpoint_save_load(tmp_path):
    rec = _tiny_traj()
    op, segs = _tiny_operator(rec, rank=2, seed=1)
    path = str(tmp_path / "op.pt")
    save_operator(op, path, segs, 2, feature_dim("full", SMALL_K), 8, 4,
                  history=[{"step": 0, "train_rel_mse": 0.5}])
    op2, hist = load_operator(path, 2, segs)
    assert len(hist) == 1 and hist[0]["train_rel_mse"] == 0.5
    from src.phase4.operator import generate_deltas
    d1 = generate_deltas(op, rec, SMALL_K, 2, segs, "full", 3e-4)
    d2 = generate_deltas(op2, rec, SMALL_K, 2, segs, "full", 3e-4)
    for name in d1:
        assert torch.allclose(d1[name], d2[name], atol=1e-6)


# ---------------------------------------------------------------------------
# 9. deterministic evaluation
# ---------------------------------------------------------------------------
def test_deterministic_evaluation():
    rec = _tiny_traj()
    cfg, corpora = _tiny_setup()
    from src.phase3.evaluate import Phase3Eval
    eval_h = Phase3Eval(cfg, corpora)
    m1 = eval_h.eval_state(rec["w_states"][SMALL_H])
    m2 = eval_h.eval_state(rec["w_states"][SMALL_H])
    assert m1["val_loss"] == m2["val_loss"]


# ---------------------------------------------------------------------------
# 10. compute accounting
# ---------------------------------------------------------------------------
def test_compute_accounting():
    rec = _tiny_traj()
    segs = build_segments(rec["w_states"][0])
    c = learned_direct_compute(rec, SMALL_K, SMALL_H, 2, 20, 8, 4,
                               total_gen(segs, 2), len(segs), 1000, segs)
    assert c["observation_fwd_bwd"] == SMALL_K
    assert c["steps_saved_vs_conventional"] == SMALL_H - SMALL_K
    assert c["direct_total_flops"] > c["observation_flops"]
    assert c["conventional_total_flops"] > c["direct_total_flops"]
    assert operator_train_flops(10, 20, 20, 8, 4, total_gen(segs, 2)) > 0
    assert c["generation_flops"] > 0


# ---------------------------------------------------------------------------
# 11. baseline comparison
# ---------------------------------------------------------------------------
def test_baseline_comparison():
    rec = _tiny_traj()
    cfg, corpora = _tiny_setup()
    from src.phase3.evaluate import Phase3Eval
    eval_h = Phase3Eval(cfg, corpora)
    no = no_update_result(rec, SMALL_K, eval_h)
    conv = conventional_result(rec, SMALL_K, SMALL_H)
    assert no["quality"]["val_loss"] == rec["val_records"][SMALL_K]["loss"]
    assert conv["quality"]["val_loss"] == rec["val_records"][SMALL_H]["loss"]
    op, segs = _tiny_operator(rec, rank=2)
    lr = learned_result(rec, SMALL_K, SMALL_H, 2, op, segs, eval_h, "full", cfg.lr)
    assert lr["kind"] == "learned"
    assert lr["quality"]["val_loss"] > 0
    assert lr["rel_param_dist_to_WH"] >= 0
    assert lr["update_norm"] >= 0


# ---------------------------------------------------------------------------
# 12. CPU resource limits
# ---------------------------------------------------------------------------
def test_cpu_resource_limits():
    configure_threads(4)
    assert torch.get_num_threads() == 4
    assert torch.get_num_threads() <= 4
