"""Phase-3 tests: predictor pipeline, no-future constraint, splits, compute.

Each required test category from the Phase-3 plan maps to a test here:
1 predictor input construction, 2 output shape, 3 delta reconstruction,
4 W_K + ΔW reconstruction, 5 no-future-information, 6 meta-training split,
7 checkpoint save/load, 8 deterministic evaluation, 9 compute accounting,
10 baseline/direct comparison.
"""
import tempfile

import pytest
import torch

from src.dataset import build_corpora
from src.utils import Config
from src.phase3.config import Phase3Config
from src.phase3.features import (apply_prediction, build_basis, build_features,
                                 compute_alpha_star, feature_dim, param_names,
                                 predict_all_layers, reconstruct_layer_delta,
                                 rel_param_distance)
from src.phase3.predictor import PredictorNet, Predictor, train_predictor
from src.phase3.run import build_meta_dataset
from src.phase3.evaluate import (Phase3Eval, direct_compute, direct_result,
                                 conventional_result, predictor_inference_flops,
                                 predictor_train_flops)

SMALL_STEPS = 8
SMALL_K = 2
SMALL_H = 8
RECORD_STEPS = [2, 4, 8]
K_MAX = 4


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
    cfg.data_dir = tempfile.mkdtemp(prefix="dlp3_")
    corpora = build_corpora(cfg.data_dir, cfg.corpus_train_chars,
                            cfg.corpus_val_chars, cfg.corpus_seed, cfg.corpus_val_seed)
    return cfg, corpora


def _tiny_trajectory(seed=10):
    cfg, corpora = _tiny_setup()
    from src.phase3.trajectory import generate_trajectory
    return generate_trajectory(cfg, corpora, seed, SMALL_STEPS, RECORD_STEPS,
                               K_MAX, eval_max_windows=64)


# ---------------------------------------------------------------------------
# 1. predictor input construction
# ---------------------------------------------------------------------------
def test_features_shape_and_finite():
    rec = _tiny_trajectory()
    layer = param_names(rec)[0]
    for fs in ("loss", "grad", "grad_loss", "full", "compressed", "rich"):
        f = build_features(rec, SMALL_K, layer, fs, lr=3e-4)
        assert f.shape == (feature_dim(fs, SMALL_K),)
        assert torch.isfinite(f).all(), f"non-finite features for {fs}"


def test_feature_dim_matches_every_set():
    for K in (2, 4):
        assert feature_dim("loss", K) == K
        assert feature_dim("grad", K) == K + 5
        assert feature_dim("grad_loss", K) == 2 * K + 5
        assert feature_dim("full", K) == 2 * K + 12
        assert feature_dim("compressed", K) == 17
        assert feature_dim("rich", K) == 2 * K + 14


# ---------------------------------------------------------------------------
# 2. predictor output shape
# ---------------------------------------------------------------------------
def test_predictor_output_shape():
    in_dim = feature_dim("full", SMALL_K)
    net = PredictorNet(in_dim, hidden=8, out_dim=2)
    out = net(torch.randn(5, in_dim))
    assert out.shape == (5, 2)
    X = torch.randn(4, in_dim)
    Y = torch.randn(4, 2)
    pred, _ = train_predictor(X, Y, hidden=8, steps=10, batch_size=4, seed=0)
    assert pred.predict(X).shape == (4, 2)


# ---------------------------------------------------------------------------
# 3. parameter delta reconstruction
# ---------------------------------------------------------------------------
def test_delta_reconstruction_with_known_coeffs():
    v1 = torch.randn(10)
    v2 = torch.randn(10)
    target = 2.0 * v1 + 3.0 * v2
    B = torch.stack([v1, v2], dim=1)
    A = B.t() @ B + 1e-6 * torch.eye(2)
    alpha = torch.linalg.solve(A, B.t() @ target)
    recon = reconstruct_layer_delta([v1, v2], alpha)
    assert torch.allclose(recon, target, atol=1e-4)


def test_delta_shape_matches_parameter():
    rec = _tiny_trajectory()
    layer = param_names(rec)[0]
    alpha = compute_alpha_star(rec, SMALL_K, SMALL_H, "mean", layer)
    delta = reconstruct_layer_delta(build_basis(rec, SMALL_K, "mean", layer), alpha)
    assert delta.shape == rec["w_states"][SMALL_K][layer].shape


# ---------------------------------------------------------------------------
# 4. W_K + ΔW reconstruction
# ---------------------------------------------------------------------------
def test_apply_prediction_adds_delta():
    rec = _tiny_trajectory()
    layer = param_names(rec)[0]
    alpha = compute_alpha_star(rec, SMALL_K, SMALL_H, "mean", layer)
    deltas = predict_all_layers(rec, SMALL_K, "mean", {layer: alpha})
    w_pred = apply_prediction(rec["w_states"][SMALL_K], deltas)
    for name in deltas:
        assert torch.equal(w_pred[name], rec["w_states"][SMALL_K][name] + deltas[name])


def test_rel_param_distance_is_key_order_invariant():
    # regression: reconstructed states are keyed in sorted name order while
    # model.state_dict() is in module order; distances must not depend on it.
    rec = _tiny_trajectory()
    w = rec["w_states"][SMALL_H]
    reordered = {name: w[name] for name in reversed(list(w.keys()))}
    assert rel_param_distance(w, w) == 0.0
    assert rel_param_distance(reordered, w) == 0.0
    assert rel_param_distance(rec["w_states"][SMALL_K], w) == \
        rel_param_distance({name: rec["w_states"][SMALL_K][name]
                            for name in reversed(list(rec["w_states"][SMALL_K].keys()))},
                           w)


# ---------------------------------------------------------------------------
# 5. no-future-information constraint
# ---------------------------------------------------------------------------
def test_features_do_not_see_future_steps():
    rec = _tiny_trajectory()
    layer = param_names(rec)[0]
    f_orig = build_features(rec, SMALL_K, layer, "full", lr=3e-4)
    # poison the future (steps > K) of a copy
    import copy
    rec2 = copy.deepcopy(rec)
    for s in range(SMALL_K + 1, K_MAX + 1):
        rec2["grad_states"][s][layer] = rec2["grad_states"][s][layer] * 1e6
        rec2["losses"][s] = rec2["losses"][s] * 1e6
        rec2["grad_norms"][s] = rec2["grad_norms"][s] * 1e6
    rec2["w_states"][SMALL_H][layer] = rec2["w_states"][SMALL_H][layer] * 0.0
    f_poisoned = build_features(rec2, SMALL_K, layer, "full", lr=3e-4)
    assert torch.allclose(f_orig, f_poisoned, atol=1e-8), \
        "features at K must not depend on steps > K"


def test_basis_uses_only_observed_steps():
    rec = _tiny_trajectory()
    layer = param_names(rec)[0]
    basis = build_basis(rec, SMALL_K, "all", layer)
    assert len(basis) == SMALL_K
    assert all(b.shape == rec["grad_states"][1][layer].shape for b in basis)


# ---------------------------------------------------------------------------
# 6. meta-training split
# ---------------------------------------------------------------------------
def test_meta_split_disjoint_and_shapes():
    cfg, corpora = _tiny_setup()
    p3 = Phase3Config()
    p3.max_steps = SMALL_STEPS
    p3.k_max = K_MAX
    train_recs = {}
    val_recs = {}
    for seed in (10, 11):
        train_recs[seed] = _tiny_trajectory(seed)
    for seed in (20,):
        val_recs[seed] = _tiny_trajectory(seed)
    assert set(train_recs) & set(val_recs) == set()

    X, Y, layers = build_meta_dataset(train_recs, p3, cfg, "full", "mean",
                                      SMALL_K, SMALL_H)
    Xv, Yv, _ = build_meta_dataset(val_recs, p3, cfg, "full", "mean",
                                   SMALL_K, SMALL_H)
    assert X.shape[0] == 2 * len(layers)
    assert Xv.shape[0] == 1 * len(layers)
    assert Y.shape[1] == 1  # basis='mean' -> one coefficient per layer
    assert X.shape[1] == feature_dim("full", SMALL_K)


# ---------------------------------------------------------------------------
# 7. checkpoint save/load
# ---------------------------------------------------------------------------
def test_predictor_save_load_roundtrip(tmp_path):
    in_dim = feature_dim("full", SMALL_K)
    X = torch.randn(6, in_dim)
    Y = torch.randn(6, 1)
    pred, _ = train_predictor(X, Y, hidden=8, steps=10, batch_size=6, seed=0)
    path = tmp_path / "pred.pt"
    pred.save(str(path))
    loaded = Predictor.load(str(path), in_dim, 8, 1)
    assert torch.allclose(pred.predict(X), loaded.predict(X), atol=1e-6)


# ---------------------------------------------------------------------------
# 8. deterministic evaluation
# ---------------------------------------------------------------------------
def test_evaluation_is_deterministic():
    rec = _tiny_trajectory()
    cfg, corpora = _tiny_setup()
    eval_h = Phase3Eval(cfg, corpora)
    m1 = eval_h.eval_state(rec["w_states"][SMALL_H])
    m2 = eval_h.eval_state(rec["w_states"][SMALL_H])
    assert m1["val_loss"] == m2["val_loss"]
    assert m1["val_ppl"] == m2["val_ppl"]


# ---------------------------------------------------------------------------
# 9. compute accounting
# ---------------------------------------------------------------------------
def test_compute_accounting_consistency():
    rec = _tiny_trajectory()
    infer = predictor_inference_flops([10, 8, 8, 1])
    assert infer > 0
    tr = predictor_train_flops(10, 4, [10, 8, 8, 1])
    assert tr > infer * 10
    c = direct_compute(rec, SMALL_K, SMALL_H, infer, n_params=1000, eval_flops=5.0)
    assert isinstance(c["observation_fwd_bwd"], int)
    assert c["direct_total_flops"] == pytest.approx(
        c["observation_flops"] + infer + 2000 + 5.0)
    assert c["steps_saved_vs_conventional"] == SMALL_H - SMALL_K
    assert c["conventional_total_flops"] > c["direct_total_flops"]


# ---------------------------------------------------------------------------
# 10. baseline/direct comparison
# ---------------------------------------------------------------------------
def test_baseline_direct_comparison_structure():
    rec = _tiny_trajectory()
    cfg, corpora = _tiny_setup()
    p3 = Phase3Config()
    p3.max_steps = SMALL_STEPS
    p3.k_max = K_MAX
    X, Y, _ = build_meta_dataset({10: rec}, p3, cfg, "full", "mean",
                                 SMALL_K, SMALL_H)
    pred, _ = train_predictor(X, Y, hidden=8, steps=30, batch_size=4, seed=0)
    eval_h = Phase3Eval(cfg, corpora)
    d = direct_result(rec, SMALL_K, SMALL_H, "mean", pred, eval_h, "full", cfg.lr)
    c = conventional_result(rec, SMALL_K, SMALL_H)
    assert d["kind"] == "direct"
    assert c["quality"]["val_loss"] == rec["val_records"][SMALL_H]["loss"]
    assert d["quality"]["val_loss"] > 0
    assert d["rel_param_dist_to_WH"] >= 0
    assert d["rel_delta_error"] >= 0
    assert rel_param_distance(rec["w_states"][SMALL_H], rec["w_states"][SMALL_H]) == 0.0
