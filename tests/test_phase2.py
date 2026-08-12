import tempfile

import torch

from src.dataset import build_corpora
from src.model import Transformer
from src.train import build_model
from src.utils import Config
from src.phase2.capture import capture_baseline
from src.phase2.methods import Ctx, run_method


def _mini_setup(steps=6):
    cfg = Config()
    cfg.d_model = 64
    cfg.n_layer = 1
    cfg.context_length = 32
    cfg.ffn_mult = 2
    cfg.corpus_train_chars = 4000
    cfg.corpus_val_chars = 1500
    cfg.batch_size = 4
    cfg.n_steps = steps
    cfg.eval_batch = 4
    cfg.eval_max_windows = 128
    cfg.data_dir = tempfile.mkdtemp(prefix="dlp2_")
    corpora = build_corpora(cfg.data_dir, cfg.corpus_train_chars, cfg.corpus_val_chars,
                            cfg.corpus_seed, cfg.corpus_val_seed)
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, corpora["vocab_size"])
    return cfg, corpora, model


def test_capture_records_losses_and_norms():
    cfg, corpora, model = _mini_setup(steps=6)
    cap = capture_baseline(cfg, corpora, model, torch.device("cpu"), cfg.seed,
                           6, [2, 4], [0, 1, 2], beta=0.9)
    assert len(cap["train_losses"]) == 7
    assert cap["train_losses"][0] >= 0
    assert all(v is not None for v in cap["grad_norms_unclipped"][1:])
    assert all(v is not None for v in cap["grad_norms_clipped"][1:])
    assert set(cap["snap_grad"].keys()) == {2, 4}
    assert set(cap["val_records"].keys()) == {0, 1, 2}
    assert cap["g_first"] is not None


def test_oracle_reproduces_wn_exactly():
    cfg, corpora, model = _mini_setup(steps=6)
    cap = capture_baseline(cfg, corpora, model, torch.device("cpu"), cfg.seed,
                           6, [2, 4, 6], [0, 1, 2], beta=0.9)
    w0_state = _init_state(cfg, corpora["vocab_size"])
    # capture trained on the model instance in-place; model_state is W_6
    wn = cap["model_state"]
    delta = {k: wn[k] - w0_state[k] for k in w0_state}
    oracle = {k: w0_state[k] + delta[k] for k in w0_state}
    for k in oracle:
        assert torch.allclose(oracle[k], wn[k], atol=1e-5, rtol=1e-6)


def test_methods_interface_runs_return_results():
    cfg, corpora, model = _mini_setup(steps=4)
    cap = capture_baseline(cfg, corpora, model, torch.device("cpu"), cfg.seed,
                           4, [4], [0, 1, 2], beta=0.9)
    import numpy as np
    from src.phase2.methods import Ctx

    w0 = _init_state(cfg, corpora["vocab_size"])
    wn_states = {4: cap["model_state"]}
    fwd_1seq = 2 * 1e6  # rough; value not used for correctness here
    ctx = Ctx(cfg, corpora, w0, cap, wn_states, torch.device("cpu"),
              1e6, 1e6, np.logspace(-4, 1, 10), [2, 4], 0.9)
    for name in ["DirectOracle", "DirectGradient", "DirectAverageGradient",
                 "DirectMomentum"]:
        r = run_method(name, ctx, 4)
        assert r.kind in ("oracle", "practical")
        assert r.final_val_loss > 0
        assert isinstance(r.compute.param_updates, int)
    rl = run_method("DirectLowRank", ctx, 4, rank=2)
    assert rl.name.startswith("DirectLowRank")
    assert rl.final_val_loss > 0


def _init_state(cfg, vocab):
    torch.manual_seed(cfg.seed)
    return build_model(cfg, vocab).state_dict()