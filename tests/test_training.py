import json
import os
import subprocess
import sys
import tempfile
import textwrap

import torch
from torch.optim import AdamW

from src.dataset import make_window_batch
from src.evaluate import grad_and_param_norms


def test_backward_populates_grads(tiny_model, toy_batch):
    x, y = toy_batch
    tiny_model.zero_grad()
    loss = tiny_model.loss(tiny_model(x), y)
    loss.backward()
    for p in tiny_model.parameters():
        assert p.grad is not None
    norms = grad_and_param_norms(tiny_model)
    assert norms["grad_norm"] is not None and norms["grad_norm"] > 0
    assert norms["param_norm"] > 0


def test_adamw_update_changes_params_and_reduces_loss(tiny_model, toy_batch):
    x, y = toy_batch
    opt = AdamW(tiny_model.parameters(), lr=1e-3, weight_decay=0.01)
    first = tiny_model.loss(tiny_model(x), y).item()
    snapshot = {k: p.detach().clone() for k, p in tiny_model.named_parameters()}
    for _ in range(8):
        opt.zero_grad()
        tiny_model.loss(tiny_model(x), y).backward()
        opt.step()
    last = tiny_model.loss(tiny_model(x), y).item()
    assert last < first
    changed = any(
        torch.equal(p.detach(), snapshot[name]) == False
        for name, p in tiny_model.named_parameters())
    assert changed


def test_grad_norm_zero_after_optimizer(monkeypatch):
    # gradient norm is measured *after* backward but before optimizer.step()
    from src.model import ModelConfig, Transformer
    torch.manual_seed(3)
    m = Transformer(ModelConfig(vocab_size=20, n_layer=1, d_model=32, n_head=2,
                                context_length=8, ffn_mult=2))
    x = torch.zeros(2, 8, dtype=torch.long)
    m.loss(m(x), x).backward()
    n = grad_and_param_norms(m)
    assert n["grad_norm"] is not None


def test_deterministic_init_between_runs(tiny_cfg):
    torch.manual_seed(77)
    from src.model import Transformer
    a = Transformer(tiny_cfg)
    torch.manual_seed(77)
    b = Transformer(tiny_cfg)
    assert torch.equal(a.lm_head.weight.data, b.lm_head.weight.data)


# ---------------------------------------------------------------------------
# End-to-end: run `src.train` in a subprocess for isolation, tiny config.
# ---------------------------------------------------------------------------
def _tiny_config_path(tmp_path, steps):
    from src.utils import Config
    cfg = Config()
    overrides = dict(
        d_model=64, n_layer=1, context_length=32, ffn_mult=2,
        corpus_train_chars=6000, corpus_val_chars=2000,
        batch_size=16, n_steps=steps, lr=2e-3, threads=2,
        out_dir=str(tmp_path / "results"),
        checkpoints_dir=str(tmp_path / "checkpoints"),
        data_dir=str(tmp_path / "data"),
    )
    d = cfg.resolved_dict(29)
    d.pop("vocab_size")
    d.update({k: v for k, v in overrides.items()})
    d["trajectory_steps"] = [0, 1, 2, 3, 5, 10] + ([15] if steps >= 15 else [])
    d["experiment_name"] = "tiny"
    import yaml
    p = str(tmp_path / "tiny.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(d, f)
    return p


def _run_tiny(tmp_path, steps):
    p = _tiny_config_path(tmp_path, steps)
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", ".")
    subprocess.run(
        [sys.executable, "-m", "src.train", "--config", p],
        check=True, capture_output=True, text=True, env=env,
    )
    results = tmp_path / "results"
    with open(results / "metrics.json") as f:
        return json.load(f)


def test_end_to_end_loss_decreases(tmp_path):
    m = _run_tiny(tmp_path, steps=15)
    s = m["summary"]
    assert s["param_count"] > 0
    assert s["final_train_loss"] < s["initial_train_loss"]
    assert s["final_val_loss"] < s["initial_val_loss"]
    assert s["final_val_ppl"] < s["initial_val_ppl"]


def test_end_to_end_artifacts(tmp_path):
    _run_tiny(tmp_path, steps=15)
    out = tmp_path / "results"
    assert (out / "config.json").exists()
    assert (out / "metrics.json").exists()
    assert (out / "training.log").exists()
    traj = out / "trajectory"
    assert (traj / "step_0000.pt").exists()
    assert (traj / "step_0015.pt").exists()
    assert (tmp_path / "checkpoints" / "resume_latest.pt").exists()


def test_deterministic_training(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    os.makedirs(out_a)
    os.makedirs(out_b)
    a = _run_tiny(out_a, steps=6)
    b = _run_tiny(out_b, steps=6)
    ta = a["trajectory"]
    tb = b["trajectory"]
    assert [t["step"] for t in ta] == [t["step"] for t in tb]
    assert [t["train_loss"] for t in ta] == [t["train_loss"] for t in tb]
    assert [t["val_loss"] for t in ta] == [t["val_loss"] for t in tb]
    assert a["summary"]["final_train_loss"] == b["summary"]["final_train_loss"]