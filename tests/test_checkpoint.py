import os

import torch
from torch.optim import AdamW

from src.checkpoint import (load_resume, load_trajectory_step,
                            load_trajectory_step_last, save_resume,
                            save_trajectory_step)
from src.model import Transformer


def _allclose_state(a, b):
    assert set(a.keys()) == set(b.keys())
    for k in a:
        assert torch.equal(a[k], b[k]), f"tensor mismatch at {k}"


def test_checkpoint_roundtrip(tiny_model: Transformer, toy_batch, tmp_path):
    p = str(tmp_path / "step_0000.pt")
    metrics = {"step": 0, "train_loss": 1.2}
    path = save_trajectory_step(os.path.dirname(p), 0, tiny_model, metrics)
    assert path == p
    loaded = load_trajectory_step(p)
    assert loaded["step"] == 0
    assert loaded["metrics"]["train_loss"] == 1.2
    _allclose_state(loaded["model"].state_dict(), tiny_model.state_dict())


def test_loaded_model_forward_identical(tiny_model: Transformer, toy_batch,
                                        tmp_path):
    p = str(tmp_path / "step_0000.pt")
    save_trajectory_step(str(tmp_path), 0, tiny_model, {"step": 0})
    loaded = load_trajectory_step(p)["model"]
    x, _ = toy_batch
    assert torch.equal(loaded(x), tiny_model(x))


def test_arch_saved_and_recover(tiny_model: Transformer, tmp_path):
    p = str(tmp_path / "step_0000.pt")
    save_trajectory_step(str(tmp_path), 0, tiny_model, {"step": 0})
    payload = torch.load(p, map_location="cpu", weights_only=False)
    assert payload["arch"] == tiny_model.cfg.to_dict()


def test_load_last_finds_max(tiny_model: Transformer, tmp_path):
    save_trajectory_step(str(tmp_path), 0, tiny_model, {"step": 0})
    save_trajectory_step(str(tmp_path), 5, tiny_model, {"step": 5})
    save_trajectory_step(str(tmp_path), 10, tiny_model, {"step": 10})
    last = load_trajectory_step_last(str(tmp_path))
    assert last["step"] == 10


def test_resume_roundtrip(tiny_model: Transformer, toy_batch, tmp_path):
    x, y = toy_batch
    opt1 = AdamW(tiny_model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt1, lambda _: 1.0)
    opt1.zero_grad()
    tiny_model.loss(tiny_model(x), y).backward()
    opt1.step()
    p = str(tmp_path / "resume.pt")
    save_resume(p, tiny_model, opt1, sched, step=3, seed=7)
    assert os.path.exists(p)

    torch.manual_seed(99)
    from src.model import ModelConfig
    fresh = Transformer(ModelConfig.from_dict(tiny_model.cfg.to_dict()))
    opt2 = AdamW(fresh.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda _: 1.0)
    info = load_resume(p, fresh, opt2, sched2)
    assert info["step"] == 3 and info["seed"] == 7
    _allclose_state(fresh.state_dict(), tiny_model.state_dict())
    assert _state_hash(opt1.state) == _state_hash(opt2.state)


def _state_hash(state):
    out = []
    for v in state["state"].values():
        for t in v.values():
            if isinstance(t, torch.Tensor):
                out.append(float(t.double().sum().item()))
    return tuple(out)