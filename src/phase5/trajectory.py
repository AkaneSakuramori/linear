"""Phase-5 trajectory generation and caching.

Reuses the Phase-3 recipe (same corpus, same window sampling, same clipping)
but supports *decoupled* initialization and data-sampling seeds so the
generalization tests can isolate:
  B — unseen batch ordering (seen init, unseen data seed)
  C — unseen initialization (unseen init seed, seen data seed)
Trajectories are cached under results/phase5/checkpoints/.
"""
from __future__ import annotations

import os

import torch

from src.dataset import make_window_batch
from src.evaluate import evaluate_loss
from src.train import build_model, build_scheduler
from src.utils import estimate_flops, set_seed
from src.phase3.trajectory import global_grad_norm


def generate_trajectory_decoupled(phase1_cfg, corpora, init_seed: int,
                                  data_seed: int, max_steps: int,
                                  record_steps, k_max: int,
                                  eval_max_windows: int = 1024,
                                  device=None) -> dict:
    """Run one AdamW trajectory with explicit (init, data) seeds."""
    import time
    import resource
    from torch.optim import AdamW

    if device is None:
        device = torch.device("cpu")
    t0 = time.time()
    train_ids = corpora["train_ids"].to(device)
    val_ids = corpora["val_ids"].to(device)
    vocab = corpora["vocab_size"]

    set_seed(init_seed)                       # initialization
    model = build_model(phase1_cfg, vocab).to(device)
    optimizer = AdamW(model.parameters(), lr=phase1_cfg.lr,
                      weight_decay=phase1_cfg.weight_decay, betas=tuple(phase1_cfg.betas))
    scheduler = build_scheduler(optimizer, phase1_cfg)

    w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    w_states = {0: w0}
    losses = {}
    val_records = {}
    grad_states = {}
    grad_norms = {}
    param_norms = {}
    adam_stats = {}

    def _pnorm(state):
        tot = 0.0
        for t in state.values():
            tot += float(t.double().pow(2).sum().item())
        return float(torch.sqrt(torch.tensor(tot)).item())

    param_norms[0] = _pnorm(w0)
    v0 = evaluate_loss(model, val_ids, phase1_cfg.context_length,
                       batch_size=phase1_cfg.batch_size, max_windows=eval_max_windows)
    losses[0] = v0["loss"]
    val_records[0] = v0
    record_set = set(record_steps)
    adam_steps = sorted(set(k for k in record_steps if k <= k_max))

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        x, y = make_window_batch(train_ids, phase1_cfg.context_length, step - 1,
                                 data_seed, phase1_cfg.batch_size)
        logits = model(x)
        loss = model.loss(logits, y)
        loss.backward()
        losses[step] = float(loss.item())
        if step <= k_max:
            grad_norms[step] = global_grad_norm(model)
            grad_states[step] = {name: p.grad.detach().clone()
                                 for name, p in model.named_parameters()}
        if phase1_cfg.grad_clip is not None and phase1_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), phase1_cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        if step in record_set:
            w_states[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}
            param_norms[step] = _pnorm(w_states[step])
            val_records[step] = evaluate_loss(
                model, val_ids, phase1_cfg.context_length,
                batch_size=phase1_cfg.batch_size, max_windows=eval_max_windows)
            if step in adam_steps:
                st = {}
                for name, p in model.named_parameters():
                    st[name] = {
                        "m_norm": float(optimizer.state[p]["exp_avg"].double().pow(2).sum().sqrt().item()),
                        "v_norm": float(optimizer.state[p]["exp_avg_sq"].double().pow(2).sum().sqrt().item()),
                    }
                adam_stats[step] = st

    ef_batch = phase1_cfg.batch_size * phase1_cfg.grad_accumulation
    flops_per_step = estimate_flops(model, phase1_cfg.context_length,
                                    max_steps, ef_batch)["approx_flops_per_step"]
    return {
        "seed": data_seed, "init_seed": init_seed, "data_seed": data_seed,
        "w_states": w_states, "losses": losses, "val_records": val_records,
        "grad_states": grad_states, "grad_norms": grad_norms,
        "param_norms": param_norms, "adam_stats": adam_stats,
        "flops_per_step": flops_per_step,
        "tokens_per_step": ef_batch * phase1_cfg.context_length,
        "elapsed_sec": time.time() - t0,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0,
    }


def load_or_generate(phase1_cfg, corpora, seed: int, cache_dir: str, p5,
                     init_seed: int = None, data_seed: int = None) -> dict:
    """Cache-aware trajectory load/generate (standard or decoupled)."""
    if init_seed is not None and data_seed is not None:
        fname = f"traj_i{init_seed}_d{data_seed}.pt"
    else:
        fname = f"traj_{seed}.pt"
    path = os.path.join(cache_dir, fname)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    # reuse a compatible Phase-4 trajectory (record_steps incl. 10/15/25/50/100,
    # k_max=25, max_steps=100) when available
    if init_seed is None and data_seed is None:
        p4 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "results", "phase4", "checkpoints",
            f"trajectory_seed{seed}.pt")
        if os.path.exists(p4):
            rec = torch.load(p4, map_location="cpu", weights_only=False)
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(rec, path)
            return rec
    if init_seed is not None and data_seed is not None:
        rec = generate_trajectory_decoupled(phase1_cfg, corpora, init_seed,
                                            data_seed, p5.max_steps,
                                            p5.record_steps, p5.k_max,
                                            phase1_cfg.eval_max_windows)
    else:
        from src.phase3.trajectory import generate_trajectory
        rec = generate_trajectory(phase1_cfg, corpora, seed, p5.max_steps,
                                  p5.record_steps, p5.k_max,
                                  phase1_cfg.eval_max_windows)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(rec, path)
    return rec
