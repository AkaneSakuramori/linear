"""Deterministic instrumented AdamW trajectory generation for Phase 3.

Replays the exact Phase-1 training recipe (same corpus, window sampling,
clipping, scheduler) for an arbitrary seed, while recording everything the
Direct Update Predictor is allowed to see up to observation horizon K:

  * unclipped per-step gradients for steps 1..K_MAX,
  * loss history and global gradient norms for steps 1..K_MAX,
  * model states W_k at the target horizons and at step 0,
  * Adam optimizer-state norms (exp_avg / exp_avg_sq) at the observation steps,
  * validation loss at every recorded state.

The target ΔW_target = W_H − W_K is *not* part of the returned observation;
it is derived separately by callers (and only used while meta-training the
predictor, never during direct application).
"""
from __future__ import annotations

import resource
import time
from typing import Dict, List

import torch
from torch.optim import AdamW

from src.dataset import make_window_batch
from src.evaluate import evaluate_loss
from src.train import build_model, build_scheduler
from src.utils import estimate_flops, set_seed


def global_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.double().pow(2).sum().item())
    return float(torch.sqrt(torch.tensor(total)).item())


def params_zero_like(model) -> Dict[str, torch.Tensor]:
    return {name: torch.zeros_like(p.data) for name, p in model.named_parameters()}


def generate_trajectory(phase1_cfg, corpora, seed: int, max_steps: int,
                        record_steps: List[int], k_max: int,
                        eval_max_windows: int = 1024, device=None) -> Dict:
    """Run one full AdamW trajectory (seed) and return its observation record."""
    if device is None:
        device = torch.device("cpu")
    t0 = time.time()

    train_ids = corpora["train_ids"].to(device)
    val_ids = corpora["val_ids"].to(device)
    vocab = corpora["vocab_size"]

    # identical to Phase 1: seed before build => seed controls init + batches
    set_seed(seed)
    model = build_model(phase1_cfg, vocab).to(device)
    optimizer = AdamW(model.parameters(), lr=phase1_cfg.lr,
                      weight_decay=phase1_cfg.weight_decay, betas=tuple(phase1_cfg.betas))
    scheduler = build_scheduler(optimizer, phase1_cfg)

    w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    w_states: Dict[int, Dict[str, torch.Tensor]] = {0: w0}
    losses: Dict[int, float] = {}
    val_records: Dict[int, Dict] = {}
    grad_states: Dict[int, Dict[str, torch.Tensor]] = {}
    grad_norms: Dict[int, float] = {}
    param_norms: Dict[int, float] = {}
    adam_stats: Dict[int, Dict[str, Dict[str, float]]] = {}

    def _param_norm(state_dict) -> float:
        tot = 0.0
        for t in state_dict.values():
            tot += float(t.double().pow(2).sum().item())
        return float(torch.sqrt(torch.tensor(tot)).item())

    param_norms[0] = _param_norm(w0)
    v0 = evaluate_loss(model, val_ids, phase1_cfg.context_length,
                       batch_size=phase1_cfg.batch_size, max_windows=eval_max_windows)
    losses[0] = v0["loss"]
    val_records[0] = v0

    record_set = set(record_steps)
    adam_steps = sorted(set(k for k in record_steps if k <= k_max))

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        x, y = make_window_batch(train_ids, phase1_cfg.context_length, step - 1,
                                 seed, phase1_cfg.batch_size)
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
            param_norms[step] = _param_norm(w_states[step])
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
        "seed": seed,
        "w_states": w_states,
        "losses": losses,
        "val_records": val_records,
        "grad_states": grad_states,
        "grad_norms": grad_norms,
        "param_norms": param_norms,
        "adam_stats": adam_stats,
        "flops_per_step": flops_per_step,
        "tokens_per_step": ef_batch * phase1_cfg.context_length,
        "elapsed_sec": time.time() - t0,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0,
    }
