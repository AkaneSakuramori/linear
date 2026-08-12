"""Instrumented re-run of the Phase-1 baseline.

Reproduces the exact AdamW training loop (same batches, same clipping, same
schedule) while additionally recording:
  * unclipped vs clipped global gradient norm per step,
  * per-step raw (unclipped) gradients accumulated into an average gradient,
  * a momentum-style EMA of raw gradients,
both snapshotted at the requested horizontal cut-offs (e.g. 5, 10, ... steps).
Nothing is saved to disk here; all captured data is returned in memory.
"""
from __future__ import annotations

import time
from typing import Dict, List
import resource

import torch
from torch.optim import AdamW

from src.dataset import make_window_batch
from src.evaluate import evaluate_loss
from src.train import build_model, build_scheduler
from src.utils import Config, estimate_flops


def global_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.double().pow(2).sum().item())
    return float(torch.sqrt(torch.tensor(total)).item())


def params_zero_like(model) -> Dict[str, torch.Tensor]:
    return {name: torch.zeros_like(p.data) for name, p in model.named_parameters()}


def capture_baseline(cfg: Config, corpora, model, device, seed: int,
                     n_steps: int, horizons: List[int], trajectory_steps: List[int],
                     beta: float = 0.9) -> Dict:
    """Run the baseline; return losses, norms, gradients, momentum, final model."""
    t0 = time.time()
    train_ids = corpora["train_ids"].to(device)
    val_ids = corpora["val_ids"].to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                      betas=tuple(cfg.betas))
    scheduler = build_scheduler(optimizer, cfg)
    tokens_per_step = cfg.batch_size * cfg.grad_accumulation * cfg.context_length

    grad_sum = params_zero_like(model)
    momentum = params_zero_like(model)
    g_first = None
    snap_grad = {}   # horizon -> mean-gradient dict snapshot
    snap_momentum = {}
    grad_norms_unclipped = [None] * (n_steps + 1)
    grad_norms_clipped = [None] * (n_steps + 1)
    train_losses = [None] * (n_steps + 1)
    train_emas = [None] * (n_steps + 1)
    val_records = {}

    ef_batch = cfg.batch_size * cfg.grad_accumulation
    flops_per_step = estimate_flops(model, cfg.context_length, n_steps, ef_batch)["approx_flops_per_step"]

    model.train()
    ema_loss = None
    # w0 recorded as-is (loss at init)
    val0 = evaluate_loss(model, val_ids, cfg.context_length,
                         batch_size=cfg.batch_size, max_windows=cfg.eval_max_windows)
    train_losses[0] = val0["loss"]
    train_emas[0] = val0["loss"]
    val_records[0] = val0

    horizon_idx = 0
    for step in range(1, n_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        x, y = make_window_batch(train_ids, cfg.context_length, step - 1,
                                 seed, cfg.batch_size)
        logits = model(x)
        loss = model.loss(logits, y)
        loss.backward()

        train_losses[step] = float(loss.item())
        step_loss = float(loss.item())
        ema_loss = step_loss if ema_loss is None else ema_loss * 0.9 + step_loss * 0.1
        train_emas[step] = ema_loss

        unclipped = global_grad_norm(model)
        grad_norms_unclipped[step] = unclipped

        if g_first is None:
            g_first = {name: p.grad.detach().clone() for name, p in model.named_parameters()}

        for name, p in model.named_parameters():
            g = p.grad
            if g is not None:
                grad_sum[name].add_(g.detach())
                momentum[name].mul_(beta).add_(g.detach())

        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        grad_norms_clipped[step] = global_grad_norm(model)

        optimizer.step()
        scheduler.step()

        if horizon_idx < len(horizons) and step == horizons[horizon_idx]:
            snap_grad[step] = {n: t.clone() for n, t in grad_sum.items()}
            snap_momentum[step] = {n: t.clone() for n, t in momentum.items()}
            horizon_idx += 1

        if step in trajectory_steps:
            val = evaluate_loss(model, val_ids, cfg.context_length,
                                batch_size=cfg.batch_size,
                                max_windows=cfg.eval_max_windows)
            val_records[step] = val

    elapsed = time.time() - t0
    return {
        "model_state": model.state_dict(),
        "train_losses": train_losses,
        "train_emas": train_emas,
        "grad_norms_unclipped": grad_norms_unclipped,
        "grad_norms_clipped": grad_norms_clipped,
        "val_records": val_records,
        "g_first": g_first,
        "grad_sum": grad_sum,          # running sums, can query up to final step
        "momentum": momentum,
        "snap_grad": snap_grad,        # horizon -> {name: tensor} raw gradient sums
        "snap_momentum": snap_momentum,
        "flops_per_step": flops_per_step,
        "tokens_per_step": tokens_per_step,
        "elapsed_sec": elapsed,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0,
    }


def flatten_state(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1).float() for t in state.values()])


def state_l2(state: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for t in state.values():
        total += float(t.double().pow(2).sum().item())
    return float(torch.sqrt(torch.tensor(total)).item())