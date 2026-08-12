"""Common benchmark interface for Phase-2 update methods.

Every method follows the same contract:

    run_method(ctx, horizon) -> ExperimentResult

which starts from the identical W0, performs a small number of parameter
transformations, and reports model quality plus full compute accounting
(forward/backward/optimizer/update counts, wall time, FLOPs, peak RAM,
tokens). No method receives W_N (the future answer); methods that replay a
rollout only use gradients available along the way. The oracle is the single
explicit exception and is labelled as such.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from src.evaluate import evaluate_loss
from src.train import build_model


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ComputeMetrics:
    fwd_count: int = 0
    bwd_count: int = 0
    optimizer_updates: int = 0
    param_updates: int = 0
    tuning_evals: int = 0
    total_evals: int = 0
    wall_time_sec: float = 0.0
    flops_est: float = 0.0          # total = intrinsic + eval + tuning
    intrinsic_flops_est: float = 0.0  # fwd/bwd + parameter update only
    eval_flops_est: float = 0.0       # final metric evals
    tuning_flops_est: float = 0.0     # alpha search on validation
    tokens_seen: int = 0
    peak_ram_mib: float = 0.0

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class ExperimentResult:
    name: str
    kind: str                      # "baseline" | "oracle" | "practical"
    horizon: int
    alpha: float
    final_train_loss: float
    final_val_loss: float
    final_val_ppl: float
    param_distance_rel: float     # ||final - W_N|| / ||W_N|| (0 for oracle)
    compute: ComputeMetrics = field(default_factory=ComputeMetrics)
    trajectory: list = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "horizon": self.horizon,
            "alpha": self.alpha, "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "final_val_ppl": self.final_val_ppl,
            "param_distance_rel": self.param_distance_rel,
            "compute": self.compute.to_dict(),
            "trajectory": self.trajectory, "notes": self.notes,
        }


class EvalHarness:
    """Reusable model + state loading for loss evaluation (no grad)."""

    def __init__(self, phase1_cfg, corpora, device, vocab_size):
        self.cfg = phase1_cfg
        self.model = build_model(phase1_cfg, vocab_size).to(device)
        self.device = device
        self.val_ids = corpora["val_ids"].to(device)
        self.train_ids = corpora["train_ids"].to(device)
        self.ctx = phase1_cfg.context_length
        self.batch = phase1_cfg.batch_size
        self.max_windows = phase1_cfg.eval_max_windows

    def eval_val(self, state: Dict[str, torch.Tensor]) -> float:
        self.model.load_state_dict(state)
        self.model.eval()
        with torch.no_grad():
            loss = evaluate_loss(self.model, self.val_ids, self.ctx,
                                 batch_size=self.batch,
                                 max_windows=self.max_windows)["loss"]
        self.model.train()
        return loss

    def eval_both(self, state: Dict[str, torch.Tensor]) -> dict:
        self.model.load_state_dict(state)
        self.model.eval()
        v = evaluate_loss(self.model, self.val_ids, self.ctx,
                          batch_size=self.batch, max_windows=self.max_windows)
        t = evaluate_loss(self.model, self.train_ids, self.ctx,
                          batch_size=self.batch, max_windows=self.max_windows)
        self.model.train()
        return {"train_loss": t["loss"], "val_loss": v["loss"], "val_ppl": v["ppl"]}


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
class Ctx:
    def __init__(self, phase1_cfg, corpora, w0_state, captured, wn_states,
                 device, flops_per_step, fwd_1seq_flops, alpha_grid, ranks,
                 beta: float):
        self.phase1_cfg = phase1_cfg
        self.corpora = corpora
        self.w0 = {k: v.clone() for k, v in w0_state.items()}
        self.captured = captured
        self.wn = wn_states                     # horizon -> W_N state dict
        self.device = device
        self.flops_per_step = flops_per_step
        self.fwd_1seq_flops = fwd_1seq_flops
        self.alpha_grid = alpha_grid
        self.ranks = ranks
        self.beta = beta
        self.average_alpha_cache: Dict[int, float] = {}
        self.eval = EvalHarness(phase1_cfg, corpora, device,
                                corpora["vocab_size"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _flatten(state) -> torch.Tensor:
    return torch.cat([t.reshape(-1).float() for t in state.values()])


def _rel_param_distance(a: Dict, b: Dict) -> float:
    da = _flatten(a) - _flatten(b)
    nb = _flatten(b).norm().item()
    return float(da.norm().item() / max(nb, 1e-12))


def _apply_update(w0: Dict[str, torch.Tensor], direction: Dict[str, torch.Tensor],
                  alpha: float) -> Dict[str, torch.Tensor]:
    return {name: w0[name] - alpha * direction[name]
            for name in w0 if name in direction}


def _tune_alpha(ctx: Ctx, direction: Dict[str, torch.Tensor],
                compute: ComputeMetrics) -> float:
    best_a, best_v = 0.0, None
    for a in ctx.alpha_grid.tolist():
        state = _apply_update(ctx.w0, direction, a)
        v = ctx.eval.eval_val(state)
        compute.tuning_evals += 1
        compute.total_evals += 1
        if best_v is None or v < best_v:
            best_v, best_a = v, a
    return best_a


def _evaluate_with(ctx: Ctx, state, compute: ComputeMetrics) -> dict:
    m = ctx.eval.eval_both(state)
    compute.total_evals += 1
    return m


def _eval_flops(ctx: Ctx, n_evals: int) -> float:
    return ctx.fwd_1seq_flops * 1024.0 * n_evals


def _finalize_compute(ctx: Ctx, compute: ComputeMetrics,
                      intrinsic_flops: float, wall_sec: float) -> None:
    compute.intrinsic_flops_est = intrinsic_flops
    compute.eval_flops_est = _eval_flops(ctx, max(0, compute.total_evals - compute.tuning_evals))
    compute.tuning_flops_est = _eval_flops(ctx, compute.tuning_evals)
    compute.flops_est = (compute.intrinsic_flops_est + compute.eval_flops_est
                         + compute.tuning_flops_est)
    compute.wall_time_sec = wall_sec


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
def baseline_adamw(ctx: Ctx, horizon: int) -> ExperimentResult:
    cap = ctx.captured
    n = ctx.phase1_cfg.n_steps
    n_eval_steps = len(ctx.phase1_cfg.trajectory_steps)
    intrinsic = n * ctx.flops_per_step
    compute = ComputeMetrics(
        fwd_count=n,
        bwd_count=n,
        optimizer_updates=n,
        param_updates=n,
        total_evals=n_eval_steps + 1,
        tokens_seen=n * ctx.phase1_cfg.batch_size * ctx.phase1_cfg.context_length,
        peak_ram_mib=cap["peak_rss_gib"] * 1024.0,
    )
    final = _evaluate_with(ctx, cap["model_state"], compute)
    _finalize_compute(ctx, compute, intrinsic, cap["elapsed_sec"])
    traj = [{"step": k, "train_loss": cap["train_losses"][k],
             "val_loss": rec["loss"], "val_ppl": rec["ppl"]}
            for k, rec in cap["val_records"].items()]
    try:
        wn_horizon = n
        wn = ctx.wn[wn_horizon]
    except KeyError:
        wn = cap["model_state"]
    return ExperimentResult(
        name="BaselineAdamW", kind="baseline", horizon=n,
        alpha=float(ctx.phase1_cfg.lr),
        final_train_loss=final["train_loss"],
        final_val_loss=final["val_loss"], final_val_ppl=final["val_ppl"],
        param_distance_rel=_rel_param_distance(cap["model_state"], wn),
        compute=compute, trajectory=traj,
        notes="Full Phase-1 AdamW run (replayed once for instrumentation; N=100)",
    )


def direct_gradient(ctx: Ctx, horizon: int) -> ExperimentResult:
    start = time.time()
    g = ctx.captured["g_first"]
    compute = ComputeMetrics(fwd_count=1, bwd_count=1, optimizer_updates=0,
                             param_updates=1)
    a = _tune_alpha(ctx, g, compute)
    state = _apply_update(ctx.w0, g, a)
    m = _evaluate_with(ctx, state, compute)
    _finalize_compute(ctx, compute, ctx.flops_per_step, time.time() - start)
    return ExperimentResult(
        name="DirectGradient", kind="practical", horizon=horizon, alpha=a,
        final_train_loss=m["train_loss"], final_val_loss=m["val_loss"],
        final_val_ppl=m["val_ppl"],
        param_distance_rel=_rel_param_distance(state, ctx.wn[ctx.phase1_cfg.n_steps]),
        compute=compute, trajectory=[{"marker": "direct", "val_loss": m["val_loss"]}],
        notes="Single scaled first-gradient update: W0 - alpha*gradL(W0); alpha tuned on val",
    )


def direct_average_gradient(ctx: Ctx, horizon: int) -> ExperimentResult:
    start = time.time()
    gsum = ctx.captured["snap_grad"][horizon]
    direction = {name: t / horizon for name, t in gsum.items()}
    compute = ComputeMetrics(fwd_count=horizon, bwd_count=horizon,
                             optimizer_updates=0, param_updates=1)
    a = _tune_alpha(ctx, direction, compute)
    state = _apply_update(ctx.w0, direction, a)
    m = _evaluate_with(ctx, state, compute)
    _finalize_compute(ctx, compute, horizon * ctx.flops_per_step, time.time() - start)
    return ExperimentResult(
        name="DirectAverageGradient", kind="practical", horizon=horizon, alpha=a,
        final_train_loss=m["train_loss"], final_val_loss=m["val_loss"],
        final_val_ppl=m["val_ppl"],
        param_distance_rel=_rel_param_distance(state, ctx.wn[ctx.phase1_cfg.n_steps]),
        compute=compute, trajectory=[{"marker": "direct", "val_loss": m["val_loss"]}],
        notes=f"Single update along mean of {horizon} roll-out gradients",
    )


def direct_momentum(ctx: Ctx, horizon: int) -> ExperimentResult:
    start = time.time()
    direction = ctx.captured["snap_momentum"][horizon]
    compute = ComputeMetrics(fwd_count=horizon, bwd_count=horizon,
                             optimizer_updates=0, param_updates=1)
    a = _tune_alpha(ctx, direction, compute)
    state = _apply_update(ctx.w0, direction, a)
    m = _evaluate_with(ctx, state, compute)
    _finalize_compute(ctx, compute, horizon * ctx.flops_per_step, time.time() - start)
    return ExperimentResult(
        name="DirectMomentum", kind="practical", horizon=horizon, alpha=a,
        final_train_loss=m["train_loss"], final_val_loss=m["val_loss"],
        final_val_ppl=m["val_ppl"],
        param_distance_rel=_rel_param_distance(state, ctx.wn[ctx.phase1_cfg.n_steps]),
        compute=compute, trajectory=[{"marker": "direct", "val_loss": m["val_loss"]}],
        notes=f"Momentum-styled cumulative EMA direction over {horizon} roll-out steps",
    )


def direct_lowrank(ctx: Ctx, horizon: int, rank: int) -> ExperimentResult:
    start = time.time()
    gsum = ctx.captured["snap_grad"][horizon]
    direction = {name: t / horizon for name, t in gsum.items()}
    lr_dir = {}
    for name, t in direction.items():
        t2 = t.float()
        if t2.ndim == 2:
            u, s, v = torch.linalg.svd(t2, full_matrices=False)
            k = min(rank, min(u.shape[0], v.shape[0]))
            lr_dir[name] = (u[:, :k] * s[:k]) @ v[:k, :]
        else:
            lr_dir[name] = t2
    compute = ComputeMetrics(fwd_count=horizon, bwd_count=horizon,
                             optimizer_updates=0, param_updates=1)
    if horizon not in ctx.average_alpha_cache:
        d_full = {k: v / horizon for k, v in gsum.items()}
        ctx.average_alpha_cache[horizon] = _tune_alpha(ctx, d_full, compute)
    a = ctx.average_alpha_cache[horizon]
    state = _apply_update(ctx.w0, lr_dir, a)
    m = _evaluate_with(ctx, state, compute)
    _finalize_compute(ctx, compute, horizon * ctx.flops_per_step, time.time() - start)
    return ExperimentResult(
        name=f"DirectLowRank_r{rank}", kind="practical", horizon=horizon, alpha=a,
        final_train_loss=m["train_loss"], final_val_loss=m["val_loss"],
        final_val_ppl=m["val_ppl"],
        param_distance_rel=_rel_param_distance(state, ctx.wn[ctx.phase1_cfg.n_steps]),
        compute=compute, trajectory=[{"marker": "direct", "val_loss": m["val_loss"]}],
        notes=f"Rank-{rank} SVD truncation of mean roll-out gradient; alpha=avg-alpha",
    )


def direct_oracle(ctx: Ctx, horizon: int) -> ExperimentResult:
    start = time.time()
    wn = ctx.wn[horizon]
    state = {name: v.clone() for name, v in wn.items()}
    compute = ComputeMetrics(fwd_count=0, bwd_count=0, optimizer_updates=0,
                             param_updates=1)
    m = _evaluate_with(ctx, state, compute)
    _finalize_compute(ctx, compute, 0.0, time.time() - start)
    return ExperimentResult(
        name="DirectOracle", kind="oracle", horizon=horizon, alpha=1.0,
        final_train_loss=m["train_loss"], final_val_loss=m["val_loss"],
        final_val_ppl=m["val_ppl"],
        param_distance_rel=0.0,
        compute=compute, trajectory=[{"marker": "oracle W_N", "val_loss": m["val_loss"]}],
        notes="ORACLE ONLY - transports the future answer W_N to W0 in one update. "
              "Not a practical training algorithm; an upper bound.",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def run_method(name: str, ctx: Ctx, horizon: int, rank: int | None = None) -> ExperimentResult:
    if name == "BaselineAdamW":
        return baseline_adamw(ctx, horizon)
    if name == "DirectOracle":
        return direct_oracle(ctx, horizon)
    if name == "DirectGradient":
        return direct_gradient(ctx, horizon)
    if name == "DirectAverageGradient":
        return direct_average_gradient(ctx, horizon)
    if name == "DirectMomentum":
        return direct_momentum(ctx, horizon)
    if name == "DirectLowRank":
        return direct_lowrank(ctx, horizon, rank or 64)
    raise ValueError(f"unknown method {name}")