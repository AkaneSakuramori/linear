"""Structural analysis of cumulative updates ΔW_N = W_N − W_0.

Computes norms, cosine alignments against candidate directions (first
gradient, average gradient, momentum), per-layer magnitudes, fractions of
meaningfully-changed parameters, and low-rank / singular-value structure.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import numpy as np

from src.checkpoint import load_trajectory_step_last, load_trajectory_step


def load_trajectory_models(dirpath: str, steps: List[int]) -> Dict[int, Dict]:
    """Load state-dicts for the given trajectory steps (cheap, no RNG)."""
    out = {}
    for s in steps:
        p = f"{dirpath}/step_{s:04d}.pt"
        out[s] = torch.load(p, map_location="cpu", weights_only=False)
    return out


def state_delta(w, w0) -> Dict[str, torch.Tensor]:
    return {name: w[name].float() - w0[name].float()
            for name in w if name in w0}


def vec(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1).float() for t in state.values()])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm().item(), b.norm().item()
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b).item() / (na * nb))


def cosine_state(da: Dict[str, torch.Tensor], db: Dict[str, torch.Tensor]) -> float:
    return cosine(vec(da), vec(db))


def delta_stats(wn, w0, grad_state=None, momentum=None, threshold_rel=1e-3):
    """Per-horizon stats for cumulative update ΔW = W_N − W_0."""
    d = state_delta(wn, w0)
    flat = vec(d)
    d_norm = flat.norm().item()
    w0_norm = vec(w0).norm().item()
    w_frac = d_norm / max(1e-12, w0_norm)

    # per-parameter "meaningful change" count with a per-tensor relative floor
    changed = 0
    total_params = 0
    per_layer = {}
    total_w0 = w0_norm
    for name in d:
        dd = d[name]
        w0t = w0[name].float()
        n = dd.numel()
        total_params += n
        floor = threshold_rel * max(1e-9, w0t.norm().item() / math.sqrt(max(1, n)))
        changed += int((dd.abs() > max(1e-8, floor)).sum().item())
        per_layer[name] = {
            "update_l2": float(dd.norm().item()),
            "update_l2_frac_of_delta": float(dd.norm().item() / max(1e-12, d_norm)),
            "grad_l2": None,
            "grad_frac": None,
        }

    def _layer_grad_norms(grad_state):
        out = {}
        for name in grad_state:
            g = grad_state[name].float()
            out[name] = float(g.norm().item())
        return out

    cos_g0 = None
    cos_avg = None
    cos_mom = None
    if grad_state is not None and "g_first" in grad_state:
        cos_g0 = cosine_state(d, grad_state["g_first"])
        gl = _layer_grad_norms(grad_state["g_first"])
        for name in per_layer:
            per_layer[name]["grad_l2"] = gl.get(name)
            per_layer[name]["grad_frac"] = (gl.get(name, 0.0)
                                            / max(1e-12, sum(gl.values())))
    if grad_state is not None and "avg" in grad_state:
        cos_avg = cosine_state(d, grad_state["avg"])
    if momentum is not None:
        cos_mom = cosine_state(d, momentum)

    return {
        "delta_l2": d_norm,
        "w0_l2": w0_norm,
        "delta_w0_ratio": w_frac,
        "changed_params_frac": changed / max(1, total_params),
        "cos_sim_first_grad": cos_g0,
        "cos_sim_avg_grad": cos_avg,
        "cos_sim_momentum": cos_mom,
        "per_layer": per_layer,
    }


def svd_analysis(delta: Dict[str, torch.Tensor], ranks: List[int]) -> Dict:
    """Per-layer singular values and low-rank energy retention for 2D weights."""
    out = {}
    for name, t in delta.items():
        t2 = t.float()
        if t2.ndim == 2:
            sv = torch.linalg.svdvals(t2)
        elif t2.ndim == 4:
            sv = torch.linalg.svdvals(t2.reshape(t2.shape[0], -1))
        else:
            continue
        s2 = (sv ** 2)
        total = float(s2.sum().item())
        if total <= 0:
            continue
        cum = torch.cumsum(s2, 0)
        energy = [float(cum[min(k, len(sv)) - 1].item() / total)
                  if min(k, len(sv)) >= 1 else 0.0
                  for k in ranks if k <= len(sv)]
        out[name] = {
            "shape": list(t2.shape),
            "sv_length": int(len(sv)),
            "sv_top20": [float(i) for i in sv[:20].tolist()],
            "rank_energy": {str(k): float(energy[i])
                            for i, k in enumerate(ranks) if k <= len(sv)},
            "eff_rank_95": int((cum / max(total, 1e-12) >= 0.95).sum().item()),
        }
    return out


def direction_change(mean_deltas: Dict[int, Dict]) -> List[Dict]:
    """Cosine between cumulative delta directions at successive horizons."""
    keys = sorted(mean_deltas.keys())
    out = []
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        out.append({
            "horizon_a": a, "horizon_b": b,
            "cosine": cosine(vec(mean_deltas[a]), vec(mean_deltas[b])),
        })
    return out