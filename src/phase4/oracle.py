"""Oracle for the generated low-rank family.

Answers the decisive question: *is the rank-r low-rank update family capable of
representing a useful transformation at all?* Given the ground-truth target
ΔW_target = W_H − W_K, apply the best rank-r approximation per 2-D layer
(truncated SVD) and keep 1-D tensors exact. This uses the future answer, so it
is an oracle — an upper bound for any learned operator whose output is
restricted to rank ≤ r. If this oracle performs badly, no learned operator in
the same family can succeed.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from src.phase3.features import rel_param_distance


def delta_target(record: Dict, K: int, H: int,
                 layers: List[str]) -> Dict[str, torch.Tensor]:
    return {layer: (record["w_states"][H][layer].float()
                    - record["w_states"][K][layer].float())
            for layer in layers}


def oracle_deltas(delta: Dict[str, torch.Tensor], segs: List[dict],
                  rank: int) -> Dict[str, torch.Tensor]:
    """Best rank-r reconstruction: SVD-truncate 2-D, exact for 1-D."""
    out = {}
    for s in segs:
        name = s["name"]
        t = delta[name]
        if s["ndim"] == 2 and rank > 0:
            u, sv, v = torch.linalg.svd(t, full_matrices=False)
            r = min(rank, len(sv))
            out[name] = (u[:, :r] * sv[:r]) @ v[:r, :]
        else:
            out[name] = t.clone()
    return out


def explained_energy(delta: Dict[str, torch.Tensor],
                     recon: Dict[str, torch.Tensor],
                     segs: List[dict]) -> float:
    """Frobenius energy fraction captured by the reconstruction (2-D only)."""
    num = denom = 0.0
    for s in segs:
        if s["ndim"] != 2:
            continue
        t = delta[s["name"]]
        num += float((recon[s["name"]] ** 2).sum().item())
        denom += float((t ** 2).sum().item())
    return float(num / denom) if denom > 0 else 0.0


def oracle_result(record: Dict, K: int, H: int, rank: int, segs: List[dict],
                  eval_h) -> dict:
    """Evaluate W_K + (best rank-r of ΔW_target) and report metrics."""
    layers = [s["name"] for s in segs]
    delta = delta_target(record, K, H, layers)
    recon = oracle_deltas(delta, segs, rank)
    w_pred = {name: record["w_states"][K][name] + recon[name] for name in recon}
    m = eval_h.eval_state(w_pred)
    return {
        "kind": "oracle_lowrank",
        "K": K, "H": H, "rank": rank,
        "quality": m,
        "explained_energy": explained_energy(delta, recon, segs),
        "rel_param_dist_to_WH": rel_param_distance(w_pred, record["w_states"][H]),
        "update_norm": float(torch.cat([r.reshape(-1) for r in recon.values()]).norm().item()),
    }