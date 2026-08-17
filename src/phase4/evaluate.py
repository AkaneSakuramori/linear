"""Baselines and direct-application evaluation for Phase 4.

Baselines compared for every experiment:
  A no update            W_K
  B conventional AdamW   W_K → AdamW → W_H   (reference numbers from record)
  C Phase-3 predictor    (reused, K=10 pairs only — Phase-3 grid was K∈{5,10})
  D oracle low-rank      W_K + best rank-r of (W_H − W_K)   (future answer)
Learned operator         W_K + ΔW generated from steps-1..K info only.

Compute accounting reports the four mandated buckets separately and an
amortized table over N applications.
"""
from __future__ import annotations

import os
from typing import Dict, List

import torch

from src.phase3.features import rel_param_distance, rel_delta_error
from src.phase3.evaluate import Phase3Eval, direct_result as p3_direct_result
from src.phase4.operator import (build_segments, generate_deltas,
                                 offsets_of, total_gen)

MAX_WINDOWS = 1024

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def no_update_result(record: Dict, K: int, eval_h: Phase3Eval) -> dict:
    return {
        "kind": "no_update", "K": K,
        "quality": {"train_loss": None,
                    "val_loss": record["val_records"][K]["loss"],
                    "val_ppl": record["val_records"][K]["ppl"]},
        "rel_param_dist_to_WH": None,
    }


def conventional_result(record: Dict, K: int, H: int) -> dict:
    return {
        "kind": "conventional", "K": K, "H": H,
        "quality": {"train_loss": None,
                    "val_loss": record["val_records"][H]["loss"],
                    "val_ppl": record["val_records"][H]["ppl"]},
        "rel_param_dist_to_WH": 0.0,
    }


def phase3_predictor_result(record: Dict, K: int, H: int, eval_h: Phase3Eval,
                            phase3_ckpt_dir: str, lr: float) -> dict:
    """Reuse the Phase-3 per-layer-scaling predictor as baseline C."""
    from src.phase3.predictor import Predictor
    path = os.path.join(phase3_ckpt_dir, f"predictor_K{K}_H{H}_full_mean.pt")
    if not os.path.exists(path):
        return {"kind": "phase3_predictor", "K": K, "H": H, "available": False}
    in_dim = 2 * K + 12  # feature_dim('full', K)
    pred = Predictor.load(path, in_dim, 64, 1)
    r = p3_direct_result(record, K, H, "mean", pred, eval_h, "full", lr)
    r["kind"] = "phase3_predictor"
    r["available"] = True
    return r


# ---------------------------------------------------------------------------
# Learned operator direct application
# ---------------------------------------------------------------------------
def learned_result(record: Dict, K: int, H: int, rank: int, operator,
                   segs: List[dict], eval_h: Phase3Eval, feature_set: str,
                   lr: float) -> dict:
    deltas = generate_deltas(operator, record, K, rank, segs, feature_set, lr)
    w_pred = {name: record["w_states"][K][name] + deltas[name] for name in deltas}
    delta_t = {name: (record["w_states"][H][name].float()
                      - record["w_states"][K][name].float())
               for name in deltas}
    m = eval_h.eval_state(w_pred)
    update_norm = float(torch.cat([d.reshape(-1).float() for d in deltas.values()]).norm().item())
    return {
        "kind": "learned", "K": K, "H": H, "rank": rank,
        "quality": m,
        "rel_param_dist_to_WH": rel_param_distance(w_pred, record["w_states"][H]),
        "rel_delta_error": rel_delta_error(deltas, delta_t),
        "update_norm": update_norm,
        "per_layer_update_norms": {name: float(d.norm().item()) for name, d in deltas.items()},
    }


# ---------------------------------------------------------------------------
# Compute accounting
# ---------------------------------------------------------------------------
def operator_inference_flops(in_dim: int, hidden: int, layer_emb_dim: int,
                             gen: int, n_layers: int) -> float:
    """One forward through the shared operator for every layer."""
    trunk = (in_dim + layer_emb_dim) * hidden + hidden * hidden
    head = hidden * gen
    macs = trunk + head
    return 2.0 * macs * n_layers


def generation_flops(segs: List[dict], rank: int) -> float:
    """FLOPs of all U V^T matmuls (2-D layers)."""
    return 2.0 * sum(s["out"] * s["in"] * rank for s in segs if s["ndim"] == 2)


def learned_direct_compute(record: Dict, K: int, H: int, rank: int,
                           in_dim: int, hidden: int, layer_emb_dim: int,
                           gen: int, n_layers: int, n_params: int,
                           segs: List[dict]) -> dict:
    f = record["flops_per_step"]
    obs = K * f
    infer = operator_inference_flops(in_dim, hidden, layer_emb_dim, gen, n_layers)
    genf = generation_flops(segs, rank)
    upd = 2.0 * n_params
    return {
        "observation_fwd_bwd": int(K),
        "operator_inference_flops": float(infer),
        "generation_flops": float(genf),
        "param_update_flops": float(upd),
        "observation_flops": float(obs),
        "direct_total_flops": float(obs + infer + genf + upd),
        "conventional_total_flops": float(H * f),
        "steps_saved_vs_conventional": int(H - K),
        "flops_per_step": float(f),
    }


def segs_from_state(record: Dict) -> List[dict]:
    return build_segments(record["w_states"][0])


def operator_train_flops(steps: int, n_examples: int, in_dim: int,
                         hidden: int, layer_emb_dim: int, gen: int,
                         backward_factor: float = 3.0) -> float:
    macs = (in_dim + layer_emb_dim) * hidden + hidden * hidden + hidden * gen
    return float(steps * n_examples * 2.0 * macs * (1.0 + backward_factor))