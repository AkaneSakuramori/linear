"""Phase-5 evaluation: baselines, generalization metrics, compute accounting.

Key new metrics:
  * % of oracle improvement recovered =
        (val_no_update − val_learned) / (val_no_update − val_oracle) × 100
        (clamped to [−inf, 100]; 100 = learned equals the oracle ceiling)
  * cosine(ΔW_pred, ΔW_target) per trajectory, compared with
    cosine(observed-gradient direction, ΔW_target).
"""
from __future__ import annotations

from typing import Dict, List

import torch

from src.phase3.features import rel_delta_error, rel_param_distance
from src.phase3.evaluate import Phase3Eval


def no_update_result(record: Dict, K: int, eval_h: Phase3Eval) -> dict:
    return {"kind": "no_update", "K": K,
            "quality": {"val_loss": record["val_records"][K]["loss"],
                        "val_ppl": record["val_records"][K]["ppl"],
                        "train_loss": None}}


def conventional_result(record: Dict, K: int, H: int) -> dict:
    return {"kind": "conventional", "K": K, "H": H,
            "quality": {"val_loss": record["val_records"][H]["loss"],
                        "val_ppl": record["val_records"][H]["ppl"],
                        "train_loss": None}}


def _flatten(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1).float() for _, t in sorted(state.items())])


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm().item(), b.norm().item()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a @ b).item() / (na * nb))


def _gen_deltas(gen, record, K, rank, segs, feature_set, lr):
    if hasattr(gen, "deltas_for"):  # StructuredGenerator (phase 5)
        from src.phase5.generator import generate_deltas
        return generate_deltas(gen, record, K, rank, segs, feature_set, lr)
    # Phase-4 flat UpdateOperator ('direct' arch ablation)
    from src.phase4.operator import generate_deltas as p4gen
    return p4gen(gen, record, K, rank, segs, feature_set, lr)


def learned_result(gen, record: Dict, K: int, H: int, rank: int, segs: list,
                   eval_h: Phase3Eval, feature_set: str, lr: float) -> dict:
    deltas = _gen_deltas(gen, record, K, rank, segs, feature_set, lr)
    w_pred = {name: record["w_states"][K][name] + deltas[name] for name in deltas}
    delta_target = {s["name"]: (record["w_states"][H][s["name"]].float()
                                - record["w_states"][K][s["name"]].float())
                    for s in segs}
    m = eval_h.eval_state(w_pred)
    dp = _flatten(deltas)
    dt = _flatten(delta_target)
    # observed-gradient direction (mean of observed gradients, available at K)
    grad_mean = {s["name"]: sum(record["grad_states"][s2][s["name"]].float()
                                for s2 in range(1, K + 1)) / K for s in segs}
    gm = _flatten(grad_mean)
    return {
        "kind": "learned", "K": K, "H": H, "rank": rank,
        "quality": m,
        "rel_param_dist_to_WH": rel_param_distance(w_pred, record["w_states"][H]),
        "rel_delta_error": rel_delta_error(deltas, delta_target),
        "update_norm": float(dp.norm().item()),
        "cos_pred_target": _cos(dp, dt),
        "cos_gradmean_target": _cos(gm, dt),
    }


def oracle_lowrank_result(record: Dict, K: int, H: int, rank: int, segs: list,
                          eval_h: Phase3Eval) -> dict:
    from src.phase4.oracle import oracle_result
    r = oracle_result(record, K, H, rank, segs, eval_h)
    r["kind"] = "oracle_lowrank"
    return r


def pct_oracle_recovered(no_up: float, learned: float, oracle: float) -> float:
    """% of oracle improvement recovered (clamped to 100 at the top)."""
    denom = no_up - oracle
    if abs(denom) < 1e-12:
        return 100.0 if learned <= no_up + 1e-12 else 0.0
    return min(100.0, (no_up - learned) / denom * 100.0)


# ---------------------------------------------------------------------------
# compute accounting
# ---------------------------------------------------------------------------
def generator_inference_flops(in_dim: int, hidden: int, latent_dim: int,
                              layer_emb_dim: int, coef_dim: int,
                              n_layers: int) -> float:
    enc = (in_dim + layer_emb_dim) * hidden + hidden * latent_dim
    coef = latent_dim * hidden + hidden * coef_dim
    return 2.0 * (enc + coef) * n_layers


def gen_uv_flops(segs: list, rank: int) -> float:
    return 2.0 * sum(s["out"] * s["in"] * rank for s in segs if s["ndim"] == 2)


def generator_compute(record: Dict, K: int, H: int, rank: int,
                      in_dim: int, hidden: int, latent_dim: int,
                      layer_emb_dim: int, coef_dim: int, n_layers: int,
                      n_params: int, segs: list) -> dict:
    f = record["flops_per_step"]
    obs = K * f
    infer = generator_inference_flops(in_dim, hidden, latent_dim,
                                      layer_emb_dim, coef_dim, n_layers)
    uv = gen_uv_flops(segs, rank)
    upd = 2.0 * n_params
    return {
        "observation_fwd_bwd": int(K),
        "generator_inference_flops": float(infer),
        "uv_generation_flops": float(uv),
        "param_update_flops": float(upd),
        "observation_flops": float(obs),
        "direct_total_flops": float(obs + infer + uv + upd),
        "conventional_total_flops": float(H * f),
        "flops_per_step": float(f),
        "steps_saved_vs_conventional": int(H - K),
    }
