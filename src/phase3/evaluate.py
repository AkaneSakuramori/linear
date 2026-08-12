"""Direct-application evaluation and compute accounting for Phase 3.

Separates the two experiments demanded by the research plan:

* Oracle ceiling  — coefficients alpha* chosen by least squares against the
  ground-truth ΔW_target = W_H − W_K. Uses the future answer, so it is an
  upper bound for the *per-layer-scaling family*, never a practical method.
* Direct application — the meta-trained Predictor, given only information
  available at step K, produces alpha and therefore W_pred. No future state
  is ever touched here.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from src.evaluate import evaluate_loss
from src.model import Transformer
from src.train import build_model
from src.phase3.features import (apply_prediction, build_basis,
                                 build_features, compute_alpha_star,
                                 param_names, predict_all_layers,
                                 reconstruct_layer_delta, rel_delta_error,
                                 rel_param_distance)

MAX_WINDOWS = 1024


class Phase3Eval:
    def __init__(self, phase1_cfg, corpora, device=None):
        if device is None:
            device = torch.device("cpu")
        self.cfg = phase1_cfg
        self.device = device
        self.model = build_model(phase1_cfg, corpora["vocab_size"]).to(device)
        self.val_ids = corpora["val_ids"].to(device)
        self.train_ids = corpora["train_ids"].to(device)
        self.batch = phase1_cfg.batch_size

    def eval_state(self, state: Dict[str, torch.Tensor], max_windows: int = MAX_WINDOWS) -> Dict:
        self.model.load_state_dict(state)
        self.model.eval()
        with torch.no_grad():
            v = evaluate_loss(self.model, self.val_ids, self.cfg.context_length,
                              batch_size=self.batch, max_windows=max_windows)
            t = evaluate_loss(self.model, self.train_ids, self.cfg.context_length,
                              batch_size=self.batch, max_windows=max_windows)
        self.model.train()
        return {"train_loss": t["loss"], "val_loss": v["loss"], "val_ppl": v["ppl"]}


def predict_alphas(record: Dict, K: int, predictor, feature_set: str, lr: float):
    """Features for every layer -> predicted coefficients (L, J)."""
    layers = param_names(record)
    X = torch.stack([build_features(record, K, layer, feature_set, lr)
                     for layer in layers])
    return layers, X, predictor.predict(X)


def oracle_alphas(record: Dict, K: int, H: int, basis: str) -> Dict[str, torch.Tensor]:
    layers = param_names(record)
    return {layer: compute_alpha_star(record, K, H, basis, layer) for layer in layers}


def build_w_pred(record: Dict, K: int, basis: str,
                 alphas: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    deltas = predict_all_layers(record, K, basis, alphas)
    return apply_prediction(record["w_states"][K], deltas)


def direct_result(record: Dict, K: int, H: int, basis: str,
                  predictor, eval_h: Phase3Eval, feature_set: str, lr: float) -> Dict:
    """Practical Direct Learning: predictor receives only info up to step K."""
    layers, X, alphas = predict_alphas(record, K, predictor, feature_set, lr)
    alphas_by_layer = {layer: alphas[i] for i, layer in enumerate(layers)}
    w_pred = build_w_pred(record, K, basis, alphas_by_layer)
    delta_target = {layer: (record["w_states"][H][layer].float()
                            - record["w_states"][K][layer].float())
                    for layer in layers}
    deltas_pred = {layer: reconstruct_layer_delta(build_basis(record, K, basis, layer),
                                                  alphas_by_layer[layer])
                   for layer in layers}
    m = eval_h.eval_state(w_pred)
    return {
        "kind": "direct",
        "K": K, "H": H,
        "alphas": {layer: alphas[i].tolist() for i, layer in enumerate(layers)},
        "quality": m,
        "rel_param_dist_to_WH": rel_param_distance(w_pred, record["w_states"][H]),
        "rel_delta_error": rel_delta_error(deltas_pred, delta_target),
        "prediction_used": True,
    }


def oracle_result(record: Dict, K: int, H: int, basis: str,
                  eval_h: Phase3Eval) -> Dict:
    """Oracle ceiling for the per-layer-scaling family (uses ΔW_target)."""
    layers = param_names(record)
    alphas = oracle_alphas(record, K, H, basis)
    w_pred = build_w_pred(record, K, basis, alphas)
    m = eval_h.eval_state(w_pred)
    return {
        "kind": "oracle",
        "K": K, "H": H,
        "quality": m,
        "rel_param_dist_to_WH": rel_param_distance(w_pred, record["w_states"][H]),
        "prediction_used": False,
    }


def conventional_result(record: Dict, K: int, H: int) -> Dict:
    """Conventional AdamW W_K -> W_H (H−K further steps). Reference numbers."""
    return {
        "kind": "conventional",
        "K": K, "H": H,
        "quality": {
            "train_loss": None,
            "val_loss": record["val_records"][H]["loss"],
            "val_ppl": record["val_records"][H]["ppl"],
        },
        "rel_param_dist_to_WH": 0.0,
        "prediction_used": False,
    }


# ---------------------------------------------------------------------------
# Compute accounting
# ---------------------------------------------------------------------------
def predictor_inference_flops(net_params: List[int]) -> float:
    """FLOPs for one forward pass of the Predictor MLP through all layers.

    net_params = [in_dim, hidden, hidden, out_dim]; a batch of `L` layers is
    processed at once, so matmuls scale with L (the number of layers).
    """
    macs = 0.0
    for a, b in zip(net_params[:-1], net_params[1:]):
        macs += a * b
    return 2.0 * macs


def direct_compute(record: Dict, K: int, H: int, predictor_inference_flops: float,
                   n_params: int, eval_flops: float = 0.0) -> Dict:
    flops_per_step = record["flops_per_step"]
    obs = K * flops_per_step
    infer = predictor_inference_flops
    update = 2.0 * n_params
    return {
        "observation_fwd_bwd": int(K),
        "predictor_inference_flops": float(infer),
        "param_update_flops": float(update),
        "observation_flops": float(obs),
        "direct_total_flops": float(obs + infer + update + eval_flops),
        "conventional_total_flops": float(H * flops_per_step),
        "steps_saved_vs_conventional": int(H - K),
        "flops_per_step": float(flops_per_step),
    }


def predictor_train_flops(steps: int, batch_size: int, net_params: List[int],
                          backward_factor: float = 3.0) -> float:
    macs = 0.0
    for a, b in zip(net_params[:-1], net_params[1:]):
        macs += a * b
    return float(steps * batch_size * 2.0 * macs * (1.0 + backward_factor))


def trajectory_gen_flops(record: Dict) -> float:
    return record["flops_per_step"] * 100.0  # trajectories are rolled to H_MAX=100
