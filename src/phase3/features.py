"""Feature construction, structured parameterisation, and targets for Phase 3.

The Direct Update Predictor is a *per-layer scalar predictor*: given a compact
per-layer + global feature vector (built only from steps 1..K of a trajectory),
it outputs coefficients alpha_{l,j} for a small basis of observed-gradient
directions per layer. The predicted parameter transformation is

    ΔW_pred = Σ_l Σ_j alpha_{l,j} · D_{l,j}
    W_pred  = W_K + ΔW_pred

with D_{l,j} chosen among the mean / first / last / all observed gradients
(available at step K, so no future information). The supervised target used
only during meta-training is the best-possible coefficient vector for the
ground-truth ΔW_target = W_H − W_K, computed per layer by least squares.

Feature sets (ablation study A-F):
  'loss'       global loss history only
  'grad'       global + per-layer gradient statistics only
  'grad_loss'  gradient + loss (no parameter statistics)
  'full'       gradient + loss + parameter statistics + per-layer info (primary)
  'compressed' full with the step-history compressed to [first, last, mean]
  'rich'       full + Adam optimizer-state statistics at step K
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

HISTORY_SETS = ("loss", "grad", "grad_loss", "full", "rich")
ALL_SETS = ("loss", "grad", "grad_loss", "full", "compressed", "rich")
BASIS_NAMES = ("mean", "first_last", "first_last_mean", "all")


def param_names(record: Dict) -> List[str]:
    return sorted(record["w_states"][0].keys())


# ---------------------------------------------------------------------------
# Parameterisation: basis directions + reconstruction
# ---------------------------------------------------------------------------
def build_basis(record: Dict, K: int, basis: str, layer: str) -> List[torch.Tensor]:
    """Observed-gradient basis directions for one layer (info up to step K)."""
    grads = [record["grad_states"][s][layer].float() for s in range(1, K + 1)]
    mean_g = sum(grads[1:], grads[0]) / K if len(grads) > 1 else grads[0]
    if basis == "mean":
        return [mean_g]
    if basis == "first_last":
        return [grads[0], grads[-1]]
    if basis == "first_last_mean":
        return [grads[0], grads[-1], mean_g]
    if basis == "all":
        return list(grads)
    raise ValueError(f"unknown basis {basis}")


def compute_alpha_star(record: Dict, K: int, H: int, basis: str, layer: str) -> torch.Tensor:
    """Least-squares coefficients: alpha* = argmin ||ΔW_target − D·alpha||².

    Uses the ground-truth ΔW_target = W_H − W_K. Available ONLY while
    meta-training the predictor; never at direct-application time.
    """
    target = record["w_states"][H][layer].float() - record["w_states"][K][layer].float()
    basis_vecs = build_basis(record, K, basis, layer)
    B = torch.stack([v.reshape(-1) for v in basis_vecs], dim=1)  # (N, J)
    A = B.t() @ B + 1e-6 * torch.eye(B.shape[1], dtype=B.dtype)
    b = B.t() @ target.reshape(-1)
    return torch.linalg.solve(A, b)


def reconstruct_layer_delta(basis_vecs: List[torch.Tensor],
                            alphas: torch.Tensor) -> torch.Tensor:
    """Σ_j alpha_j · D_j for one layer."""
    out = None
    for a, v in zip(alphas, basis_vecs):
        out = a * v if out is None else out + a * v
    return out


def apply_prediction(w_k: Dict[str, torch.Tensor],
                     deltas: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """W_pred = W_K + ΔW_pred (only over layers that have a delta)."""
    return {name: w_k[name] + deltas[name] for name in deltas}


def predict_all_layers(record: Dict, K: int, basis: str, alphas_by_layer: Dict[str, torch.Tensor]):
    """Build ΔW_pred from predicted coefficients (practical application)."""
    deltas = {}
    for layer, alpha in alphas_by_layer.items():
        deltas[layer] = reconstruct_layer_delta(build_basis(record, K, basis, layer), alpha)
    return deltas


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def feature_dim(feature_set: str, K: int) -> int:
    if feature_set == "loss":
        return K
    if feature_set == "grad":
        return K + 5
    if feature_set == "grad_loss":
        return 2 * K + 5
    if feature_set == "full":
        return 2 * K + 12
    if feature_set == "compressed":
        return 17
    if feature_set == "rich":
        return 2 * K + 14
    raise ValueError(f"unknown feature set {feature_set}")


def _hist3(vals) -> List[float]:
    return [vals[0], vals[-1], sum(vals) / len(vals)]


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm().item(), b.norm().item()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a @ b).item() / (na * nb))


def build_features(record: Dict, K: int, layer: str, feature_set: str,
                   lr: float) -> torch.Tensor:
    """Feature vector for (trajectory, layer). Uses ONLY steps 1..K."""
    if feature_set not in ALL_SETS:
        raise ValueError(f"unknown feature set {feature_set}")
    feats: List[float] = []

    loss_hist = [record["losses"][s] / record["losses"][0] for s in range(1, K + 1)]
    grad_hist = [record["grad_norms"][s] / max(record["grad_norms"][1], 1e-12)
                 for s in range(1, K + 1)]

    # ---- global features ----------------------------------------------------
    if feature_set in ("loss", "grad_loss", "full", "rich", "compressed"):
        feats += (_hist3(loss_hist) if feature_set == "compressed" else loss_hist)
    if feature_set in ("grad", "grad_loss", "full", "rich", "compressed"):
        feats += (_hist3(grad_hist) if feature_set == "compressed" else grad_hist)
    if feature_set in ("full", "rich", "compressed"):
        feats.append(record["param_norms"][K] / max(record["param_norms"][0], 1e-12))
        feats.append(lr)

    # ---- per-layer features -------------------------------------------------
    gnorm = [record["grad_states"][s][layer].float().norm().item() for s in range(1, K + 1)]
    frac_K = gnorm[-1] / max(record["grad_norms"][K], 1e-12)
    if feature_set in ("grad", "grad_loss", "full", "rich", "compressed"):
        if feature_set == "compressed":
            feats += [gnorm[0], gnorm[-1], sum(gnorm) / len(gnorm), frac_K]
        else:
            mean = sum(gnorm) / len(gnorm)
            std = (sum((g - mean) ** 2 for g in gnorm) / len(gnorm)) ** 0.5
            feats += [mean, std, gnorm[0], gnorm[-1], frac_K]

    if feature_set in ("full", "rich", "compressed"):
        w0 = record["w_states"][0][layer].float()
        wk = record["w_states"][K][layer].float()
        n0, nk = w0.norm().item(), wk.norm().item()
        mean_g = sum(record["grad_states"][s][layer].float() for s in range(1, K + 1)) / K
        cgp = _cosine(mean_g.reshape(-1), wk.reshape(-1))
        cg_hist = [_cosine(record["grad_states"][s][layer].float().reshape(-1),
                           record["grad_states"][s + 1][layer].float().reshape(-1))
                   for s in range(1, K)]
        cgc = sum(cg_hist) / len(cg_hist) if cg_hist else 0.0
        feats += [nk, n0, nk / max(n0, 1e-12), cgp, cgc]

    if feature_set == "rich":
        st = record["adam_stats"][K][layer]
        feats += [st["m_norm"], st["v_norm"]]

    return torch.tensor(feats, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Parameter-distance metrics
# ---------------------------------------------------------------------------
def flatten(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    # canonical order (sorted names): reconstructed states are keyed in sorted
    # order while model.state_dict() is in module order — keep all comparisons
    # aligned regardless of which came from where.
    return torch.cat([t.reshape(-1).float() for _, t in sorted(state.items())])


def rel_param_distance(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    da = flatten(a) - flatten(b)
    nb = flatten(b).norm().item()
    return float(da.norm().item() / max(nb, 1e-12))


def rel_delta_error(delta_pred: Dict[str, torch.Tensor],
                    delta_target: Dict[str, torch.Tensor]) -> float:
    d = flatten(delta_pred) - flatten(delta_target)
    return float(d.norm().item() / max(flatten(delta_target).norm().item(), 1e-12))
