"""The Phase-4 learned update operator.

Given information available at observation step K (compact per-layer + global
features, plus a learned per-layer identity embedding), the operator GENERATES
new per-layer parameter directions as low-rank factors:

    ΔW_l = U_l V_l^T          (2-D weights, rank r)
    Δw_l = u_l                (1-D tensors, generated vector)

Unlike Phase 3 (which scaled observed-gradient directions), the U_l / V_l
factors are outputs of a shared neural network and are NOT restricted to the
span of the observed gradients. The network is shared across layers; each
layer's output occupies a fixed segment of the shared head (positional layer
identity + per-layer features distinguish layers).

The supervised objective is the RECONSTRUCTED update:

    loss_update = Σ_l ||U_l V_l^T − ΔW_target_l||_F² / ||ΔW_target_l||_F²

which is invariant to the factorization ambiguity UV^T = (UR)(R⁻¹V)^T. An
optional behavioural term (validation loss of W_K + ΔW, back-propagated
through a functional forward pass of the transformer) can be added.

META-TRAINING (future target allowed) and DIRECT APPLICATION (future target
forbidden) are kept in separate code paths: features come only from steps 1..K
and the operator never sees W_H.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layer segments (fixed layout of the shared output head)
# ---------------------------------------------------------------------------
def build_segments(state0: Dict[str, torch.Tensor]) -> List[dict]:
    segs = []
    for name in sorted(state0.keys()):
        t = state0[name]
        if t.ndim == 2:
            segs.append({"name": name, "out": int(t.shape[0]),
                         "in": int(t.shape[1]), "ndim": 2})
        else:
            segs.append({"name": name, "out": int(t.numel()),
                         "in": int(t.numel()), "ndim": 1})
    return segs


def segment_length(seg: dict, rank: int) -> int:
    return (seg["out"] + seg["in"]) * rank if seg["ndim"] == 2 else seg["in"]


def total_gen(segs: List[dict], rank: int) -> int:
    return sum(segment_length(s, rank) for s in segs)


def offsets_of(segs: List[dict], rank: int) -> List[int]:
    offs, o = [], 0
    for s in segs:
        offs.append(o)
        o += segment_length(s, rank)
    return offs


def delta_from_row(row: torch.Tensor, idx: int, segs: List[dict],
                   offs: List[int], rank: int) -> torch.Tensor:
    """Extract and reconstruct one layer's delta from a head row."""
    s = segs[idx]
    o = offs[idx]
    if s["ndim"] == 2:
        u = row[o:o + s["out"] * rank].view(s["out"], rank)
        v = row[o + s["out"] * rank:o + s["out"] * rank + s["in"] * rank].view(s["in"], rank)
        return u @ v.t()
    return row[o:o + s["in"]]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class UpdateOperator(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layer_emb_dim: int,
                 n_layers: int, total_gen: int, feat_mean: torch.Tensor,
                 feat_std: torch.Tensor):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.layer_emb_dim = layer_emb_dim
        self.total_gen = total_gen
        self.register_buffer("feat_mean", feat_mean)
        self.register_buffer("feat_std", feat_std.clamp_min(1e-8))
        self.layer_emb = nn.Embedding(n_layers, layer_emb_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim + layer_emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, total_gen)
        # zero-init the output head: the operator starts at "no update"
        # (ΔW ≈ 0) and training grows the update from there. A random init
        # would emit ||U V^T|| ~ sqrt(out·in)·r and destroy the model.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feats: torch.Tensor, layer_idx: torch.Tensor) -> torch.Tensor:
        xs = (feats - self.feat_mean) / self.feat_std
        emb = self.layer_emb(layer_idx)
        h = self.trunk(torch.cat([xs, emb], dim=-1))
        return self.head(h)


# ---------------------------------------------------------------------------
# Reconstruction loss
# ---------------------------------------------------------------------------
def recon_loss(full: torch.Tensor, layer_idx: torch.Tensor, segs: List[dict],
               offs: List[int], rank: int, targets: List[torch.Tensor]) -> torch.Tensor:
    """Mean relative Frobenius MSE over a batch of (features, layer) examples."""
    losses = []
    for i in range(full.shape[0]):
        delta = delta_from_row(full[i], int(layer_idx[i]), segs, offs, rank)
        t = targets[i].float()
        losses.append(((delta - t) ** 2).sum() / (t ** 2).sum().clamp_min(1e-12))
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_operator(X: torch.Tensor, layer_idx: torch.Tensor,
                   targets: List[torch.Tensor], segs: List[dict], rank: int,
                   in_dim: int, hidden: int = 128, layer_emb_dim: int = 16,
                   steps: int = 800, lr: float = 1e-3, batch_size: int = 0,
                   seed: int = 2345,
                   extra_loss_fn: Optional[Callable[[nn.Module, int], torch.Tensor]] = None,
                   extra_weight: float = 0.0, extra_frac: float = 0.8):
    """Meta-train the update operator on (features -> reconstructed ΔW).

    Returns (operator, history). `extra_loss_fn(operator, step)` returns an
    additional differentiable loss (e.g. behavioural validation loss), only
    used during the last (1 - extra_frac) fraction of steps.
    """
    torch.manual_seed(seed)
    n = X.shape[0]
    feat_mean = X.mean(dim=0)
    feat_std = X.std(dim=0).clamp_min(1e-8)
    offs = offsets_of(segs, rank)
    gen = total_gen(segs, rank)
    operator = UpdateOperator(in_dim, hidden, layer_emb_dim, len(segs), gen,
                              feat_mean, feat_std)
    opt = torch.optim.Adam(operator.parameters(), lr=lr)
    history = []

    def _step_losses(batch_idx):
        xb = X[batch_idx]
        ib = layer_idx[batch_idx]
        full = operator(xb, ib)
        return recon_loss(full, ib, segs, offs, rank,
                          [targets[i] for i in batch_idx])

    start_extra = int(steps * extra_frac)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        idx = (torch.randperm(n) if batch_size <= 0
               else torch.randint(0, n, (min(batch_size, n),)))
        loss = _step_losses(idx)
        if extra_loss_fn is not None and extra_weight > 0 and step >= start_extra:
            loss = loss + extra_weight * extra_loss_fn(operator, step)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                train_mse = float(_step_losses(torch.arange(n)).item())
                extra = 0.0
                if extra_loss_fn is not None and step >= start_extra:
                    extra = float(extra_loss_fn(operator, step).item())
            history.append({"step": step, "train_rel_mse": train_mse,
                            "extra_loss": extra})
    return operator, history


# ---------------------------------------------------------------------------
# Application (no future information)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_deltas(operator: UpdateOperator, record: Dict, K: int, rank: int,
                    segs: List[dict], feature_set: str, lr: float,
                    layers: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    """Predict ΔW for every layer from steps-1..K information only."""
    from src.phase4.features import build_features
    if layers is None:
        layers = [s["name"] for s in segs]
    operator.eval()
    feats = torch.stack([build_features(record, K, layer, feature_set, lr)
                         for layer in layers])
    idx = torch.arange(len(layers), dtype=torch.long)
    full = operator(feats, idx)
    offs = offsets_of(segs, rank)
    deltas = {}
    for i, layer in enumerate(layers):
        deltas[layer] = delta_from_row(full[i], i, segs, offs, rank)
    return deltas