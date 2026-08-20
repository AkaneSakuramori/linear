"""Phase-5 structured update generator.

Architecture (compressed / latent generation):
    global+per-layer features + layer embedding
        -> shared feature encoder -> latent z_l
        -> small factor generator -> coefficients
        -> apply onto learned per-layer bases
    ΔW_l = U_l V_l^T,  U_l = P_l C_l,  V_l = Q_l D_l     (2-D weights)
    Δw_l = B_l c_l                                        (1-D tensors)

P_l (out×m), Q_l (in×m), B_l (dim×m) are LEARNED per-layer bases (a dictionary
of directions shared across trajectories — NOT observed gradients). The network
predicts only the small coefficient matrices C_l, D_l (m×r) / c_l (m), so the
number of parameters the meta-learner must produce is tiny. The output head is
zero-initialized so the generator starts at "no update".

Objectives (A/B/C):
  A recon       Σ_l ‖ΔW_pred_l − ΔW_target_l‖² / ‖ΔW_target_l‖²
  B behavior    CE(W_K + ΔW_pred) on the TRAIN corpus (functional forward)
  C combined    A + λ·B
Validation-based checkpoint selection keeps the best meta-validation model.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _orth(d: int, m: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(d, m))
    return q


class StructuredGenerator(nn.Module):
    def __init__(self, in_dim: int, hidden: int, latent_dim: int,
                 layer_emb_dim: int, m_basis: int, rank: int, segs: List[dict],
                 feat_mean: torch.Tensor, feat_std: torch.Tensor):
        super().__init__()
        self.segs = segs
        self.m_basis = m_basis
        self.rank = rank
        self.in_dim = in_dim
        self.hidden = hidden
        self.latent_dim = latent_dim
        self.layer_emb_dim = layer_emb_dim
        self.register_buffer("feat_mean", feat_mean)
        self.register_buffer("feat_std", feat_std.clamp_min(1e-8))
        self.layer_emb = nn.Embedding(len(segs), layer_emb_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim + layer_emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim), nn.ReLU())
        coef_dim = 2 * m_basis * rank
        self.coef_gen = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, coef_dim))
        nn.init.zeros_(self.coef_gen[-1].weight)
        nn.init.zeros_(self.coef_gen[-1].bias)
        self.bases = nn.ParameterDict()
        self._base_key = {}
        for s in segs:
            if s["ndim"] == 2:
                for suf, shape in (("_U", (s["out"], m_basis)),
                                   ("_V", (s["in"], m_basis))):
                    key = s["name"] + suf
                    skey = key.replace(".", "_")
                    self._base_key[key] = skey
                    self.register_parameter(skey, nn.Parameter(_orth(*shape)))
            else:
                key = s["name"] + "_B"
                skey = key.replace(".", "_")
                self._base_key[key] = skey
                self.register_parameter(skey, nn.Parameter(_orth(s["in"], m_basis)))

    def _base(self, name: str, suf: str) -> torch.Tensor:
        return self.get_parameter(self._base_key[name + suf])

    def forward(self, feats: torch.Tensor, layer_idx: torch.Tensor):
        xs = (feats - self.feat_mean) / self.feat_std
        emb = self.layer_emb(layer_idx)
        z = self.encoder(torch.cat([xs, emb], dim=-1))
        coefs = self.coef_gen(z)
        return z, coefs

    def delta_for_one(self, coefs: torch.Tensor, i: int) -> torch.Tensor:
        s = self.segs[i]
        m, r = self.m_basis, self.rank
        if s["ndim"] == 2:
            U = self._base(s["name"], "_U") @ coefs[:m * r].view(m, r)
            V = self._base(s["name"], "_V") @ coefs[m * r:2 * m * r].view(m, r)
            return U @ V.t()
        return self._base(s["name"], "_B") @ coefs[:m]

    def deltas_for(self, coefs: torch.Tensor, layer_idx: torch.Tensor) -> List[torch.Tensor]:
        return [self.delta_for_one(coefs[i], int(layer_idx[i]))
                for i in range(coefs.shape[0])]


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------
def recon_loss(deltas: List[torch.Tensor],
               targets: List[torch.Tensor]) -> torch.Tensor:
    losses = []
    for d, t in zip(deltas, targets):
        losses.append(((d - t.float()) ** 2).sum() / (t.float() ** 2).sum().clamp_min(1e-12))
    return torch.stack(losses).mean()


def make_behavior_loss_fn(records: List[Dict], K: int, rank: int, segs: list,
                          feature_set: str, lr: float, phase1_cfg, gen: nn.Module,
                          train_ids: torch.Tensor, windows: int = 64,
                          batch: int = 32) -> Callable[[], torch.Tensor]:
    """Behavioral objective: CE of W_K + ΔW_pred on train ids (no future info).

    Each call samples one meta-train trajectory's features (K only) to build the
    predicted model, so the behavioural signal sees a diversity of trajectories.
    """
    from src.phase4.functional import state_cross_entropy
    rng = torch.Generator()

    def fn() -> torch.Tensor:
        rec = records[int(torch.randint(0, len(records), (1,), generator=rng).item())]
        deltas = _all_layer_deltas(gen, rec, K, rank, segs, feature_set, lr)
        w_pred = {name: rec["w_states"][K][name] + deltas[name]
                  for name in deltas}
        return state_cross_entropy(w_pred, train_ids, phase1_cfg.context_length,
                                   phase1_cfg.d_model, phase1_cfg.n_layer,
                                   phase1_cfg.n_head, phase1_cfg.ffn_mult,
                                   batch, windows)

    return fn


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_deltas(gen: StructuredGenerator, record: Dict, K: int, rank: int,
                    segs: List[dict], feature_set: str, lr: float) -> Dict[str, torch.Tensor]:
    from src.phase4.features import build_features
    gen.eval()
    layers = [s["name"] for s in segs]
    feats = torch.stack([build_features(record, K, l, feature_set, lr) for l in layers])
    idx = torch.arange(len(layers), dtype=torch.long)
    _, coefs = gen(feats, idx)
    deltas = gen.deltas_for(coefs, idx)
    return {layers[i]: deltas[i] for i in range(len(layers))}


def _all_layer_deltas(gen: StructuredGenerator, record: Dict, K: int, rank: int,
                      segs: list, feature_set: str, lr: float) -> Dict[str, torch.Tensor]:
    """Grad-enabled all-layer delta generation (for behavior objective)."""
    from src.phase4.features import build_features
    layers = [s["name"] for s in segs]
    feats = torch.stack([build_features(record, K, l, feature_set, lr) for l in layers])
    idx = torch.arange(len(layers), dtype=torch.long)
    _, coefs = gen(feats, idx)
    deltas = gen.deltas_for(coefs, idx)
    return {layers[i]: deltas[i] for i in range(len(layers))}


# ---------------------------------------------------------------------------
# training with objectives + validation-based checkpoint selection
# ---------------------------------------------------------------------------
def train_generator(X: torch.Tensor, layer_idx: torch.Tensor,
                    targets: List[torch.Tensor], segs: List[dict], rank: int,
                    in_dim: int, hidden: int = 128, latent_dim: int = 64,
                    layer_emb_dim: int = 16, m_basis: int = 32,
                    steps: int = 600, lr: float = 1e-3, weight_decay: float = 1e-4,
                    batch_size: int = 0, seed: int = 3456,
                    objective: str = "combined", lambda_b: float = 0.1,
                    behavior: Optional[dict] = None,
                    update_norm_reg: float = 0.0,
                    val_X: Optional[torch.Tensor] = None,
                    val_idx: Optional[torch.Tensor] = None,
                    val_targets: Optional[List[torch.Tensor]] = None,
                    validate_every: int = 100):
    torch.manual_seed(seed)
    n = X.shape[0]
    feat_mean = X.mean(dim=0)
    feat_std = X.std(dim=0).clamp_min(1e-8)
    gen = StructuredGenerator(in_dim, hidden, latent_dim, layer_emb_dim, m_basis,
                              rank, segs, feat_mean, feat_std)
    behavior_fn = None
    if behavior is not None:
        behavior_fn = make_behavior_loss_fn(
            behavior["records"], behavior["K"], rank, segs, behavior["feature_set"],
            behavior["lr"], behavior["phase1_cfg"], gen, behavior["train_ids"],
            windows=behavior.get("windows", 64), batch=32)
    opt = torch.optim.Adam(gen.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    best_state, best_val = None, float("inf")

    def _deltas(batch_idx):
        ib = layer_idx[batch_idx]
        _, coefs = gen(X[batch_idx], ib)
        return gen.deltas_for(coefs, ib)

    def _recon(batch_idx):
        return recon_loss(_deltas(batch_idx), [targets[i] for i in batch_idx])

    def _val_recon():
        with torch.no_grad():
            _, coefs = gen(val_X, val_idx)
            ds = gen.deltas_for(coefs, val_idx)
            return float(recon_loss(ds, val_targets).item())

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        idx = (torch.randperm(n) if batch_size <= 0
               else torch.randint(0, n, (min(batch_size, n),)))
        loss = torch.zeros((), dtype=torch.float32)
        if objective in ("recon", "combined"):
            loss = _recon(idx)
        if objective == "behavior" and behavior_fn is not None:
            loss = behavior_fn()
        elif objective == "combined" and behavior_fn is not None and lambda_b > 0:
            loss = loss + lambda_b * behavior_fn()
        if update_norm_reg > 0:
            ds = _deltas(idx)
            reg = sum((d ** 2).sum() for d in ds) / max(len(ds), 1)
            loss = loss + update_norm_reg * reg
        loss.backward()
        opt.step()

        if step % validate_every == 0 or step == steps - 1:
            with torch.no_grad():
                tr = float(_recon(torch.arange(n)).item())
                va = _val_recon() if val_X is not None else None
                if va is not None and va < best_val:
                    best_val = va
                    best_state = {k: v.clone() for k, v in gen.state_dict().items()}
            history.append({"step": step, "train_rel_mse": tr, "val_rel_mse": va})

    if best_state is not None:
        gen.load_state_dict(best_state)
    return gen, history, best_val
