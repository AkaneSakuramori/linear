"""The Direct Update Predictor: a small MLP from per-layer features to the
coefficients alpha_{l,j} of the structured parameter transformation.

The network is *shared across layers*: every (trajectory, layer) is one
training example, which multiplies the effective meta-training set size and
lets the predictor generalise to unseen layers and trajectories. Features and
targets are standardised on the meta-train split only (no leakage).
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictorNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor:
    def __init__(self, net: PredictorNet, feat_mean: torch.Tensor,
                 feat_std: torch.Tensor, target_mean: torch.Tensor,
                 target_std: torch.Tensor):
        self.net = net
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.target_mean = target_mean
        self.target_std = target_std

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict alpha coefficients. X: (N, in_dim). Output: (N, out_dim)."""
        self.net.eval()
        xs = (X - self.feat_mean) / self.feat_std
        with torch.no_grad():
            y = self.net(xs)
        return y * self.target_std + self.target_mean

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "net": self.net.state_dict(),
            "feat_mean": self.feat_mean,
            "feat_std": self.feat_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }, path)
        return path

    @classmethod
    def load(cls, path: str, in_dim: int, hidden: int, out_dim: int) -> "Predictor":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        net = PredictorNet(in_dim, hidden, out_dim)
        net.load_state_dict(payload["net"])
        return cls(net, payload["feat_mean"], payload["feat_std"],
                   payload["target_mean"], payload["target_std"])


def _mse(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.mse_loss(pred, y).item())


def train_predictor(X: torch.Tensor, Y: torch.Tensor,
                    X_val: Optional[torch.Tensor] = None,
                    Y_val: Optional[torch.Tensor] = None,
                    hidden: int = 64, steps: int = 2000, lr: float = 1e-3,
                    batch_size: int = 32, seed: int = 1234):
    """Meta-train the predictor on (features, alpha-star) examples.

    Returns (Predictor, history) where history records train/val MSE in the
    original (unstandardised) target scale every 100 steps.
    """
    torch.manual_seed(seed)
    in_dim = X.shape[1]
    out_dim = Y.shape[1]
    n = X.shape[0]

    feat_mean = X.mean(dim=0)
    feat_std = X.std(dim=0).clamp_min(1e-8)
    target_mean = Y.mean(dim=0)
    target_std = Y.std(dim=0).clamp_min(1e-8)

    Xs = (X - feat_mean) / feat_std
    Ys = (Y - target_mean) / target_std
    Xs_val = (X_val - feat_mean) / feat_std if X_val is not None else None
    Ys_val = (Y_val - target_mean) / target_std if Y_val is not None else None

    net = PredictorNet(in_dim, hidden, out_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history = []

    def _unstd(y: torch.Tensor) -> torch.Tensor:
        return y * target_std + target_mean

    batch = min(batch_size, n)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        idx = torch.randperm(n)[:batch]
        loss = F.mse_loss(net(Xs[idx]), Ys[idx])
        loss.backward()
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                tr = _mse(_unstd(net(Xs)), Y)
                va = None
                if Xs_val is not None:
                    va = _mse(_unstd(net(Xs_val)), Y_val)
            history.append({"step": step, "train_mse": tr, "val_mse": va})

    predictor = Predictor(net, feat_mean, feat_std, target_mean, target_std)
    return predictor, history
