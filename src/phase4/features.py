"""Phase-4 feature construction.

Reuses the Phase-3 per-layer + global feature vectors (steps 1..K only, so no
future information) and adds one richer variant, `full_rich`, which appends the
per-layer gradient-norm history (K values) to the full Phase-3 feature set.

The feature vector is the ONLY input to the update operator besides the layer's
(learned) identity embedding. It intentionally contains statistics — norms,
ratios, cosines, history — never the raw gradient vectors, so any direction
structure in the generated U_l, V_l must come from learned operator weights
rather than being a direct copy of an observed gradient.
"""
from __future__ import annotations

import torch

from src.phase3.features import ALL_SETS, build_features as _build_features_p3
from src.phase3.features import feature_dim as _feature_dim_p3

EXTRA_SETS = ("full_rich",)


def feature_dim(feature_set: str, K: int) -> int:
    if feature_set in ALL_SETS:
        return _feature_dim_p3(feature_set, K)
    if feature_set == "full_rich":
        return _feature_dim_p3("full", K) + K
    raise ValueError(f"unknown feature set {feature_set}")


def build_features(record: dict, K: int, layer: str, feature_set: str,
                   lr: float) -> torch.Tensor:
    if feature_set in ALL_SETS:
        return _build_features_p3(record, K, layer, feature_set, lr)
    if feature_set == "full_rich":
        base = _build_features_p3(record, K, layer, "full", lr)
        grad_hist = torch.tensor(
            [record["grad_states"][s][layer].float().norm().item() for s in range(1, K + 1)],
            dtype=torch.float32)
        return torch.cat([base, grad_hist])
    raise ValueError(f"unknown feature set {feature_set}")