"""Phase 4 experiment configuration.

Phase 4 tests a learned update operator that *generates* new per-layer
parameter directions as low-rank factors ΔW_l = U_l V_l^T (rather than scaling
observed gradients like Phase 3). Priorities per the research plan: short
horizons first (K=10/15/25 → H=25), rank 1-8, CPU-only, ≤4 threads.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Phase4Config:
    # --- reuse Phase-1 hyper-parameters -------------------------------------
    phase1_config: str = "configs/baseline.yaml"

    out_dir: str = "results/phase4"

    # --- data splits (seeds => different init + batch sampling) --------------
    meta_train_seeds: list = field(default_factory=lambda: [10, 11, 12, 13, 14, 15])
    meta_val_seeds: list = field(default_factory=lambda: [20, 21])
    meta_test_seeds: list = field(default_factory=lambda: [30, 31, 32, 33])
    reference_seed: int = 7

    # --- task grid ------------------------------------------------------------
    # (K, H) pairs; (25, 25) is a zero-target control (ΔW = 0).
    pairs: list = field(default_factory=lambda: [(10, 25), (15, 25), (25, 25),
                                                 (10, 50), (25, 50)])
    max_steps: int = 100
    k_max: int = 25
    record_steps: list = field(default_factory=lambda: [10, 15, 25, 50, 100])

    # --- oracle (best rank-r of ΔW_target via SVD) ---------------------------
    oracle_ranks: list = field(default_factory=lambda: [1, 2, 4, 8])

    # --- learned operator -----------------------------------------------------
    learned_ranks: list = field(default_factory=lambda: [1, 2, 4])
    default_rank: int = 4
    feature_set: str = "full"          # see features.py
    hidden: int = 128
    layer_emb_dim: int = 16
    train_steps: int = 800
    lr: float = 1e-3
    batch_size: int = 0                # 0 => full-batch gradient descent
    operator_seed: int = 2345
    # optional behavioural (validation-loss) term in the meta-training objective
    behavior_weight: float = 0.0
    behavior_windows: int = 64
    behavior_steps: int = 150

    # --- evaluation -----------------------------------------------------------
    eval_max_windows: int = 1024
    threads: int = 4

    # --- ablations ------------------------------------------------------------
    ablation_ranks: list = field(default_factory=lambda: [1, 2, 4])
    ablation_feature_sets: list = field(default_factory=lambda:
                                        ["loss", "grad", "grad_loss", "full"])
    ablation_pairs: list = field(default_factory=lambda: [(10, 25)])
    # update-structure comparison reuses the Phase-3 predictor (K=10 pairs only)

    def resolved_dict(self) -> dict:
        return asdict(self)
