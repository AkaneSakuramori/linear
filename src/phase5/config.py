"""Phase 5 experiment configuration.

Primary question: can a learned update operator GENERALIZE its
parameter-transformation strategy to unseen training trajectories (unseen seed,
unseen batch ordering, unseen initialization, unseen corpus)?

Defaults are CPU-friendly (≤4 threads) and target short horizons first.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Phase5Config:
    phase1_config: str = "configs/baseline.yaml"
    out_dir: str = "results/phase5"

    # --- data splits (unseen seeds) ------------------------------------------
    # train_16 = seeds[0:16]; train_32 = seeds[0:32] (a prefix relationship)
    meta_train_seeds: list = field(default_factory=lambda: list(range(10, 42)))  # 32
    meta_val_seeds: list = field(default_factory=lambda: list(range(42, 50)))   # 8
    meta_test_seeds: list = field(default_factory=lambda: list(range(50, 58)))  # 8 (unseen seed A)
    reference_seed: int = 7

    # decoupled init/data seeds for generalization tests B (batch order) & C (init)
    unseen_data_seeds: list = field(default_factory=lambda: [60, 61])
    unseen_init_seeds: list = field(default_factory=lambda: [62, 63])

    # --- task grid -------------------------------------------------------------
    pairs: list = field(default_factory=lambda: [(10, 25), (15, 25), (25, 50)])
    max_steps: int = 50
    k_max: int = 25
    record_steps: list = field(default_factory=lambda: [10, 15, 25, 50])

    # --- oracle -----------------------------------------------------------------
    oracle_ranks: list = field(default_factory=lambda: [1, 2, 4, 8])

    # --- generator ----------------------------------------------------------------
    # 'compressed' = shared encoder -> latent z_l -> coefficients onto learned
    #                per-layer bases (primary); 'direct' = Phase-4 flat MLP (ablation)
    arch: str = "compressed"
    rank: int = 4
    learned_ranks: list = field(default_factory=lambda: [4, 8])
    hidden: int = 128
    latent_dim: int = 64
    m_basis: int = 32          # per-layer basis width (compressed arch)
    layer_emb_dim: int = 16
    feature_set: str = "full"
    train_steps: int = 600
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 0        # 0 => full batch
    seed: int = 3456
    update_norm_reg: float = 0.0   # optional penalty on ||ΔW_pred||
    early_stopping_steps: int = 100  # validate every N steps; keep best by meta-val recon

    # --- objectives ---------------------------------------------------------------
    # objective: 'recon' | 'behavior' | 'combined'  (A / B / C)
    objective: str = "combined"
    lambda_b: float = 0.1      # weight on behavior in combined
    behavior_windows: int = 64
    # ablation objective settings
    objective_ablation: list = field(default_factory=lambda:
                                     [("recon", 0.0), ("combined", 0.01),
                                      ("combined", 0.1), ("combined", 1.0),
                                      ("behavior", 1.0)])
    # --- ablations -----------------------------------------------------------------
    data_size_ablation: list = field(default_factory=lambda: [6, 16, 32])
    arch_ablation: bool = True
    behavior_note: str = "behavioural term uses TRAIN corpus CE (avoids eval-corpus leakage)"

    # --- generalization test trajectories counts ------------------------------------
    corpus_b_test_seeds: list = field(default_factory=lambda: [70, 71, 72, 73])

    # --- evaluation -------------------------------------------------------------
    eval_max_windows: int = 1024
    threads: int = 4

    def resolved_dict(self) -> dict:
        return asdict(self)
