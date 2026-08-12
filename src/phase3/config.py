"""Phase 3 experiment configuration.

The Direct Update Predictor is meta-trained on conventional AdamW trajectories
run from different random seeds, then applied as a single parameter
transformation starting from an observed state W_K. See README (Phase 3) and
results/phase3/phase3_report.md for the full design.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Phase3Config:
    # --- reuse Phase-1 hyper-parameters (same tiny model, same corpus) ------
    phase1_config: str = "configs/baseline.yaml"

    # --- experiment directory ------------------------------------------------
    out_dir: str = "results/phase3"

    # --- data splits (different training seeds => different init + batches) --
    meta_train_seeds: list = field(default_factory=lambda: [10, 11, 12, 13, 14, 15])
    meta_val_seeds: list = field(default_factory=lambda: [20, 21])
    meta_test_seeds: list = field(default_factory=lambda: [30, 31, 32, 33])
    reference_seed: int = 7          # the Phase-1 reference trajectory (held out)

    # --- task grid -----------------------------------------------------------
    obs_horizons: list = field(default_factory=lambda: [5, 10])      # K
    target_horizons: list = field(default_factory=lambda: [25, 50, 100])  # H > K
    max_steps: int = 100              # every trajectory is rolled out to H_MAX

    # --- observation window ---------------------------------------------------
    k_max: int = 10                   # per-step gradients recorded up to here

    # --- parameterisation -----------------------------------------------------
    # basis of observed-gradient directions used to reconstruct ΔW:
    #   'mean'            ΔW ≈ Σ_l α_l · mean(grad_1..K)
    #   'first_last'      ΔW ≈ Σ_l α_l1·grad_1 + α_l2·grad_K
    #   'first_last_mean' three basis vectors per layer
    #   'all'             all K observed gradients as basis vectors
    basis: str = "mean"

    # --- predictor network ----------------------------------------------------
    hidden: int = 64
    train_steps: int = 2000
    lr: float = 1e-3
    batch_size: int = 32
    predictor_seed: int = 1234

    # --- feature set (see features.py). 'full' is the primary first experiment.
    feature_set: str = "full"
    # feature ablations A-F (loss / grad / grad_loss / full / compressed / rich)

    # --- evaluation ------------------------------------------------------------
    eval_max_windows: int = 1024
    threads: int = 4

    # --- which ablations to run -----------------------------------------------
    ablation_feature_sets: list = field(default_factory=lambda: ["loss", "grad", "grad_loss", "full", "compressed", "rich"])
    ablation_bases: list = field(default_factory=lambda: ["mean", "first_last", "first_last_mean"])
    ablation_pairs: list = field(default_factory=lambda: [(5, 100)])

    def all_pairs(self) -> list:
        pairs = []
        for k in self.obs_horizons:
            for h in self.target_horizons:
                if h > k:
                    pairs.append((k, h))
        return pairs

    def resolved_dict(self) -> dict:
        return asdict(self)
