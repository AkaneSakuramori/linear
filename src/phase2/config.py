"""Phase 2 experiment configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Phase2Config:
    phase1_config: str = "configs/baseline.yaml"     # reuses Phase-1 hyper-parameters
    out_dir: str = "results/phase2"
    reference_trajectory_dir: str = "results/trajectory"  # Phase-1 W_k snapshots

    horizons: list = field(default_factory=lambda: [5, 10, 25, 50, 100])
    use_horizon_100: bool = True                      # also treat full 100-step run

    alpha_grid: str = "logspace(-4, 2, 25)"           # eval string -> np.array
    ranks: list = field(default_factory=lambda: [4, 16, 64, 128, 256])
    momentum_beta: float = 0.9

    threads: int = 4                                  # must match Phase-1 baseline
    eval_batch: int = 32
    eval_max_windows: int = 1024

    methods: list = field(default_factory=lambda: [
        "BaselineAdamW",
        "DirectOracle",
        "DirectGradient",
        "DirectAverageGradient",
        "DirectMomentum",
        "DirectLowRank",
    ])
    seed: int = 7

    def alpha_grid_values(self) -> np.ndarray:
        return np.logspace(-4.0, 2.0, 25)

    def resolved_dict(self) -> dict:
        d = asdict(self)
        return d