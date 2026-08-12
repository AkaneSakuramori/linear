"""Shared utilities: configuration, hardware report, threading, seeding, FLOPs estimation."""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Resolved experiment configuration.

    The YAML file contains the same fields; overrides can be applied from the
    CLI. `vocab_size` is left empty and resolved from the tokenizer at runtime.
    """

    # --- architecture ------------------------------------------------------
    n_layer: int = 2
    d_model: int = 256
    n_head: int = 4
    context_length: int = 64
    ffn_mult: int = 4
    dropout: float = 0.0
    tie_embeddings: bool = False
    vocab_size: Optional[int] = None

    # --- dataset -----------------------------------------------------------
    data_dir: str = "data"
    corpus_train_chars: int = 60000
    corpus_val_chars: int = 20000
    corpus_seed: int = 12345
    corpus_val_seed: int = 4242
    regenerate: bool = False

    # --- training ----------------------------------------------------------
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    batch_size: int = 32
    grad_accumulation: int = 1
    grad_clip: float = 1.0
    n_steps: int = 100
    seed: int = 7
    threads: int = 4
    device: str = "cpu"
    lr_schedule: str = "constant"  # constant | cosine
    warmup_steps: int = 0
    eval_max_windows: int = 1024
    log_every: int = 1

    # --- checkpoints / results ---------------------------------------------
    out_dir: str = "results"
    checkpoints_dir: str = "checkpoints"
    save_trajectory: bool = True
    resume: bool = False
    trajectory_steps: list = field(default_factory=lambda: [0, 1, 2, 5, 10, 25, 50, 75, 100])

    # --- experiment bookkeeping ---------------------------------------------
    experiment_name: str = "baseline"

    def resolved_dict(self, vocab_size: int) -> dict:
        d = asdict(self)
        d["vocab_size"] = vocab_size
        d["betas"] = list(self.betas)
        return d


def load_config(path: str) -> Config:
    import yaml  # deferred import keeps CLI helpers independent of yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - set(Config.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    cfg = Config()
    for k, v in raw.items():
        if k == "betas":
            v = tuple(v)
        setattr(cfg, k, v)
    return cfg


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Hardware report and thread configuration
# ---------------------------------------------------------------------------
def ram_available_gib() -> float:
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except (FileNotFoundError, ValueError):
        pass
    return 0.0


def peek_free_ram_mib() -> float:
    return ram_available_gib() * 1024.0


def print_hardware_report(threads: int) -> None:
    gpu = "cuda" if torch.cuda.is_available() else "none"
    print("Hardware:")
    print(f"CPU cores available: {os.cpu_count()}")
    print(f"CPU threads allocated: {threads}")
    print(f"RAM available: {ram_available_gib():.1f} GiB")
    print(f"GPU: {gpu}")


def configure_threads(threads: int) -> None:
    """Constrain PyTorch/BLAS thread usage. Must be called early."""
    threads = max(1, int(threads))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized; thread limit already applies


def resolve_device(prefer: str = "cpu") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------
def count_parameters(model: torch.nn.Module) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


# ---------------------------------------------------------------------------
# FLOPs estimation (approximate)
# ---------------------------------------------------------------------------
def estimate_flops(model: torch.nn.Module, context_length: int, n_steps: int,
                   effective_batch: int, backward_factor: float = 3.0) -> dict:
    """Rough analytic FLOP estimate for a decoder-only transformer.

    Counts every Linear layer (token embedding lookups excluded, logits head
    included) plus attention QK^T / PV matmuls and needs-rough softmax. The
    backward pass is approximated as `backward_factor` times forward. Only the
    same batch is accumulated; validation is small relative to training.
    """
    linear_macs = 0.0
    for _name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            linear_macs += m.in_features * m.out_features

    d = getattr(model, "cfg", None)
    if d is None:
        raise ValueError("estimate_flops requires model.cfg with d_model/n_layer/n_head")
    T = int(context_length)
    fwd_1seq = 2.0 * linear_macs * T
    fwd_1seq += d.n_layer * 4.0 * T * T * d.d_model      # QK^T + PV
    fwd_1seq += d.n_layer * 2.0 * T * T * d.n_head * 2.0  # softmax + scaling

    per_step = fwd_1seq * effective_batch * (1.0 + backward_factor)
    tokens_per_step = effective_batch * T
    return {
        "approx_flops_per_step": float(per_step),
        "approx_flops_total": float(per_step * n_steps),
        "approx_tokens_per_step": float(tokens_per_step),
        "approx_tokens_total": float(tokens_per_step * n_steps),
        "estimate_method": "analytic (Linear layers + attention matmuls; backward=3x forward)",
    }