"""Checkpoint save/load: model, optimizer, scheduler, architecture config.

Trajectory checkpoints (one per recorded step) store the model parameters
plus per-step metrics. A single rolling `resume_latest.pt` keeps optimizer +
scheduler + RNG state so a run can be stopped and resumed deterministically.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

from src.model import ModelConfig, Transformer

SCHEMA_VERSION = 1


def _path(directory: str, step: int) -> str:
    return os.path.join(directory, f"step_{step:04d}.pt")


def save_trajectory_step(directory: str, step: int, model: Transformer,
                         metrics: Dict) -> str:
    os.makedirs(directory, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": int(step),
        "arch": model.cfg.to_dict(),
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    path = _path(directory, step)
    torch.save(payload, path)
    return path


def load_trajectory_step(path: str) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig.from_dict(payload["arch"]))
    model.load_state_dict(payload["model_state_dict"])
    return {"model": model, "step": payload["step"], "metrics": payload["metrics"]}


def load_trajectory_step_last(directory: str) -> Optional[Dict]:
    if not os.path.isdir(directory):
        return None
    steps = []
    for fname in os.listdir(directory):
        if fname.startswith("step_") and fname.endswith(".pt"):
            try:
                steps.append(int(fname.split("_")[1].split(".")[0]))
            except ValueError:
                pass
    if not steps:
        return None
    return load_trajectory_step(_path(directory, max(steps)))


def save_resume(path: str, model: Transformer, optimizer, scheduler,
                step: int, seed: int, extra: Optional[Dict] = None) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": int(step),
        "arch": model.cfg.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "seed": int(seed),
        "torch_rng_state": torch.random.get_rng_state(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_resume(path: str, model: Transformer, optimizer, scheduler) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    torch.random.set_rng_state(payload["torch_rng_state"])
    return {
        "model": model,
        "step": payload["step"],
        "seed": payload["seed"],
        "extra": payload.get("extra") or {},
    }