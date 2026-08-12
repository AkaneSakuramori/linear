"""Evaluation helpers: validation loss/perplexity and gradient/parameter norms."""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F

from src.dataset import get_eval_starts
from src.model import Transformer


@torch.no_grad()
def evaluate_loss(model: Transformer, ids: torch.Tensor, context_length: int,
                  batch_size: int = 32, max_windows: int = 1024) -> Dict:
    """Average loss + perplexity over stride-sampled windows of `ids`."""
    model.eval()
    starts = get_eval_starts(int(ids.numel()), context_length, max_windows)
    loss_sum = 0.0
    n_seq = 0
    total_tokens = 0
    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        s = torch.tensor(chunk, dtype=torch.long)
        pos = s[:, None] + torch.arange(context_length, dtype=torch.long)
        x = ids[pos]
        y = ids[pos + 1]
        logits = model(x)
        loss = model.loss(logits, y)
        loss_sum += loss.item() * len(chunk)
        n_seq += len(chunk)
        total_tokens += len(chunk) * int(context_length)
    model.train()
    avg = loss_sum / max(1, n_seq)
    return {
        "loss": float(avg),
        "ppl": float(math.exp(avg)) if avg < 300 else float(1e9),
        "n_windows": int(n_seq),
        "n_tokens": int(total_tokens),
    }


def grad_and_param_norms(model: Transformer) -> Dict:
    """L2 norm of gradients (post-backward) and of parameters, plus grads."""
    grad_sq, param_sq = 0.0, 0.0
    for p in model.parameters():
        param_sq += float(p.data.double().pow(2).sum().item())
        if p.grad is not None:
            grad_sq += float(p.grad.double().pow(2).sum().item())
    grad_norm = math.sqrt(grad_sq) if grad_sq > 0 else None
    return {
        "grad_norm": grad_norm,
        "param_norm": math.sqrt(param_sq),
        "has_grads": grad_norm is not None,
    }