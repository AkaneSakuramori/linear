"""Functional transformer forward pass from a state dict.

Used only for the optional behavioural term in the update operator's
meta-training objective: evaluate the validation loss of W_K + ΔW_pred while
keeping the autograd graph through the operator. The standard nn.Module path
would copy parameter data and lose the graph; this functional path does not.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from src.dataset import get_eval_starts


def _mask(T: int, device):
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def forward_from_state(state: Dict[str, torch.Tensor], x: torch.Tensor,
                       d_model: int, n_layer: int, n_head: int,
                       ffn_mult: int, context_length: int) -> torch.Tensor:
    """Logits for token ids `x` using weights in `state` (autograd-friendly)."""
    B, T = x.shape
    tok = F.embedding(x, state["tok_emb.weight"])
    pos = F.embedding(torch.arange(T, device=x.device), state["pos_emb.weight"])
    h = tok + pos
    for i in range(n_layer):
        ln1 = F.layer_norm(h, (d_model,), state[f"blocks.{i}.ln1.weight"],
                           state[f"blocks.{i}.ln1.bias"])
        qkv = ln1 @ state[f"blocks.{i}.attn.qkv.weight"].t()
        Bq, Tq, _ = qkv.shape
        qkv = qkv.view(Bq, Tq, 3, n_head, d_model // n_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (d_model // n_head) ** -0.5
        att = att.masked_fill(~_mask(T, att.device)[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).reshape(Bq, Tq, d_model)
        h = h + y @ state[f"blocks.{i}.attn.proj.weight"].t()
        ln2 = F.layer_norm(h, (d_model,), state[f"blocks.{i}.ln2.weight"],
                           state[f"blocks.{i}.ln2.bias"])
        inner = d_model * ffn_mult
        h = h + F.gelu(ln2 @ state[f"blocks.{i}.mlp.fc1.weight"].t()) @ \
            state[f"blocks.{i}.mlp.fc2.weight"].t()
    h = F.layer_norm(h, (d_model,), state["ln_f.weight"], state["ln_f.bias"])
    logits = h @ state["lm_head.weight"].t()
    return logits


def state_cross_entropy(state: Dict[str, torch.Tensor], ids: torch.Tensor,
                        context_length: int, d_model: int, n_layer: int,
                        n_head: int, ffn_mult: int, batch_size: int,
                        max_windows: int) -> torch.Tensor:
    """Mean cross-entropy over stride-sampled windows (autograd-friendly)."""
    starts = get_eval_starts(int(ids.numel()), context_length, max_windows)
    losses = []
    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        s = torch.tensor(chunk, dtype=torch.long)
        pos = s[:, None] + torch.arange(context_length, dtype=torch.long)
        x = ids[pos]
        y = ids[pos + 1]
        logits = forward_from_state(state, x, d_model, n_layer, n_head,
                                    ffn_mult, context_length)
        B, T, V = logits.shape
        losses.append(F.cross_entropy(logits.reshape(B * T, V),
                                      y.reshape(B * T)))
    return torch.stack(losses).mean()


def _generate_deltas_grad(operator, record, K: int, rank: int, segs: list,
                          feature_set: str, lr: float):
    """Grad-enabled delta generation (for the behavioural objective)."""
    from src.phase4.features import build_features
    from src.phase4.operator import delta_from_row, offsets_of
    layers = [s["name"] for s in segs]
    feats = torch.stack([build_features(record, K, layer, feature_set, lr)
                         for layer in layers])
    idx = torch.arange(len(layers), dtype=torch.long)
    full = operator(feats, idx)
    offs = offsets_of(segs, rank)
    return {layers[i]: delta_from_row(full[i], i, segs, offs, rank)
            for i in range(len(layers))}


def make_behavior_loss_fn(record: Dict, K: int, H: int, rank: int,
                          segs: list, feature_set: str, lr: float,
                          phase1_cfg, train_ids: torch.Tensor,
                          windows: int = 64, batch: int = 32):
    """Build the behavioural extra-loss callable for train_operator.

    Uses the TRAIN corpus (not the held-out val corpus) so that meta-training
    never touches the evaluation distribution. No future trajectory
    information is used: only steps-1..K features and W_K.
    """

    def fn(operator, step: int) -> torch.Tensor:
        deltas = _generate_deltas_grad(operator, record, K, rank, segs,
                                       feature_set, lr)
        w_pred = {name: record["w_states"][K][name] + deltas[name]
                  for name in deltas}
        return state_cross_entropy(
            w_pred, train_ids, phase1_cfg.context_length, phase1_cfg.d_model,
            phase1_cfg.n_layer, phase1_cfg.n_head, phase1_cfg.ffn_mult,
            batch, windows)

    return fn