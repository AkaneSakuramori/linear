# Phase 2 — Baseline Audit

Date: 2026-08-12T00:41:20.566647+00:00
Audit run id: 20260812-004120

## 1. FLOPs accounting

The reported per-step FLOPs **2.699e+10** and total
**2.699e+12** for 100 optimizer steps come from
`src/utils.py:estimate_flops`. We independently recomputed the formula by hand
and by script and the number matches exactly.

What is included:
- Every `nn.Linear` applied to all 64 token
  positions (token embeddings + position embeddings excluded — lookups).
- Attention QK^T and PV matmuls: 4·T²·d per layer (`T=64, d=256, 2 layers`).
- Softmax + scaling as a rough extras term.
- Backward pass approximated as **3× forward** (a standard approximation; the
  true backward for a transformer is usually ≈2–3× forward).
- The full batch (effective batch 32 sequences).

What is NOT included:
- AdamW's own parameter-update arithmetic (≈1.6M multiply-accumulates/step) —
  negligible (<1% of a step).
- Gradient clipping (scalar rescale) — negligible.
- Loss / norms instrumentation.
- **Validation evaluations** — each Phase-1 eval runs ≈1024
  forward passes over the validation split. These are charged to methods in
  Phase-2's compute accounting but are not part of the Phase-1
  `approx_flops_total`.

Verdict: the reported **2.7e12 FLOPs** is a fair estimate of forward+backward
training compute. It does **not** include the optimizer step or evals, both of
which are small relative to the training steps. Fine for cross-method
computational comparisons.

## 2. Gradient clipping

Clipping (max norm **1.0**) is active in the reported trajectory.
The Phase-1 `metrics.json` `grad_norm` column records the **clipped** global
gradient norm (it sits at 1.0 for most steps, i.e. clipping is
binding).

Phase-2's instrumented run records **both** norms:

| statistic | value |
|---|---|
| mean unclipped grad norm | 3.111 |
| mean clipped grad norm  | 1.000 |
| median unclipped        | 2.957 |
| max unclipped           | 7.107 |
| steps where clip bound  | 100 / 100 |

Because `||g|| ≥ 1.0` on the large majority of steps, every Phase-1 update was
in fact applied in a direction rescaled to unit norm. Default phase direction
estimates in Phase-2 use the **unclipped** raw gradients and this clipping
behaviour is documented per method.

## 3. Checkpoint contents

`results/trajectory/step_XXXX.pt` contains exactly, one file per recorded step:

- `schema_version: 1`
- `step`: the optimizer step index (`0` = initialization `W0`)
- `arch`: the full `ModelConfig` dict (vocab size, layers, width, heads,
  context, ffn multiplier, dropout, tie_embeddings)
- `model_state_dict`: 25 float32 tensors (token/position embeddings, 2×
  (attention qkv/proj, layer norms, mlp fc1/fc2), ln_f, lm_head) — enough to
  reconstruct `Wk` exactly
- `metrics`: step, train_loss, train_loss_ema, val_loss, val_ppl, grad_norm
  (clipped), param_norm, lr, elapsed_sec, tokens_seen, approx_flops_so_far

The attention causal mask is a `persistent=False` buffer and is **not** stored
(it is recreated deterministically from config). No optimizer state is kept in
the trajectory files; `checkpoints/resume_latest.pt` holds optimizer +
scheduler + RNG state only for resume. Phase-2 loads trajectory files with
`torch.load(..., weights_only=True)`-safe payloads (only tensors + primitive
types).

## 4. Trajectory determinism

The exact same seed (7) and thread budget
(4) reproduce the trajectory across independent process runs.

- Two full `python -m src.train` runs (same config, separate processes) produced
  **bit-identical** loss series and final weights (0.0 difference).
- The Phase-2 *instrumented* replay agrees with the saved
  `results/metrics.json` trajectory to within 1e-05
  (the saved file rounds losses/norms to 6 decimal places, which accounts for
  the observed ~5e-7 differences):
  `within_tolerance=True`
- Final model weights of the replay are **bit-identical** to the saved
  `step_0100.pt` snapshot: `True`
- All 9 trajectory steps compared
  ([0, 1, 2, 5, 10, 25, 50, 75, 100]).
- Several sources of randomness are pinned: corpus generation seed, torch init
  seed, and window sampling is a pure function of `(step, seed)`.

This is the guarantee that lets Phase-2 treat the saved `W0 … W100` snapshots
as the authoritative reference rollout.

## 5. Correctness of the effective trajectory used in Phase 2

Phase-2 replays the identical loop with instrumentation (Step 4 module
`src/phase2/capture.py`). It records the same losses at the same trajectory
steps (differences are pure 6-decimal rounding of the saved file, not
nondeterminism):

- max |Δ train_loss| between replay and saved metrics: 4.691e-07
- max |Δ val_loss|: 4.016e-07
- replay final model == saved step_0100 weights (bitwise): `True`

The saved trajectory is therefore a faithful record of what actually happened
during Phase-1 training.
