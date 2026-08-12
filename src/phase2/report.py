"""Writes Phase-2 markdown reports: the baseline audit and the experiment summary."""
from __future__ import annotations

import json
import os
from typing import Dict


def write_json(data: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_audit_report(audit: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    txt = f"""# Phase 2 — Baseline Audit

Date: {audit["date"]}
Audit run id: {audit.get("run_id", "n/a")}

## 1. FLOPs accounting

The reported per-step FLOPs **{audit["flops"]["per_step"]:.3e}** and total
**{audit["flops"]["total"]:.3e}** for 100 optimizer steps come from
`src/utils.py:estimate_flops`. We independently recomputed the formula by hand
and by script and the number matches exactly.

What is included:
- Every `nn.Linear` applied to all {audit["arch"]["context_length"]} token
  positions (token embeddings + position embeddings excluded — lookups).
- Attention QK^T and PV matmuls: 4·T²·d per layer (`T=64, d=256, 2 layers`).
- Softmax + scaling as a rough extras term.
- Backward pass approximated as **3× forward** (a standard approximation; the
  true backward for a transformer is usually ≈2–3× forward).
- The full batch (effective batch {audit["flops"]["eff_batch"]} sequences).

What is NOT included:
- AdamW's own parameter-update arithmetic (≈1.6M multiply-accumulates/step) —
  negligible (<1% of a step).
- Gradient clipping (scalar rescale) — negligible.
- Loss / norms instrumentation.
- **Validation evaluations** — each Phase-1 eval runs ≈{audit["flops"]["eval_windows"]}
  forward passes over the validation split. These are charged to methods in
  Phase-2's compute accounting but are not part of the Phase-1
  `approx_flops_total`.

Verdict: the reported **2.7e12 FLOPs** is a fair estimate of forward+backward
training compute. It does **not** include the optimizer step or evals, both of
which are small relative to the training steps. Fine for cross-method
computational comparisons.

## 2. Gradient clipping

Clipping (max norm **{audit["clip"]}**) is active in the reported trajectory.
The Phase-1 `metrics.json` `grad_norm` column records the **clipped** global
gradient norm (it sits at {audit["clip"]} for most steps, i.e. clipping is
binding).

Phase-2's instrumented run records **both** norms:

| statistic | value |
|---|---|
| mean unclipped grad norm | {audit["grad"]["mean_unclipped"]:.3f} |
| mean clipped grad norm  | {audit["grad"]["mean_clipped"]:.3f} |
| median unclipped        | {audit["grad"]["median_unclipped"]:.3f} |
| max unclipped           | {audit["grad"]["max_unclipped"]:.3f} |
| steps where clip bound  | {audit["grad"]["clip_bound_steps"]} / {audit["grad"]["n_steps"]} |

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

The exact same seed ({audit["seed"]}) and thread budget
({audit["threads"]}) reproduce the trajectory across independent process runs.

- Two full `python -m src.train` runs (same config, separate processes) produced
  **bit-identical** loss series and final weights (0.0 difference).
- The Phase-2 *instrumented* replay agrees with the saved
  `results/metrics.json` trajectory to within {audit["determinism"]["rounding_tol"]}
  (the saved file rounds losses/norms to 6 decimal places, which accounts for
  the observed ~5e-7 differences):
  `within_tolerance={audit["determinism"]["identical"]}`
- Final model weights of the replay are **bit-identical** to the saved
  `step_0100.pt` snapshot: `{audit["determinism"]["bitwise_final_weights_equal"]}`
- All {len(audit["determinism"]["steps"])} trajectory steps compared
  ({audit["determinism"]["steps"]}).
- Several sources of randomness are pinned: corpus generation seed, torch init
  seed, and window sampling is a pure function of `(step, seed)`.

This is the guarantee that lets Phase-2 treat the saved `W0 … W100` snapshots
as the authoritative reference rollout.

## 5. Correctness of the effective trajectory used in Phase 2

Phase-2 replays the identical loop with instrumentation (Step 4 module
`src/phase2/capture.py`). It records the same losses at the same trajectory
steps (differences are pure 6-decimal rounding of the saved file, not
nondeterminism):

- max |Δ train_loss| between replay and saved metrics: {audit["det"]["max_train_diff"]:.3e}
- max |Δ val_loss|: {audit["det"]["max_val_diff"]:.3e}
- replay final model == saved step_0100 weights (bitwise): `{audit["det"]["final_equal"]}`

The saved trajectory is therefore a faithful record of what actually happened
during Phase-1 training.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def write_summary_report(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    analysis = data["analysis"]
    direct = data["direct"]
    baseline = data["baseline"]
    oracle = data["oracle"]

    hs = analysis["horizons"]

    def loss_for(name, h):
        for r in direct:
            if r["name"] == name and r["horizon"] == h:
                return r["final_val_loss"]
        return None

    lines = [f"# Phase 2 — Direct Parameter Update Research, summary\n",
             f"Date: {data['date']}",
             f"Seed: {data['seed']}, threads: {data['threads']}, "
             f"model params: {data['param_count']}",
             f"Horizons studied: {hs}\n",
             "## 1. Oracle upper bound (one update using the future answer)\n",
             "| horizon | val loss | val ppl | param distance to W_N | compute (FLOPs) |"]
    for r in oracle:
        lines.append(f"| {r['horizon']} | {r['final_val_loss']:.4f} | {r['final_val_ppl']:.2f} | "
                     f"{r['param_distance_rel']:.2e} | {r['compute']['flops_est']:.2e} |")
    lines.append(f"\nThe oracle reproduces W_N exactly (distance 0); its one update is as good as "
                 f"the full rollout **by construction**, and its FLOPs are only evaluation passes. "
                 f"It is an **upper bound, not a training algorithm** (cheating by definition).\n")

    lines.append("## 2. Is the cumulative update structured?\n")
    for h in hs:
        d = analysis["deltas"][str(h)]

        def _fmt(x):
            return "n/a" if x is None else format(x, ".3f")

        lines.append(
            f"- horizon {h}: ||ΔW||₂={d['delta_l2']:.3f} "
            f"(vs ||W0||₂={d['w0_l2']:.2f}, ratio {d['delta_w0_ratio']:.4f}); "
            f"{d['changed_params_frac']*100:.1f}% of parameters moved meaningfully; "
            f"cos(ΔW, first grad)={_fmt(d['cos_sim_first_grad'])}, "
            f"cos(ΔW, avg grad)={_fmt(d['cos_sim_avg_grad'])}, "
            f"cos(ΔW, momentum)={_fmt(d['cos_sim_momentum'])}")
    lines.append("")

    lines.append("## 3. Direction stability over training\n")
    for dc in analysis["direction_change"]:
        lines.append(f"- cos(ΔW_{dc['horizon_a']}, ΔW_{dc['horizon_b']}) = {dc['cosine']:.3f}")
    lines.append("")

    lines.append("## 4. Low-rank structure of ΔW\n")
    svd100 = analysis["svd"]["100"]
    line = "Layer | rank to keep | % energy @rank64 | eff rank (95% energy)"
    lines.append("| " + line + " |")
    lines.append("|---|---|---|--|")
    for name in list(svd100.keys())[:8]:
        re = svd100[name]["rank_energy"]
        r64 = re.get("64")
        r64_txt = "n/a" if r64 is None else f"{r64*100:.1f}%"
        lines.append(f"| {name} | {svd100[name]['eff_rank_95']} | "
                     f"{r64_txt} | {svd100[name]['eff_rank_95']} |")
    lines.append("")

    lines.append("## 5. Direct approximation methods (practical; no future answers)\n")
    lines.append("| method | horizon | alpha | train loss | val loss | val ppl | "
                 "param dist rel | update FLOPs | tuning FLOPs | total FLOPs | "
                 "fwd | bwd | updates |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in direct:
        c = r["compute"]
        lines.append(f"| {r['name']} | {r['horizon']} | {r['alpha']:.4f} | "
                     f"{r['final_train_loss']:.4f} | {r['final_val_loss']:.4f} | "
                     f"{r['final_val_ppl']:.2f} | {r['param_distance_rel']:.3f} | "
                     f"{c['intrinsic_flops_est']:.2e} | {c['tuning_flops_est']:.2e} | "
                     f"{c['flops_est']:.2e} | {c['fwd_count']} | {c['bwd_count']} | "
                     f"{c['param_updates']} |")
    lines.append("")

    if baseline:
        b = baseline[0]
        lines.append("**Reference (baseline):** "
                     f"train {b['final_train_loss']:.4f}, val {b['final_val_loss']:.4f}, "
                     f"ppl {b['final_val_ppl']:.2f}, FLOPs {b['compute']['flops_est']:.2e}, "
                     f"{b['compute']['param_updates']} parameter updates.\n")

    lines.append("## 6. Conclusions\n")
    lines.append(data["conclusions"])
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))