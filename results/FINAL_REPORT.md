# Direct Learning — Final Report (Phases 1–5)

**Status: CONCLUDED — APPROACH ABANDONED**

Date: 2026-08-17

This document summarizes the full Direct Learning investigation performed in
this repository (Phases 1–5). It records the hypotheses tested, the experiments
performed, the quantitative results, the failures, and the final conclusion.

> **Scope of the conclusion.** The conclusion applies only to the *methodology
> investigated in this repository*. It is **not** a claim that alternative
> approaches to neural-network training are impossible in general; it is a claim
> that the specific "learned / direct update operator" methodology tested here
> did not demonstrate sufficient generalization to replace iterative
> gradient-based optimization.

---

## 1. Project overview

The project explored a research hypothesis about neural-network training:

> Can a mechanism compute a *direct* parameter transformation — replacing many
> sequential optimizer steps — such that a model reaches comparable quality with
> substantially lower total computation?

The investigation was conducted strictly as R&D on a shared CPU-only VPS
(≤ 4 threads, no GPU), using a single tiny decoder-only Transformer
(~1.6M parameters) and a small deterministic synthetic corpus. Every phase
reported validation loss, perplexity, parameter distance, and **full compute
accounting** (observation, inference, parameter generation/update, meta-training,
and amortized totals). No future-information leakage was permitted into any
direct method; oracle experiments that used the future answer were always
labelled as such.

Reference baseline (Phase 1): AdamW, 100 steps, seed 7 → **validation loss
0.8373, perplexity 2.31**, ~2.7×10¹² training FLOPs.

---

## 2. Phase-by-phase summary

### Phase 1 — Conventional baseline

Established a reproducible AdamW baseline and recorded the full parameter
trajectory `W0 → … → W100` (bit-exact reproducible given a seed). This is the
reference that all direct methods were measured against.

**Result:** a trustworthy baseline (val loss 0.8373) and a determinism guarantee.

### Phase 2 — Direct approximations (no learning)

Tested whether simple mathematical shortcuts — scaled first gradient, average
gradient, momentum direction, and low-rank versions of those — could replace the
100-step baseline with a single update.

**Key findings**
- The 100-step cumulative update `ΔW = W100 − W0` is **not** well aligned with
  the first gradient (cos ≈ −0.15), the mean gradient (cos ≈ −0.40), or the
  momentum direction (cos ≈ −0.13).
- `ΔW` is **dense** (~99.5% of parameters move) and **high-rank** (effective rank
  ≈ 180–230 at 95% energy) — no low-rank shortcut at small rank.
- Every tested single-update approximation reached val loss ≈ 2.4–3.5 vs the
  baseline's **0.84**, even those that spent the same N forward/backward passes.
- The **oracle** (one update that knows `W_N`) reproduces the final model
  exactly — an upper bound.

**Conclusion:** fixed linear combinations of local gradients are insufficient.
The open question became whether a *learned, non-linear* mechanism could do
better. → Recommendation: MODIFY (proceed to a learned predictor).

### Phase 3 — Learned per-layer scaling predictor

Tested a small neural network that maps early-training statistics to per-layer
scaling coefficients for a basis of **observed-gradient directions**
(`ΔW = Σ α_l D_l`), trained by meta-learning on multiple AdamW trajectories.

**Key findings**
- The predictor was restricted to the span of observed gradients.
- Crucially, **even the oracle** (perfect coefficients chosen with the future
  answer) could not approach AdamW quality within this family
  (val ≈ 2.0–3.3 vs conventional 0.84–1.52 across the K/H grid).
- More observation (K=10 vs K=5) helped but did not bridge the gap.

**Conclusion:** the bottleneck was the **parameterization family** (linear in
observed gradients), not predictor capacity. → Recommendation: MODIFY (move to a
learned operator that *generates* new directions).

### Phase 4 — Generated low-rank update operator

Tested a learned operator that **generates** per-layer low-rank factors
`ΔW_l = U_l V_l^T` rather than scaling observed gradients.

**Key findings**
- **Oracle**: the low-rank family is *informative* — best rank-8 SVD of the true
  update captures ~54–57% of the update energy and improves val loss
  monotonically (e.g. K=10, H=25: no-update 2.05 → oracle-r8 **1.64** vs
  conventional 1.52). But it does **not** reach conventional quality at r ≤ 8.
- **Learned operator**: overfits its 6 meta-training trajectories
  (reconstruction MSE 0.44 on train) but **generalizes to nothing** on held-out
  seeds (relative prediction error ~1.0; emits near-zero updates; quality ≈
  no-update).

**Conclusion:** the bottleneck was **learning/generalization**, not the family
and not compute. → Recommendation: MODIFY (scale the meta-training data, use a
structured generator, and test generalization explicitly).

### Phase 5 — Generalizable update operator

Tested whether scaling the meta-training data (6→16→32), a structured
**compressed-basis generator**, multiple objectives (A parameter-reconstruction,
B behavioural, C combined), and explicit generalization tests (A unseen seed,
B unseen batch order, C unseen initialization, D unseen corpus) would make the
operator generalize.

**Key findings**
- **The learned operator did NOT generalize.** Across all data sizes, objectives,
  architectures, ranks, horizons, and generalization tests, the learned update
  remained at the **no-update baseline**:
  - % of oracle improvement recovered: **~1–4%** on every test.
  - cos(ΔW_pred, ΔW_target) ≈ **0.03** — the predicted update was essentially
    orthogonal to the true update.
  - Increasing data from 6 → 32 trajectories did **not** help.
- The **oracle** still showed the low-rank family is informative (rank-8 ~56%
  energy, getting close to conventional on short horizons), so the failure is the
  **learned prediction**, not the family.

**Conclusion:** the learned update operator, even with substantial extra data, a
structured generator, and a second corpus, does not generalize to unseen
trajectories. → Recommendation: STOP THIS APPROACH.

---

## 3. Why the methodology failed

Across Phases 3–5 the same fundamental obstacle recurred:

1. **The true multi-step update is high-rank and step-dependent.** Phase 2
   measured an effective rank of ~180–230 and showed the update direction rotates
   during training. A compact (low-rank) generated update cannot span it.
2. **Early-training statistics do not determine the future update.** A generator
   conditioned only on steps 0…K must predict factors for steps K…H. Empirically
   the mapping is not learnable from the available meta-training data — the
   learned operators collapsed to predicting ≈ zero (no update) on unseen
   trajectories, recovering only ~1–4% of the available oracle improvement.
3. **More data and better architectures did not fix it.** Scaling the
   meta-training set 6→32, switching to a compressed-basis generator, and adding
   behavioural objectives all left the operator at the no-update baseline.

The **oracle** experiments consistently showed that the *family* of updates is
partly informative (it can improve on doing nothing), but the *learned* operator
could not reproduce that improvement on held-out trajectories. The failure is in
the learned prediction, and it persisted across every variation tested.

## 4. Compute accounting summary

Every phase reported compute honestly and separately:

| bucket | what it includes |
|---|---|
| Conventional training | forward/backward steps, FLOPs, wall time |
| Direct application | observation (K steps), generator/operator inference, parameter generation (U V^T), parameter update, total FLOPs |
| Meta-training | trajectory generation + predictor/operator training (reported, not hidden) |
| Amortization | total cost over N = 1, 10, 100, 1000 applications |

In no phase did the direct method reach conventional quality, so no compute
advantage was ever claimed on the basis of the compute accounting alone.

## 5. Metrics reported

- Training loss, validation loss, perplexity
- Relative parameter distance to the conventional target `W_H`
- Relative update prediction error (`‖ΔW_pred − ΔW_target‖/‖ΔW_target‖`)
- Update norm
- Explained update energy (oracle)
- **% of oracle improvement recovered** (Phase 5) — the key generalization metric
- Cosine similarity between predicted and target updates (Phase 5)
- Full compute accounting (FLOPs, forward/backward counts, wall time, RAM)

## 6. Conclusion

**CONCLUDED — APPROACH ABANDONED.**

The Direct Learning methodology as investigated in this repository — using a
learned (direct) update operator to replace iterative gradient-based
optimization — did **not** demonstrate sufficient generalization to replace
conventional training. On every held-out test, the learned operators stayed at
the no-update baseline, recovering at most a few percent of the improvement that
the (future-aware) oracle shows is available.

This conclusion is specific to the methodology tested here (early-statistics →
generated parameter update). It is **not** a claim that alternative
neural-network training approaches are impossible in general.

## 7. Final status of phases

| Phase | Focus | Outcome | Recommendation |
|---|---|---|---|
| 1 | Conventional baseline | Succeeded | — |
| 2 | Direct approximations (no learning) | Failed | MODIFY → learned predictor |
| 3 | Learned per-layer scaling | Failed (family-limited) | MODIFY → generated directions |
| 4 | Generated low-rank operator | Failed (no generalization) | MODIFY → scale data + structured gen |
| 5 | Generalizable operator | Failed (no generalization) | **STOP THIS APPROACH** |

All phase reports and raw data are preserved under `results/phase2/`,
`results/phase3/`, `results/phase4/`, and `results/phase5/`. No Phase 6 was
implemented.
