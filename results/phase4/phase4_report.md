# Phase 4 — Generated Parameter Directions

Date: 2026-08-17T13:57:17.307305+00:00

> Research status: hypothesis under investigation — negative results reported honestly. Nothing here is claimed to beat AdamW or to scale.

## 1. Research question

Can a **learned update operator** *generate* new per-layer parameter directions (as low-rank factors ΔW_l = U_l V_l^T) — directions NOT restricted to the span of observed gradients — such that a model can approximate the effect of multiple future AdamW steps with substantially fewer training steps?

Phase 3's key result was that even the *oracle* coefficient selector for a per-layer-scaling family (linear in observed gradients) could not approach AdamW quality. Phase 4 removes that restriction: the operator emits U_l and V_l whose content is learned, not sampled from observed gradients.

## 2. Architecture

A single shared network (an MLP with a learned per-layer identity embedding) maps compact per-layer + global features (steps 1..K only) to the low-rank factors of every eligible tensor. Layer l occupies a fixed segment of the shared output head (positional identity); features distinguish layers and trajectories. There is no per-layer network.

### Inputs (exactly what the operator sees)

| Feature group | Contents (steps 1..K only) |
|---|---|
| loss history | per-step train loss / loss at step 0 |
| gradient norms | global grad L2 history (1..K) |
| parameter norms | global param ratio, learning rate |
| per-layer gradient stats | mean/std/first/last grad norm, global fraction |
| per-layer parameter stats | param norms at 0 and K, ratio, grad-param & consecutive-grad cosines |
| layer identity | learned embedding (architecture, not trajectory-specific) |

No raw gradient vectors are fed to the operator, so any direction structure in U_l / V_l must come from learned operator weights — i.e. it is genuinely *generated* rather than a copy of an observed gradient.

## 3. Mathematical formulation

```
ΔW_l  = U_l V_l^T            (2-D weight l, U_l∈R^{out×r}, V_l∈R^{in×r})
Δw_l  = u_l                  (1-D tensors: generated vector)
W_new = W_K + ΔW

(U_l, V_l) = operator( features(K, l), embedding(l) )   # shared net

meta-training objective:
  loss_update = Σ_l ‖U_l V_l^T − ΔW_target_l‖² / ‖ΔW_target_l‖²
  (+ optional behavioural term λ·CE( W_K + ΔW; train corpus ))
```

The reconstruction objective is invariant to the factorization ambiguity UV^T = (UR)(R⁻¹V)^T. ΔW_target = W_H − W_K is used ONLY for meta-training; direct application feeds the operator nothing beyond step K.

## 4. Oracle results (can the low-rank family represent the update?)

Best rank-r approximation of ΔW_target per layer via SVD (1-D tensors exact). Uses the future answer — an upper bound for the whole low-rank family.

### K=10, H=25

| rank | explained energy (2-D) | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| 1 | 0.120 | 1.9332 | 6.91 | 0.0427 |
| 2 | 0.214 | 1.8771 | 6.54 | 0.0403 |
| 4 | 0.364 | 1.7736 | 5.89 | 0.0363 |
| 8 | 0.570 | 1.6418 | 5.16 | 0.0298 |

Conventional AdamW W_25 val loss: 1.5181 · no-update W_10: 2.0533

### K=10, H=50

| rank | explained energy (2-D) | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| 1 | 0.118 | 2.0062 | 7.44 | 0.0721 |
| 2 | 0.209 | 1.9378 | 6.96 | 0.0683 |
| 4 | 0.351 | 1.7583 | 5.81 | 0.0618 |
| 8 | 0.537 | 1.5018 | 4.49 | 0.0522 |

Conventional AdamW W_50 val loss: 1.1710 · no-update W_10: 2.0533

### K=15, H=25

| rank | explained energy (2-D) | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| 1 | 0.123 | 1.7177 | 5.57 | 0.0274 |
| 2 | 0.216 | 1.6892 | 5.42 | 0.0259 |
| 4 | 0.363 | 1.6378 | 5.14 | 0.0234 |
| 8 | 0.564 | 1.5816 | 4.86 | 0.0193 |

Conventional AdamW W_25 val loss: 1.5181 · no-update W_15: 1.7719

### K=25, H=25

| rank | explained energy (2-D) | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| 1 | 0.000 | 1.5181 | 4.56 | 0.0000 |
| 2 | 0.000 | 1.5181 | 4.56 | 0.0000 |
| 4 | 0.000 | 1.5181 | 4.56 | 0.0000 |
| 8 | 0.000 | 1.5181 | 4.56 | 0.0000 |

Conventional AdamW W_25 val loss: 1.5181 · no-update W_25: 1.5181

### K=25, H=50

| rank | explained energy (2-D) | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| 1 | 0.138 | 1.4837 | 4.41 | 0.0403 |
| 2 | 0.239 | 1.4532 | 4.28 | 0.0378 |
| 4 | 0.386 | 1.3860 | 4.00 | 0.0340 |
| 8 | 0.563 | 1.3075 | 3.70 | 0.0287 |

Conventional AdamW W_50 val loss: 1.1710 · no-update W_25: 1.5181

**Reading:** energy captured stays low at small ranks, and even the oracle cannot reach conventional quality — evidence about whether the low-rank family itself is sufficient.

## 5. Learned operator results

Direct application on held-out test trajectories (seeds [30, 31, 32, 33] + reference 7), mean over seeds. The operator receives only steps-1..K information.

### K=10, H=25

| method | train loss | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| no update (W_K) | - | 2.0533 | 7.80 | - |
| AdamW W_H (conventional) | - | 1.5181 | 4.56 | 0.0 |
| Phase-3 predictor (scaling) | - | 2.0834 | 8.05 | - |
| learned low-rank r=1 | - | 2.0490 | 7.76 | 0.0457 |
| learned low-rank r=2 | - | 2.0490 | 7.76 | 0.0457 |
| learned low-rank r=4 | - | 2.0490 | 7.76 | 0.0457 |

- learned r=1: relative ΔW prediction error 1.0039, update norm 0.200.
- learned r=2: relative ΔW prediction error 1.0038, update norm 0.199.
- learned r=4: relative ΔW prediction error 1.0038, update norm 0.199.

### K=10, H=50

| method | train loss | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| no update (W_K) | - | 2.0533 | 7.80 | - |
| AdamW W_H (conventional) | - | 1.1710 | 3.23 | 0.0 |
| Phase-3 predictor (scaling) | - | 2.2888 | 10.27 | - |
| learned low-rank r=1 | - | 2.0434 | 7.72 | 0.0770 |
| learned low-rank r=2 | - | 2.0434 | 7.72 | 0.0770 |
| learned low-rank r=4 | - | 2.0434 | 7.72 | 0.0770 |

- learned r=1: relative ΔW prediction error 1.0011, update norm 0.317.
- learned r=2: relative ΔW prediction error 1.0011, update norm 0.318.
- learned r=4: relative ΔW prediction error 1.0011, update norm 0.318.

### K=15, H=25

| method | train loss | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| no update (W_K) | - | 1.7719 | 5.88 | - |
| AdamW W_H (conventional) | - | 1.5181 | 4.56 | 0.0 |
| learned low-rank r=1 | - | 1.7695 | 5.87 | 0.0293 |
| learned low-rank r=2 | - | 1.7695 | 5.87 | 0.0293 |
| learned low-rank r=4 | - | 1.7695 | 5.87 | 0.0293 |

- learned r=1: relative ΔW prediction error 1.0000, update norm 0.073.
- learned r=2: relative ΔW prediction error 1.0000, update norm 0.073.
- learned r=4: relative ΔW prediction error 1.0000, update norm 0.073.

### K=25, H=25

| method | train loss | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| no update (W_K) | - | 1.5181 | 4.56 | - |
| AdamW W_H (conventional) | - | 1.5181 | 4.56 | 0.0 |
| learned low-rank r=1 | - | 1.5181 | 4.56 | 0.0000 |
| learned low-rank r=2 | - | 1.5181 | 4.56 | 0.0000 |
| learned low-rank r=4 | - | 1.5181 | 4.56 | 0.0000 |

- learned r=1: relative ΔW prediction error 0.0000, update norm 0.000.
- learned r=2: relative ΔW prediction error 0.0000, update norm 0.000.
- learned r=4: relative ΔW prediction error 0.0000, update norm 0.000.

### K=25, H=50

| method | train loss | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| no update (W_K) | - | 1.5181 | 4.56 | - |
| AdamW W_H (conventional) | - | 1.1710 | 3.23 | 0.0 |
| learned low-rank r=1 | - | 1.5147 | 4.55 | 0.0435 |
| learned low-rank r=2 | - | 1.5147 | 4.55 | 0.0435 |
| learned low-rank r=4 | - | 1.5147 | 4.55 | 0.0435 |

- learned r=1: relative ΔW prediction error 0.9994, update norm 0.123.
- learned r=2: relative ΔW prediction error 0.9994, update norm 0.123.
- learned r=4: relative ΔW prediction error 0.9994, update norm 0.123.


## 6. Compute

### 6.1 Per application (primary pair K=10, H=25, r=4)

| component | fwd/bwd steps | FLOPs |
|---|---|---|
| observation (AdamW to W_K) | 10 | 2.699e+11 |
| operator inference | 1 | 2.665e+08 |
| U V^T generation | 1 | 1.285e+07 |
| parameter update | 1 | 3.222e+06 |
| direct total | 10 + 1 | 2.702e+11 |
| conventional total (100 steps) | 100 | 6.749e+11 |

### 6.2 Meta-training cost (one-time)

| component | FLOPs |
|---|---|
| trajectory generation (6 runs × 100 steps) | 1.620e+13 |
| update-operator training | 0.000e+00 |
| total meta-training | 1.620e+13 |

### 6.3 Amortized total cost (N applications)

| N applications | Direct (meta + N·direct) | Conventional (N·conventional) |
|---|---|---|
| 1 | 1.647e+13 | 6.749e+11 |
| 10 | 1.890e+13 | 6.749e+12 |
| 100 | 4.322e+13 | 6.749e+13 |
| 1000 | 2.864e+14 | 6.749e+14 |

Meta-training costs are not hidden. Per-application direct FLOPs are observation (K steps) + operator inference + U V^T generation + one parameter update.

## 7. Ablations

### Feature sets at K=10, H=25, r=4

| feature set | val loss (learned) |
|---|---|
| loss | 2.0499 |
| grad | 2.0488 |
| grad_loss | 2.0491 |
| full | 2.0490 |

Behavioural-objective pilot (λ=0.1, train-corpus CE): val loss 2.0490.

Rank and observation-horizon effects are visible in Sections 4 and 5 (learned ranks 1/2/4; K=10/15/25). Update-structure comparison: Phase-3 per-layer *scaling* vs Phase-4 low-rank *generation* is in Section 5 where the Phase-3 predictor is available (K=10 pairs).

## 8. Generalization

Held-out seeds: [30, 31, 32, 33] + reference 7 (never used for meta-training). Results in Section 5 are means over these seeds; per-seed numbers are saved under `metrics/` and `oracle/`.

**Known limitation (documented):** all trajectories share the same small deterministic synthetic corpus; trajectories differ by initialization and batch sampling only. Generalization to a *different data distribution* was not tested here (would require a second corpus; deferred to later work given the CPU/scope constraints).

## 9. Failure analysis

**Headline: the experiment failed, and the bottleneck is the LEARNED OPERATOR'S LACK OF GENERALISATION (layer 5), with a secondary ceiling from the low-rank family at small rank (layer 1).**

1. **Family (layer 1): partially capable but insufficient at r ≤ 8.** The oracle's best rank-8 update captures only ~57% of the 2-D update energy yet improves val loss from 2.053 (no update) to 1.642, vs conventional 1.518. Energy grows monotonically with rank, so the family is *informative* — but it does not reach AdamW quality at r ≤ 8 (Phase-2 measured an effective rank of ~180-230 for the full 100-step update).

2. **Prediction (layers 2/4): the operator memorises meta-training and generalises to nothing.** On the 6 meta-train trajectories the reconstruction MSE falls to 0.44 (step-799), i.e. it learns the training deltas. But on held-out test trajectories the relative ΔW prediction error is 1.004 (~1.0 = predicting ~zero) and the update norm collapses to 0.20 — the operator emits a near-zero update on unseen seeds, so its val loss ≈ no-update (2.0490 vs 2.053). This is classic overfitting on ~150 (trajectory × layer) examples.

3. **Observation horizon (layer 3): secondary.** K=10 → K=15 improves the oracle ceiling (e.g. H=25: r8 1.64 → 1.58), but the learned operator is insensitive to K because it is stuck at ~zero update.

4. **Compute and behaviour (layers 6/7): not the bottleneck.** Direct application FLOPs are dominated by the K observation steps and are far below conventional (Section 6); the behavioural objective pilot also yields ≈ no-update, and the tiny updates produced do not damage validation (they sit just below no-update val loss).

| hypothesis layer | verdict | evidence |
|---|---|---|
| 1. low-rank family representable? | **partially (not at r≤8)** | oracle r8 energy ~54-57%, val 1.31-1.64 |
| 2. operator predicts a representable update? | **no (test)** | train recon MSE 0.44 vs test rel-err ~1.0 |
| 3. observation horizon K sufficient? | partially | oracle improves with K; learned K-insensitive |
| 4. meta-training data sufficient? | **no** | 150 examples; heavy overfit |
| 5. generalisation to unseen seeds? | **fails** | held-out rel-err ≈ 1.0 |
| 6. direct compute too expensive? | no | dominated by K observation steps |
| 7. low train loss but damaged validation? | no | updates ≈ 0, no damage |

The failure is therefore NOT a computational one and NOT a representation impossibility at large rank; it is that a tiny shared network cannot be meta-trained from 6 trajectories to emit the specific factor values that would reconstruct a held-out trajectory's update.

## 10. Recommendation

**MODIFY.**

Justification from the measurements:

- The oracle (Section 4) shows the generated low-rank family is *informative*: val loss falls monotonically with rank (e.g. K=10 H=25: 2.05 → 1.64 at r=8), and for short horizons the gap to conventional (1.52) narrows to ≈0.1. The hypothesis is not falsified at the representation level.
- The learned operator fails at *learning*, not at the family: it overfits meta-train (recon MSE 0.44) and predicts ≈0 on held-out seeds (rel-err 1.0). A zero update cannot win, but neither does it damage — this is recoverable with a better meta-training setup.

Next experiment (Phase 5, if pursued):

1. **More meta-training data** — many more seeds/trajectories (dozens, not six) and/or a second synthetic corpus, so the operator cannot memorise individual runs;
2. **More structured generation** — instead of a flat positional output head, predict per-layer low-rank factors through a shared learned per-layer basis (compressed-basis generation) or via a hypernetwork conditioned on layer statistics, which is far more sample-efficient than emitting ~10-40k numbers from ~40 inputs;
3. **Meta-objective on held-out quality** — optimise the reconstruction loss with a regulariser that favours generalising directions, and evaluate on meta-validation during training (early stopping on held-out rel-err, not train MSE);
4. **Accept larger effective rank** — if the goal is full baseline quality, the structured update must span a much larger subspace (Phase-2 effective ranks ~180-230), e.g. block-wise or hierarchical updates, not rank-1..8.

Success criterion remains unchanged: reach W_H-quality at strictly less training compute, with meta-training amortized and reported.
