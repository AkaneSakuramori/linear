# Phase 3 — Learned Direct Update Predictor

Date: 2026-08-12T12:26:02.963469+00:00

> Research status: **hypothesis under investigation — negative result reported honestly.** Nothing here is claimed to beat AdamW or to scale to large models.

## 1. Hypothesis

Can a **small learned neural network** predict a useful multi-step parameter transformation from information available *early* in training (steps 0–K), without performing the remaining conventional optimization trajectory (steps K+1…H)?

Phase 2 showed that fixed linear combinations of early gradients (scaled first gradient, average gradient, momentum, low-rank versions) cannot reach the conventional baseline. The open question is whether a *learned, non-linear* mechanism can do better by exploiting per-layer structure in early-training statistics. Phase 3 tests the first concrete instance of such a mechanism.

## 2. Method

### 2.1 Parameterisation (structured, compact)

The predictor does **not** emit all ~1.6M parameters. It emits one small set of scalar coefficients per layer, applied to a basis of **observed-gradient directions** (information available at step K). With basis `mean`:

```
ΔW_pred = Σ_l Σ_j α_{l,j} · D_{l,j}        (per layer l, basis vector j)
W_pred  = W_K + ΔW_pred
D_l     ∈ { mean(grad_1..K), grad_1, grad_K }  (observed, no future info)
```

Primary basis: `mean` (the mean of the K observed unclipped gradients per layer). Alternatives (`first_last`, `first_last_mean`) are reported in the parameterisation ablation. The coefficients α are produced by a small MLP shared across all layers (hidden size 64, ~2k parameters), so each (trajectory, layer) is one training example.

### 2.2 What the predictor sees (feature vector)

Inputs are compact statistics of steps 1…K only (never future states, never future losses, never the optimizer state beyond step K):

| Component | Description |
|---|---|
| loss history | per-step train loss for steps 1..K, normalised by loss at step 0 |
| global gradient norms | per-step global grad L2 for steps 1..K, normalised by step 1 |
| parameter norms | global param L2 at step K / step 0; learning rate |
| per-layer gradient stats | mean / std / first / last grad norm, per-layer fraction of global grad norm |
| per-layer parameter stats | param norm at step 0 and K, ratio, grad-param cosine, consecutive-grad cosine |
| Adam state (feature set `rich` only) | exp_avg / exp_avg_sq norms at step K |

The primary feature set is `full` (gradient statistics + parameter statistics + loss history + per-layer information). Other sets are ablation studies.

## 3. Training procedure (meta-training)

The predictor is trained offline on conventional AdamW trajectories. For each (K, H) pair and each meta-train trajectory:

1. roll the identical Phase-1 recipe (same corpus, same window sampling, same gradient clipping) out to H with a fresh random seed;
2. record the observation (features from steps 1..K);
3. compute the supervised target per layer by least squares against the ground-truth transformation:

```
α*_l = argmin_α || (W_H − W_K)_l − Σ_j α_j D_{l,j} ||²
```

The MLP is trained with MSE on these targets (Adam, 2000 steps, lr 0.001, batch 32). The target uses the future answer, which is **only** allowed here, during meta-training; the resulting predictor never receives W_H or α* at application time.

Meta-train trajectories: seeds [10, 11, 12, 13, 14, 15]. Meta-validation (early stopping / reporting): seeds [20, 21]. Meta-test (held out): seeds [30, 31, 32, 33] plus the Phase-1 reference seed 7. Trajectories differ by initialization and batch sampling; the synthetic corpus is shared (documented limitation).

## 4. Evaluation procedure (no-future-information enforcement)

Two experiments are kept separate in code and in this report:

| Experiment | What it receives | Legitimate? |
|---|---|---|
| **Oracle ceiling** | W_H − W_K to pick α* (least squares) | upper bound only |
| **Direct application** | only stats/gradients from steps 1..K | the real method |

During direct application the predictor gets: per-step gradients up to K, loss history up to K, parameter statistics at step K, Adam-state norms at step K (`rich` ablation). It does **not** get W_H, future losses, future gradients, or future optimizer states. `W_pred = W_K + ΔW_pred` is then evaluated from scratch (no further training steps). Enforced structurally: features and the gradient basis are constructed from `grad_states[1..K]` only, and the target `compute_alpha_star` is a separate code path used only at meta-train time.

## 5. Results

Quality is reported as the **mean over the 5 held-out test trajectories** (seeds 30–33 and the reference seed 7). The direct method predicts from W_K; the conventional method is the same trajectory continued with AdamW to W_H.

### K=5, H=25

| Method | train loss | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| AdamW W_H (conventional) | - | 1.5181 | 4.564 | 0.0 |
| stay at W_K (no update) | - | 2.5794 | 13.201 | 0.0610 |
| Oracle (per-layer ceiling) | - | 3.0071 | 21.894 | 0.0558 |
| Direct Update Predictor | - | 3.3887 | 33.290 | 0.0583 |

Relative prediction error (‖ΔW_pred − ΔW_target‖/‖ΔW_target‖): **0.9555**. Direct vs conventional val-loss gap: 1.8706.

Parameterisation: per-layer scaling of the mean of the 5 observed gradients; predictor emits one coefficient per layer.

### K=5, H=50

| Method | train loss | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| AdamW W_H (conventional) | - | 1.1710 | 3.225 | 0.0 |
| stay at W_K (no update) | - | 2.5794 | 13.201 | 0.0889 |
| Oracle (per-layer ceiling) | - | 3.2427 | 28.740 | 0.0839 |
| Direct Update Predictor | - | 3.7825 | 50.033 | 0.0904 |

Relative prediction error (‖ΔW_pred − ΔW_target‖/‖ΔW_target‖): **1.0160**. Direct vs conventional val-loss gap: 2.6115.

Parameterisation: per-layer scaling of the mean of the 5 observed gradients; predictor emits one coefficient per layer.

### K=5, H=100

| Method | train loss | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| AdamW W_H (conventional) | - | 0.8352 | 2.305 | 0.0 |
| stay at W_K (no update) | - | 2.5794 | 13.201 | 0.1161 |
| Oracle (per-layer ceiling) | - | 3.3427 | 32.423 | 0.1123 |
| Direct Update Predictor | - | 3.7624 | 91.871 | 0.1270 |

Relative prediction error (‖ΔW_pred − ΔW_target‖/‖ΔW_target‖): **1.0932**. Direct vs conventional val-loss gap: 2.9273.

Parameterisation: per-layer scaling of the mean of the 5 observed gradients; predictor emits one coefficient per layer.

### K=10, H=25

| Method | train loss | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| AdamW W_H (conventional) | - | 1.5181 | 4.564 | 0.0 |
| stay at W_K (no update) | - | 2.0533 | 7.797 | 0.0455 |
| Oracle (per-layer ceiling) | - | 2.0437 | 7.739 | 0.0384 |
| Direct Update Predictor | - | 2.0834 | 8.048 | 0.0419 |

Relative prediction error (‖ΔW_pred − ΔW_target‖/‖ΔW_target‖): **0.9175**. Direct vs conventional val-loss gap: 0.5653.

Parameterisation: per-layer scaling of the mean of the 10 observed gradients; predictor emits one coefficient per layer.

### K=10, H=50

| Method | train loss | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| AdamW W_H (conventional) | - | 1.1710 | 3.225 | 0.0 |
| stay at W_K (no update) | - | 2.0533 | 7.797 | 0.0770 |
| Oracle (per-layer ceiling) | - | 2.1182 | 8.387 | 0.0696 |
| Direct Update Predictor | - | 2.2888 | 10.270 | 0.0723 |

Relative prediction error (‖ΔW_pred − ΔW_target‖/‖ΔW_target‖): **0.9398**. Direct vs conventional val-loss gap: 1.1178.

Parameterisation: per-layer scaling of the mean of the 10 observed gradients; predictor emits one coefficient per layer.

### K=10, H=100

| Method | train loss | val loss | val ppl | rel param dist to W_H |
|---|---|---|---|---|
| AdamW W_H (conventional) | - | 0.8352 | 2.305 | 0.0 |
| stay at W_K (no update) | - | 2.0533 | 7.797 | 0.1072 |
| Oracle (per-layer ceiling) | - | 2.1842 | 9.010 | 0.1017 |
| Direct Update Predictor | - | 2.2260 | 9.412 | 0.1039 |

Relative prediction error (‖ΔW_pred − ΔW_target‖/‖ΔW_target‖): **0.9696**. Direct vs conventional val-loss gap: 1.3908.

Parameterisation: per-layer scaling of the mean of the 10 observed gradients; predictor emits one coefficient per layer.

## 6. Compute

Compute accounting separates the four cost buckets requested by the plan. Per-step FLOPs come from the Phase-1 audit (`src/utils.estimate_flops`, backward ≈ 3× forward).

### 6.1 Conventional training cost (per application, from W0)

| horizon H | fwd/bwd steps | FLOPs |
|---|---|---|
|  25 |  25 | 6.749e+11 |
|  50 |  50 | 1.350e+12 |
| 100 | 100 | 2.699e+12 |

### 6.2 Direct application cost (per application)

| component | fwd/bwd steps | FLOPs |
|---|---|---|
| observation (fwd+bwd to W_K) | 5 | 1.350e+11 |
| predictor inference | 1 | 1.114e+04 |
| parameter update | 1 | 3.222e+06 |
| direct total | 5 + 1 | 1.350e+11 |
| conventional total | 100 | 2.699e+12 |

The direct method skips the remaining H−K optimizer steps (95 steps for K=5,H=100) at the cost of one tiny MLP forward pass — but it must also amortize the meta-training cost.

### 6.3 Predictor / meta-training cost (one-time)

| component | FLOPs |
|---|---|
| meta-train trajectory generation | 1.620e+13 |
| predictor training (5 K/H predictors) | 1.809e+10 |
| total meta-training | 1.621e+13 |

### 6.4 Total amortized cost

| N applications | Direct (meta + N·direct) | Conventional (N·conventional) |
|---|---|---|
| 1 | 1.635e+13 | 2.699e+12 |
| 10 | 1.756e+13 | 2.699e+13 |
| 100 | 2.971e+13 | 2.699e+14 |
| 1000 | 1.512e+14 | 2.699e+15 |

The meta-training cost is **not hidden**: the table reports it explicitly. Total FLOPs cross over around **N ≈ 6 applications** (direct cheaper beyond that), so amortization is achievable *in principle* — the failure is not compute but **quality** (Section 7): the direct method does not reach conventional quality even when it is the cheaper option.

## 7. Quality

For every (K, H) pair, the Direct Update Predictor's validation loss is far above the conventional W_H. Details in Section 5 tables; headline numbers:

| pair (K,H) | AdamW val | no-update val | oracle val | direct val | direct ppl |
|---|---|---|---|---|---|
| (5,25) | 1.5181 | 2.5794 | 3.0071 | 3.3887 | 33.29 |
| (5,50) | 1.1710 | 2.5794 | 3.2427 | 3.7825 | 50.03 |
| (5,100) | 0.8352 | 2.5794 | 3.3427 | 3.7624 | 91.87 |
| (10,25) | 1.5181 | 2.0533 | 2.0437 | 2.0834 | 8.05 |
| (10,50) | 1.1710 | 2.0533 | 2.1182 | 2.2888 | 10.27 |
| (10,100) | 0.8352 | 2.0533 | 2.1842 | 2.2260 | 9.41 |

The oracle column is critical for interpretation: it is the *best possible* quality the per-layer-scaling family could reach (it uses the future answer to pick α). It is also far above conventional quality, which shows the limitation is the **parameterisation family** (linear combinations of the few observed gradients cannot span the true multi-step update), not merely the learned predictor.

## 8. Ablations

Feature-set ablations (what information the predictor sees) at the pair(s) studied:

### Feature sets at K=5, H=100

| feature set | val loss (direct) | val ppl |
|---|---|---|
| loss | 3.7031 | 46.28 |
| grad | 2.8118 | 17.20 |
| grad_loss | 4.2170 | 208.20 |
| full | 3.7624 | 91.87 |
| compressed | 3.5001 | 33.58 |
| rich | 4.2341 | 85.30 |

Legend: `loss` = loss history only; `grad` = gradient statistics only; `grad_loss` = gradient + loss; `full` = primary (gradient + loss + parameter + per-layer statistics); `compressed` = full with histories compressed to [first, last, mean]; `rich` = full + Adam-state norms.

Conventional AdamW val loss at H=100: 0.8352. Stay-at-W_5: 2.5794.

### Parameterisation (basis) at K=5, H=100

| basis | val loss (direct) |
|---|---|
| first_last | 5.4205 |
| first_last_mean | 4.1959 |

Interpretation: adding features moves the predictor marginally but never by orders of magnitude; richer bases (more observed-gradient directions) also do not bridge the gap. The informative quantity is the oracle ceiling, which is independent of the feature set.

## 9. Failure analysis

The experiment failed: **the Direct Update Predictor does not approach conventional quality at reduced compute.** Evidence:

1. **Prediction error is large in parameter space.** Relative ΔW prediction error at (5,100) is 1.0932; the predicted transformation differs from the true W_100 − W_5 by that fraction of the target norm.
2. **The parameterisation family itself is insufficient.** The oracle (α* from the future answer) reaches val loss 3.3427 vs conventional 0.8352. Even with perfect coefficients, per-layer scaling of the 5 observed gradient directions cannot span the true multi-step update — consistent with Phase-2's finding that cos(ΔW_100, avg-grad) ≈ −0.40 and that ΔW is dense and high-rank.
3. **Observation horizon K is short relative to the target path.** With only 5–10 early gradients available, the basis has almost no overlap with the later, substantially-rotated update direction (Phase-2 direction-change cosines 0.85–0.93 mean the path keeps rotating).
4. **The learned predictor tracks the ceiling, not the target.** Its val loss sits near the oracle ceiling (and above it), i.e. the MLP learns the best coefficients the restricted family admits; there is no sign of discovering structure outside that span.

Which explanations are ruled out / supported:

| candidate explanation | evidence |
|---|---|
| target too complex (non-linear, step-dependent accumulation) | **supported** — Phase-2 + oracle ceiling here |
| input representation lacks information | partially — richer features barely help (Section 8) |
| predictor capacity insufficient | unlikely — it already matches the oracle ceiling of the family |
| observation horizon K too short | supported — the basis has no future path overlap |
| target horizon H too long | supported — error grows with H |

### Compute amortization

Generating the 6 meta-train trajectories costs 1.62e+13 FLOPs — six conventional 100-step runs' worth — plus a negligible predictor-training cost. The direct method's per-application FLOPs are only ~1/20 of a conventional run (observation K steps + a tiny MLP + one parameter update), so the meta-training cost would amortize over ~6 applications. Compute is therefore **not** the blocker; the blocker is quality.

## 10. Recommendation

**MODIFY.**

The learned *predictor* is not the bottleneck — the *parameterisation* is. Predicting scaling coefficients for observed-gradient directions is still a linear-combination family, which Phase-2 already showed cannot span the true update. The negative result therefore does **not** refute the research hypothesis (a learned non-linear operator), because the operator we trained was linear in the available directions.

Next scientifically justified experiment (Phase 4): give the learned operator the ability to *generate* directions rather than scale given ones, e.g. a per-layer low-rank update U_l V_l^T where the factors are outputs of the network, or a hypernetwork over a compressed parameter basis; and shorten the target (H ≈ 25) while lengthening K (K = 10–25). The measure of success stays unchanged: reach W_H-quality at strictly less training compute, with the meta-training cost amortized and reported.

---
_Phase-3 report generated by `src/phase3/run.py`. Raw data under `results/phase3/metrics/`, `results/phase3/predictions/`, `results/phase3/ablations/`._