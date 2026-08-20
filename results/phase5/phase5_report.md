# Phase 5 — Generalizable Update Operator

Date: 2026-08-20T02:49:24.123127+00:00

> Research status: R&D hypothesis under investigation. Nothing here is claimed to replace gradient descent or to scale to large models.

## 1. Research question

Can a learned update operator **generalize** its parameter-transformation strategy to unseen training trajectories (unseen seed, unseen batch ordering, unseen initialization, unseen corpus)? The goal is to separate representation capacity from generalization capacity — not to beat AdamW prematurely.

## 2. Experimental design

Short horizons first: K ∈ {10, 15, 25}, H ∈ {25, 50}, ranks 4 and 8 (primary 4). Each experiment reports no-update, conventional AdamW, rank-r oracle, and the learned generator on held-out trajectories. Meta-training uses trajectory data through step K only; the future target ΔW = W_H − W_K is used solely for supervision. The behavioural objective uses the TRAIN corpus (no eval-corpus leakage).

## 3. Meta-training data

- Corpus A: existing deterministic synthetic corpus (shared, ~60k train / 20k val tokens, 32-char vocabulary).
- Meta-train: 32 trajectories (seeds 10–41); data-size ablation uses the first 6 / 16 / 32 of these.
- Meta-validation: 8 trajectories (seeds 42–49) — used for validation-based checkpoint selection.
- Held-out tests: unseen-seed test A (seeds 50–57 + reference 7); unseen batch-order test B (seen init, unseen data seed 60–61); unseen-init test C (unseen init 62–63, seen data seed); unseen-corpus test D (corpus B).

All trajectories differ by initialization and batch sampling seed. Corpus B uses the same character set/tokenizer (identical architecture) but a sharply different word distribution, providing a genuine distribution shift.

## 4. Architecture

Compressed / latent generation (shared across layers):

```
global+per-layer features + layer embedding
   -> shared feature encoder -> latent z_l
   -> small factor generator -> coefficients
   -> apply onto learned per-layer bases
ΔW_l = U_l V_l^T,  U_l = P_l C_l,  V_l = Q_l D_l   (2-D)
Δw_l = B_l c_l                                     (1-D)
P_l ∈ R^{out×m}, Q_l ∈ R^{in×m}, B_l ∈ R^{dim×m}  (learned per-layer bases)
```

The network predicts only the small coefficients C_l, D_l (m×r) / c_l (m) (m = 32 basis width, r = 4 rank). The learned bases are directions shared across trajectories (not observed gradients). The output head is zero-initialized (starts at 'no update'). Compared against the Phase-4 flat MLP ('direct') in the architecture ablation.

## 5. Training objectives

| objective | formula |
|---|---|
| A recon | Σ_l ‖U_l V_l^T − ΔW_target_l‖² / ‖ΔW_target_l‖² |
| B behavior | CE(W_K + ΔW_pred; train corpus) |
| C combined | A + λ·B (λ tested: 0.01, 0.1, 1.0) |

Regularization: Adam weight decay, validation-based checkpoint selection (keep the generator with lowest meta-val reconstruction MSE).

## 6. Generalization results

### Data-size ablation (K=10, H=25, r=4, combined λ=0.1, compressed)

| # train traj | val (test) | no update | oracle r4 | % oracle recovered |
|---|---|---|---|---|
| # train traj | val (test) | no update | oracle r4 | % oracle recovered |
| 6 | 2.0149 | 2.0182 | 1.7588 | 1.3 |
| 16 | 2.0144 | 2.0182 | 1.7588 | 1.5 |
| 32 | 2.0144 | 2.0182 | 1.7588 | 1.4 |

### Objective ablation (16 train, r=4, compressed)

| objective | val (test) | % oracle recovered |
|---|---|---|
| objective | val (test) | % oracle recovered |
| recon_0.0 | 2.0141 | 1.6 |
| combined_0.01 | 2.0140 | 1.6 |
| combined_0.1 | 2.0144 | 1.5 |
| combined_1.0 | 2.0125 | 2.2 |
| behavior_1.0 | 2.0184 | -0.1 |

### Architecture ablation (16 train, combined λ=0.1, r=4)

| arch | val (test) |
|---|---|
| arch | val (test) |
| compressed | 2.0144 |
| direct | 2.0145 |

### Rank ablation (16 train, combined λ=0.1)

| rank | val (test) |
|---|---|
| rank | val (test) |
| 4 | 2.0144 |
| 8 | 2.0128 |

### Horizon (16 train, combined λ=0.1, r=4)

| (K,H) | val (test) |
|---|---|
| (K,H) | val (test) |
| 10_25 | 2.0144 |
| 15_25 | 1.7469 |
| 25_50 | 1.5003 |

## 7. Oracle comparison

The rank-r oracle (best SVD of the true update) sets the family ceiling.

| pair | rank | val (test) | explained energy (2-D) |
|---|---|---|---|
| pair | rank | val (test) | explained energy (2-D) |
| (10,25) | 1 | 1.9307 | 0.119 |
| (10,25) | 2 | 1.8721 | 0.211 |
| (10,25) | 4 | 1.7607 | 0.358 |
| (10,25) | 8 | 1.6271 | 0.564 |
| (15,25) | 1 | 1.7123 | 0.123 |
| (15,25) | 2 | 1.6813 | 0.216 |
| (15,25) | 4 | 1.6270 | 0.360 |
| (15,25) | 8 | 1.5697 | 0.558 |
| (25,50) | 1 | 1.4743 | 0.135 |
| (25,50) | 2 | 1.4409 | 0.239 |
| (25,50) | 4 | 1.3846 | 0.390 |
| (25,50) | 8 | 1.2970 | 0.566 |

## 8. Ablations

Summarized in Section 6 (data size, objective, architecture, rank, horizon). The primary read: does adding meta-training data, changing the objective, or using a structured generator move the held-out result away from no-update?

## 9. Compute

### Per application (K=10, H=25, r=4)

| component | steps | FLOPs |
|---|---|---|
| observation (AdamW to W_K) | 10 | 2.699e+11 |
| generator inference | 1 | 2.765e+06 |
| U V^T generation | 1 | 1.285e+07 |
| parameter update | 1 | 3.222e+06 |
| direct total | - | 2.700e+11 |
| conventional total | 100 | 6.749e+11 |

### Meta-training (one-time) and amortization

| N applications | Direct (meta + N·direct) | Conventional (N·conventional) |
|---|---|---|
| N applications | Direct (meta + N·direct) | Conventional (N·conventional) |
| 1 | 4.346e+13 | 6.749e+11 |
| 10 | 4.589e+13 | 6.749e+12 |
| 100 | 7.019e+13 | 6.749e+13 |
| 1000 | 3.132e+14 | 6.749e+14 |

Meta-training trajectory generation FLOPs (32 runs × 50 steps): 4.319e+13. The direct application cost is dominated by the K observation steps.

## 10. Failure analysis

Determine which limiting factor applies (representation / generator / training data / objective / horizon / generalization / compute):

| factor | how tested |
|---|---|
| representation | oracle energy vs rank (Sec. 7) |
| generator | arch ablation: compressed vs flat (Sec. 6) |
| training data | data-size ablation 6/16/32 (Sec. 6) |
| objective | A/B/C comparison (Sec. 6) |
| horizon | K/H grid (Sec. 6) |
| generalization | tests A–D (below) |
| compute | Section 9 |

### Generalization tests (primary generator: 32 train, combined λ=0.1, r=4, K=10 H=25)

| test | no update | oracle r4 | learned | % oracle recovered | cos(ΔW_pred, ΔW_target) | cos(mean-grad, ΔW_target) |
|---|---|---|---|---|---|---|
| A_unseen_seed | 2.0182 | 1.7588 | 2.0144 | 1.4 | 0.0311 | -0.4050 |
| A_ref_seed | 2.0851 | 1.7764 | 2.0818 | 1.1 | 0.0296 | -0.3845 |
| B_unseen_batch_order | 2.0215 | 1.7670 | 2.0110 | 4.1 | 0.0447 | -0.4038 |
| C_unseen_init | 2.0924 | 1.7874 | 2.0868 | 1.8 | 0.0304 | -0.3921 |
| D_unseen_corpus | 1.8517 | 1.6016 | 1.8497 | 0.8 | 0.0180 | -0.3570 |

## 11. Recommendation

Success level 1 (research): learned > no-update on held-out trajectories with a substantial fraction of oracle improvement recovered. Success level 2 (methodological): quality ≈ AdamW at lower total compute.

**STOP THIS APPROACH — increasing meta-training data (6→32), a **

Measured basis: unseen-seed test — no update 2.0182, learned 2.0144, % oracle improvement recovered 1.4. Full numbers in Section 10.

---
_Phase-5 report generated by src/phase5/run.py. Raw data under results/phase5/metrics|ablations|generalization|oracle|predictions._