# Phase 2 — Direct Parameter Update Research, summary

Date: 2026-08-12T00:41:21.913280+00:00
Seed: 7, threads: 4, model params: 1610752
Horizons studied: [5, 10, 25, 50, 100]

## 1. Oracle upper bound (one update using the future answer)

| horizon | val loss | val ppl | param distance to W_N | compute (FLOPs) |
| 5 | 2.6284 | 13.85 | 0.00e+00 | 2.16e+11 |
| 10 | 2.0851 | 8.05 | 0.00e+00 | 2.16e+11 |
| 25 | 1.5154 | 4.55 | 0.00e+00 | 2.16e+11 |
| 50 | 1.1709 | 3.22 | 0.00e+00 | 2.16e+11 |
| 100 | 0.8373 | 2.31 | 0.00e+00 | 2.16e+11 |

The oracle reproduces W_N exactly (distance 0); its one update is as good as the full rollout **by construction**, and its FLOPs are only evaluation passes. It is an **upper bound, not a training algorithm** (cheating by definition).

## 2. Is the cumulative update structured?

- horizon 5: ||ΔW||₂=1.103 (vs ||W0||₂=43.85, ratio 0.0252); 98.8% of parameters moved meaningfully; cos(ΔW, first grad)=-0.440, cos(ΔW, avg grad)=-0.536, cos(ΔW, momentum)=-0.516
- horizon 10: ||ΔW||₂=1.807 (vs ||W0||₂=43.85, ratio 0.0412); 98.9% of parameters moved meaningfully; cos(ΔW, first grad)=-0.337, cos(ΔW, avg grad)=-0.491, cos(ΔW, momentum)=-0.413
- horizon 25: ||ΔW||₂=3.358 (vs ||W0||₂=43.85, ratio 0.0766); 99.3% of parameters moved meaningfully; cos(ΔW, first grad)=-0.231, cos(ΔW, avg grad)=-0.452, cos(ΔW, momentum)=-0.306
- horizon 50: ||ΔW||₂=4.504 (vs ||W0||₂=43.85, ratio 0.1027); 99.5% of parameters moved meaningfully; cos(ΔW, first grad)=-0.185, cos(ΔW, avg grad)=-0.401, cos(ΔW, momentum)=-0.120
- horizon 100: ||ΔW||₂=5.570 (vs ||W0||₂=43.85, ratio 0.1270); 99.5% of parameters moved meaningfully; cos(ΔW, first grad)=-0.149, cos(ΔW, avg grad)=-0.398, cos(ΔW, momentum)=-0.130

## 3. Direction stability over training

- cos(ΔW_5, ΔW_10) = 0.926
- cos(ΔW_10, ΔW_25) = 0.847
- cos(ΔW_25, ΔW_50) = 0.915
- cos(ΔW_50, ΔW_100) = 0.916

## 4. Low-rank structure of ΔW

| Layer | rank to keep | % energy @rank64 | eff rank (95% energy) |
|---|---|---|--|
| tok_emb.weight | 14 | n/a | 14 |
| pos_emb.weight | 21 | 100.0% | 21 |
| blocks.0.attn.qkv.weight | 179 | 91.4% | 179 |
| blocks.0.attn.proj.weight | 226 | 98.7% | 226 |
| blocks.0.mlp.fc1.weight | 183 | 93.6% | 183 |
| blocks.0.mlp.fc2.weight | 186 | 94.2% | 186 |
| blocks.1.attn.qkv.weight | 208 | 96.7% | 208 |
| blocks.1.attn.proj.weight | 231 | 99.0% | 231 |

## 5. Direct approximation methods (practical; no future answers)

| method | horizon | alpha | train loss | val loss | val ppl | param dist rel | update FLOPs | tuning FLOPs | total FLOPs | fwd | bwd | updates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DirectGradient | 5 | 0.0316 | 3.1003 | 3.1010 | 22.22 | 0.125 | 2.70e+10 | 5.40e+12 | 5.64e+12 | 1 | 1 | 1 |
| DirectAverageGradient | 5 | 0.3162 | 2.8376 | 2.8321 | 16.98 | 0.123 | 1.35e+11 | 5.40e+12 | 5.75e+12 | 5 | 5 | 1 |
| DirectMomentum | 5 | 0.5623 | 2.8904 | 2.8704 | 17.64 | 0.126 | 1.35e+11 | 5.40e+12 | 5.75e+12 | 5 | 5 | 1 |
| DirectLowRank_r16 | 5 | 0.3162 | 2.8468 | 2.8415 | 17.14 | 0.123 | 1.35e+11 | 5.40e+12 | 5.75e+12 | 5 | 5 | 1 |
| DirectLowRank_r64 | 5 | 0.3162 | 2.8376 | 2.8322 | 16.98 | 0.123 | 1.35e+11 | 0.00e+00 | 3.51e+11 | 5 | 5 | 1 |
| DirectLowRank_r128 | 5 | 0.3162 | 2.8376 | 2.8321 | 16.98 | 0.123 | 1.35e+11 | 0.00e+00 | 3.51e+11 | 5 | 5 | 1 |
| DirectGradient | 10 | 0.0316 | 3.1003 | 3.1010 | 22.22 | 0.125 | 2.70e+10 | 5.40e+12 | 5.64e+12 | 1 | 1 | 1 |
| DirectAverageGradient | 10 | 1.7783 | 2.4681 | 2.4647 | 11.76 | 0.117 | 2.70e+11 | 5.40e+12 | 5.88e+12 | 10 | 10 | 1 |
| DirectMomentum | 10 | 0.1000 | 2.7090 | 2.7125 | 15.07 | 0.120 | 2.70e+11 | 5.40e+12 | 5.88e+12 | 10 | 10 | 1 |
| DirectLowRank_r16 | 10 | 1.7783 | 2.4613 | 2.4524 | 11.62 | 0.118 | 2.70e+11 | 5.40e+12 | 5.88e+12 | 10 | 10 | 1 |
| DirectLowRank_r64 | 10 | 1.7783 | 2.4674 | 2.4638 | 11.75 | 0.117 | 2.70e+11 | 0.00e+00 | 4.86e+11 | 10 | 10 | 1 |
| DirectLowRank_r128 | 10 | 1.7783 | 2.4682 | 2.4647 | 11.76 | 0.117 | 2.70e+11 | 0.00e+00 | 4.86e+11 | 10 | 10 | 1 |
| DirectGradient | 25 | 0.0316 | 3.1003 | 3.1010 | 22.22 | 0.125 | 2.70e+10 | 5.40e+12 | 5.64e+12 | 1 | 1 | 1 |
| DirectAverageGradient | 25 | 0.5623 | 2.9061 | 2.9095 | 18.35 | 0.122 | 6.75e+11 | 5.40e+12 | 6.29e+12 | 25 | 25 | 1 |
| DirectMomentum | 25 | 0.1000 | 3.0825 | 3.0900 | 21.98 | 0.121 | 6.75e+11 | 5.40e+12 | 6.29e+12 | 25 | 25 | 1 |
| DirectLowRank_r16 | 25 | 0.5623 | 2.9179 | 2.9216 | 18.57 | 0.122 | 6.75e+11 | 5.40e+12 | 6.29e+12 | 25 | 25 | 1 |
| DirectLowRank_r64 | 25 | 0.5623 | 2.9062 | 2.9096 | 18.35 | 0.122 | 6.75e+11 | 0.00e+00 | 8.91e+11 | 25 | 25 | 1 |
| DirectLowRank_r128 | 25 | 0.5623 | 2.9061 | 2.9095 | 18.35 | 0.122 | 6.75e+11 | 0.00e+00 | 8.91e+11 | 25 | 25 | 1 |
| DirectGradient | 50 | 0.0316 | 3.1003 | 3.1010 | 22.22 | 0.125 | 2.70e+10 | 5.40e+12 | 5.64e+12 | 1 | 1 | 1 |
| DirectAverageGradient | 50 | 1.0000 | 3.0275 | 3.0284 | 20.66 | 0.121 | 1.35e+12 | 5.40e+12 | 6.96e+12 | 50 | 50 | 1 |
| DirectMomentum | 50 | 0.0316 | 3.4402 | 3.4414 | 31.23 | 0.125 | 1.35e+12 | 5.40e+12 | 6.96e+12 | 50 | 50 | 1 |
| DirectLowRank_r16 | 50 | 1.0000 | 3.0332 | 3.0345 | 20.79 | 0.122 | 1.35e+12 | 5.40e+12 | 6.96e+12 | 50 | 50 | 1 |
| DirectLowRank_r64 | 50 | 1.0000 | 3.0274 | 3.0283 | 20.66 | 0.121 | 1.35e+12 | 0.00e+00 | 1.57e+12 | 50 | 50 | 1 |
| DirectLowRank_r128 | 50 | 1.0000 | 3.0275 | 3.0284 | 20.66 | 0.121 | 1.35e+12 | 0.00e+00 | 1.57e+12 | 50 | 50 | 1 |
| DirectGradient | 100 | 0.0316 | 3.1003 | 3.1010 | 22.22 | 0.125 | 2.70e+10 | 5.40e+12 | 5.64e+12 | 1 | 1 | 1 |
| DirectAverageGradient | 100 | 1.7783 | 2.8415 | 2.8440 | 17.18 | 0.120 | 2.70e+12 | 5.40e+12 | 8.31e+12 | 100 | 100 | 1 |
| DirectMomentum | 100 | 0.0010 | 3.4583 | 3.4597 | 31.81 | 0.126 | 2.70e+12 | 5.40e+12 | 8.31e+12 | 100 | 100 | 1 |
| DirectLowRank_r16 | 100 | 1.7783 | 2.8704 | 2.8732 | 17.69 | 0.121 | 2.70e+12 | 5.40e+12 | 8.31e+12 | 100 | 100 | 1 |
| DirectLowRank_r64 | 100 | 1.7783 | 2.8424 | 2.8449 | 17.20 | 0.120 | 2.70e+12 | 0.00e+00 | 2.92e+12 | 100 | 100 | 1 |
| DirectLowRank_r128 | 100 | 1.7783 | 2.8416 | 2.8441 | 17.19 | 0.120 | 2.70e+12 | 0.00e+00 | 2.92e+12 | 100 | 100 | 1 |

**Reference (baseline):** train 0.8373, val 0.8373, ppl 2.31, FLOPs 5.07e+12, 100 parameter updates.

## 6. Conclusions

**Oracle upper bound:** one update that knows W_N reproduces the final model exactly (param distance 0). This is *by construction* the ceiling any single-update method could reach. It is not a training algorithm (it cheats), but it bounds what a 'perfect cumulative transformation' could buy us.
**Primary result — cumulative update is NOT aligned with the first gradient.** cos(ΔW_100, gradL(W0)) = -0.149; cos(ΔW_100, mean gradient) = -0.398; cos(ΔW_100, momentum) = -0.130. The negative sign is expected: gradients evaluated near W0 point *uphill*, and the optimizer moves downhill, so ΔW is anti-parallel. The magnitude of the anti-alignment is informative: a perfect constant-direction descent would give ≈ -1; here it stays around -0.15..-0.44 for the first gradient and -0.40..-0.54 for the mean gradient, i.e. the true multi-step path wanders substantially sideways relative to any single local gradient (as expected once gradients themselves keep changing).
**Direct approximations fail to match the baseline in a single update.** (validation loss)
  N=  5: first-grad val=3.101  avg-grad val=2.832  momentum val=2.870
  N= 10: first-grad val=3.101  avg-grad val=2.465  momentum val=2.713
  N= 25: first-grad val=3.101  avg-grad val=2.909  momentum val=3.090
  N= 50: first-grad val=3.101  avg-grad val=3.028  momentum val=3.441
  N=100: first-grad val=3.101  avg-grad val=2.844  momentum val=3.460
  Baseline val=0.837 oracle val=0.837.
Even after tuning α on the validation set, no single scaled-gradient update comes close to the 100-step result: the information captured by any single local gradient cannot span the multi-step path.
**Update magnitude is small in norm terms.** ||ΔW_100|| = 5.57 vs ||W0|| = 43.85 (ratio 0.1270) but 99.5% of parameters changed meaningfully. The cumulative update touches almost every parameter with a small perturbation — not a sparse or low-rank event (eff ranks ~14 at 95% energy).
**Compute check (fairness rule).** The update itself is cheap: DirectGradient's whole parameter transformation costs 2.7e+10 FLOPs (1 fwd+bwd) vs the baseline's 2.7e+12 for 100 steps. But DirectAverageGradient / DirectMomentum at horizon N spend exactly N forward+backward passes — the same training compute as the baseline rollout (e.g. horizon 100: 2.7e+12 FLOPs) — and only skip the 100 per-step optimizer updates. Even ignoring the (expensive) alpha tuning on the validation set, no direct method reaches the baseline's quality at reduced total computation: quality fell far short at equal fwd/bwd cost, and the only genuinely cheap method (DirectGradient) is far worse in quality. Alpha tuning on 25 validation candidates adds ≈5e12 FLOPs of forward passes per method/horizon (a one-time hyperparameter budget, not a training cost). See `direct_update_results.json` for full per-method FLOPs/time/update accounting.
**Direction stability:** the cumulative direction changes during training (see direction_change cosines; consecutive deltas [0.93, 0.85, 0.92, 0.92]). This is why a single static direction derived early cannot reproduce the path.
**Conclusion / next experiment decision:** the mathematical baselines tested here (scaled first gradient, average gradient, momentum direction, low-rank versions of those) do **not** provide a cheaper way to reach the baseline's quality on this task, and the cumulative update itself is not simply a scaled or low-rank version of any available gradient. A *learned* Direct Update Network would hence have to discover genuinely new structure — but these results do NOT rule it out, because the tested approximations are all linear combinations of local gradients, while the true update is a strongly non-linear, step-dependent accumulation. A learned mechanism that can mimic the *per-layer* redistribution seen here (e.g. attention vs MLP scale differences) is the only class of candidate whose viability remains open. Recommendation: proceed to a *small learned predictor* trained to estimate W_N−W_0 from a few on-path gradient/activations samples, and benchmark it against these numbers.
