<div align="center">

# 🧠 Direct Learning

### A CPU-first research framework for testing whether neural-network training can be made dramatically more compute-efficient.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch CPU](https://img.shields.io/badge/pytorch-2.x%20CPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![No GPU required](https://img.shields.io/badge/GPU-none_required-6B46C1?style=for-the-badge)
![Tests](https://img.shields.io/badge/tests-81%20passing-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-yellowgreen?style=for-the-badge)
![Status](https://img.shields.io/badge/status-CONCLUDED%20%E2%80%93%20APPROACH%20ABANDONED-critical?style=for-the-badge)

**⚠️ PROJECT STOPPED — RESEARCH CONCLUDED**

</div>

---

## 🛑 Project status: CONCLUDED — APPROACH ABANDONED

This project has been **stopped**. It is an archived R&D investigation, preserved
for reference only; it is **no longer being developed**.

The Direct Learning methodology investigated here — a learned (direct) update
operator that attempts to replace iterative gradient-based optimization — **did
not demonstrate sufficient generalization** to replace conventional training. In
every phase and on every held-out test, the learned operators stayed at the
no-update baseline, recovering only ~1–4% of the improvement that the (future
aware) oracle shows is available.

**This conclusion applies only to the methodology investigated in this
repository.** It is **not** a claim that alternative approaches to
neural-network training are impossible in general.

- See [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md) for the complete
  Phases 1–5 summary, quantitative results, failures, and the final conclusion.
- All phase source code, tests, configurations, checkpoints, and experimental
  results remain preserved and are not modified.
- No further phases will be implemented.


---

## 🔬 The research question

Today, neural networks are trained by repeating a tiny, greedy step thousands of times:

```
training data → forward pass → loss → backpropagation → gradient
                                                          │
                                                    optimizer
                                                          │
                                                        W + ΔW   (repeat ~10⁴–10⁶ times)
```

**Hypothesis (not proven):** a mathematical or *learned* mechanism could compute a
substantially more effective parameter transformation **directly** — replacing many
sequential updates with one (or a few) large, intelligent updates at **comparable model
quality but substantially lower total computation**.

> ⚠️ **Honesty first.** This repo does not assume the hypothesis is true. It builds the
> *evidence base*: a rock-solid conventional baseline, an oracle upper bound, and a
> structure analysis of multi-step updates — so that any future "direct" method can be
> judged fairly on **total compute**, not on the number of optimizer steps.

---

## ✨ Why you want a baseline before a new idea

| Name | Meaning |
|---|---|
| `W0 → W1 → … → WN` | The parameter trajectory produced by a conventional optimizer |
| `ΔW = WN − W0` | The *cumulative* effect of N updates — the thing Direct Learning wants to approximate |
| **Oracle** | `W0 + ΔW` reproduces `WN` exactly — an upper bound that **cheats** (it uses the future answer) |
| **Direct method** | Any update derived from information available at the moment of the update — the only fair game |

We never compare "1 step" vs "100 steps". Every method reports **FLOPs, wall-clock time,
forward/backward counts, memory, quality, and convergence**. A method that does 1 update
in 20 minutes is *not* an improvement over 100 updates in 10 minutes.

---

## 📦 What's inside

| Component | Description |
|---|---|
| 🏗️ **`src/model.py`** | Tiny decoder-only Transformer (GPT-style, pre-LayerNorm), ~1.6M params, fully CPU-friendly |
| 🔤 **`src/tokenizer.py`** | Deterministic character-level tokenizer |
| 📚 **`src/dataset.py`** | Tiny deterministic synthetic corpus (no internet downloads) + causal-LM windowing |
| 🏃 **`src/train.py`** | AdamW training loop with trajectory checkpoints, metrics, resume support |
| 📊 **`src/evaluate.py`** | Validation loss/perplexity, gradient & parameter norms |
| 💾 **`src/checkpoint.py`** | Trajectory snapshots (`W0…W100`) + resumable optimizer state |
| 🔬 **`src/phase2/`** | Direct-Update research: baseline audit, oracle, structure analysis, method benchmark |
| 🧠 **`src/phase3/`** | Learned Direct Update Predictor: meta-training, structured parameterisation, ablations, compute accounting |
| ⚙️ **`src/phase4/`** | Generated low-rank directions: update operator, per-layer SVD oracle, behavioural objective, ablations |
| 🔭 **`src/phase5/`** | Generalizable update operator: structured generator, objectives A/B/C, generalization tests A–D, second corpus |
| ✅ **`tests/`** | 81 tests covering model, data, training, checkpoints, Phase-2/3/4/5 research code |

### 📁 Project layout

```
direct-learning/
├── configs/baseline.yaml      # every knob, fully reproducible
├── src/
│   ├── model.py · tokenizer.py · dataset.py · train.py
│   ├── evaluate.py · checkpoint.py · utils.py
│   ├── phase2/                # oracle · analysis · methods · plots · reports
│   ├── phase3/                # trajectory · features · predictor · eval · plots · report
│   ├── phase4/                # operator · oracle · functional · eval · plots · report
│   └── phase5/                # generator · corpus2 · trajectory · eval · plots · report
├── tests/                     # 81 pytest cases
├── results/
│   ├── phase2/                # baseline_audit.md · phase2_summary.md · plots/ · *.json
│   ├── phase3/                # configs · metrics · predictions · plots · ablations · phase3_report.md
│   ├── phase4/                # configs · oracle · metrics · plots · ablations · phase4_report.md
│   └── phase5/                # configs · generalization · ablations · metrics · plots · phase5_report.md
├── data/                      # generated corpus (deterministic, tiny)
└── README.md · requirements.txt · pytest.ini · LICENSE
```

---

## 🚀 Quickstart

```bash
git clone https://github.com/AkaneSakuramori/linear
cd linear

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# CPU-only PyTorch (recommended for this machine):
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch

# Run the tests (all 81 should pass)
.venv/bin/pytest

# Run the Phase-1 baseline (100 AdamW steps, ~1 min)
.venv/bin/python -m src.train --config configs/baseline.yaml

# Run the Phase-2 Direct-Update research (audit + oracle + analysis + methods)
.venv/bin/python -m src.phase2.run --phase1-config configs/baseline.yaml

# Run the Phase-3 Learned Direct Update Predictor (meta-train + direct eval + ablations)
.venv/bin/python -m src.phase3.run --phase1-config configs/baseline.yaml

# Run the Phase-4 Generated Parameter Directions (oracle + operator + ablations)
.venv/bin/python -m src.phase4.run --phase1-config configs/baseline.yaml

# Run the Phase-5 Generalizable Update Operator (objectives + generalization tests)
.venv/bin/python -m src.phase5.run --phase1-config configs/baseline.yaml
```

🖥️ **Hardware-conscious by design** — this runs happily on a shared CPU-only VPS:

```
Hardware:
CPU cores available: 24
CPU threads allocated: 4        # ← never grabs all cores
RAM available: 94.2 GiB
GPU: none                       # ← no CUDA required, ever
```

Resume an interrupted run any time with `--resume` — window sampling is a pure function
of `(step, seed)`, so restarts are bit-exact.

---

## 📈 What Phase 1 delivers

A trustworthy conventional baseline with a **recorded parameter trajectory**:

```
results/
├── config.json          # exact configuration of the run
├── metrics.json         # loss/ppl/norms/FLOPs/RAM summary
├── training.log         # one row per optimizer step (CSV)
└── trajectory/
    ├── step_0000.pt     # W0 (initialization)
    ├── step_0001.pt     # W1
    ├── …                # W2, W5, W10, W25, W50, W75
    └── step_0100.pt     # W100 (final)
```

Phase-1 result on the tiny synthetic corpus (seed 7, 4 threads):

| Metric | Value |
|---|---|
| Parameters | 1,610,752 |
| Train / val tokens | 60,003 / 20,000 |
| Initial → final train loss | 3.46 → **0.85** |
| Final validation loss / ppl | 0.84 / 2.31 |
| Wall time / peak RAM | ~27 s / 0.55 GiB |
| Estimated total FLOPs | ~2.7×10¹² |

---

## 🔍 What Phase 2 investigates

1. **Baseline audit** — verify the FLOPs formula, gradient-clipping behaviour
   (unclipped vs clipped norms), checkpoint contents, and seed reproducibility.
2. **Oracle upper bound** — `W0 + (WN − W0)` reproduces `WN` exactly. *Cheats, but sets the ceiling.*
3. **Structure analysis** — is `ΔW` aligned with any available gradient? Is it low-rank? Sparse?
   How fast does the direction rotate during training?
4. **Direct method benchmark** — `DirectGradient`, `DirectAverageGradient`, `DirectMomentum`,
   `DirectLowRank`, all starting from `W0` with a **single** parameter update.

### 📌 Headline findings

- ✅ The oracle proves a single "perfect" update *can* equal the final model.
- ❌ `ΔW` is **not** a scaled version of any local gradient (cosine ≈ −0.15 with the first
  gradient, ≈ −0.40 with the mean gradient — negative sign is expected: gradients point uphill).
- ❌ `ΔW` is **dense** (~99.5% of parameters move) and **high-rank** (effective rank ≈ 180–230
  at 95% energy) — no low-rank shortcut.
- ❌ Every tested single-update approximation reaches val loss ≈ 2.4–3.5 vs the baseline's **0.84**
  at 100 steps — even ones that spend the *same* N forward/backward passes.
- 🧭 The results do **not** rule out a *learned* update predictor (the true update is a strongly
  non-linear, step-dependent accumulation) — but it must discover genuinely new structure.

Full details: [`results/phase2/baseline_audit.md`](results/phase2/baseline_audit.md) and
[`results/phase2/phase2_summary.md`](results/phase2/phase2_summary.md) · plots in
`results/phase2/plots/`.

---

## 🧠 What Phase 3 investigates

Phase 3 answers the question Phase 2 left open: *can a small **learned** predictor
produce a useful multi-step parameter transformation from information available early
in training (steps 0–K), without running the remaining H−K optimizer steps?*

1. **Meta-training** — replay the Phase-1 recipe from fresh seeds; record only
   step-≤K information (per-step gradients, loss history, per-layer statistics).
2. **Structured parameterisation** — the predictor emits per-layer coefficients for an
   *observed-gradient basis* (mean / first / last gradient), not 1.6M raw parameters:
   `ΔW_pred = Σ_l Σ_j α_lj · D_lj`, `W_pred = W_K + ΔW_pred`.
3. **No-cheat enforcement** — the target `ΔW = W_H − W_K` is used *only* for
   supervised meta-training; the direct method receives nothing beyond step K.
4. **Grid** — K ∈ {5, 10}, H ∈ {25, 50, 100} on 6 train / 2 val / 5 held-out test
   trajectories (seeds 30–33 + reference seed 7).
5. **Ablations** — feature sets A–F (loss / grad / grad+loss / full / compressed /
   rich) and parameterisation bases (mean / first-last / first-last-mean).
6. **Full compute accounting** — meta-training, direct application, conventional
   rollout, and amortized totals are all reported separately.

### 📌 Headline findings

- ❌ **Negative result.** The Direct Update Predictor reaches val loss ≈ 2.1–3.8 across
  the K/H grid vs conventional **0.84–1.52** — it does not approach AdamW quality.
- 🎯 **The limitation is the parameterisation family, not the learning.** Even the
  *oracle* (coefficients chosen with the future answer) reaches only ≈ 2.0–3.3: per-layer
  scaling of a few observed gradients cannot span the true multi-step update.
- 🧭 More observation (K=10) helps a lot (val 2.1 vs 3.4 at K=5) but does not bridge the
  gap; richer features and bases barely move quality.
- 💸 **Compute is not the blocker.** Meta-training costs ~6 conventional runs; the direct
  method amortizes to cheaper-than-conventional after ≈ 6 applications. The failure is
  quality, and it is reported with the meta-training FLOPs **not** hidden.

Full details: [`results/phase3/phase3_report.md`](results/phase3/phase3_report.md) ·
plots in `results/phase3/plots/`, raw data in `results/phase3/metrics|predictions|ablations`.

---

## ⚙️ What Phase 4 investigates

Phase 4 tests the recommendation Phase 3 ended with: can a **learned operator
GENERATE new parameter directions** — as per-layer low-rank factors
`ΔW_l = U_l V_l^T` — rather than scaling observed gradients?

1. **Oracle first** — per-layer SVD of `ΔW_target = W_H − W_K`: does the low-rank
   family even represent a useful update? (The decisive question.)
2. **Learned operator** — a shared MLP + per-layer identity embedding maps
   steps-1..K features to `U_l, V_l` (zero-initialized, reconstruction objective
   `‖U_l V_l^T − ΔW_target_l‖²/‖ΔW_target_l‖²`, invariant to factorization ambiguity).
3. **Grid** — K ∈ {10, 15, 25}, H ∈ {25, 50} (+ a K=25/H=25 zero-target control);
   ranks 1/2/4/8 (oracle), 1/2/4 (learned); 6 train / 2 val / 5 held-out test trajectories.
4. **Ablations** — feature sets (loss / grad / grad+loss / full), behavioural
   (validation-loss) objective pilot.
5. **Full compute accounting** — observation, operator inference, U V^T generation,
   parameter update, meta-training, and amortized totals.

### 📌 Headline findings

- 🎯 **The low-rank family is informative but insufficient at r ≤ 8.** The *oracle*
  (best rank-r of ΔW via SVD) captures ~12% energy at r=1 and ~57% at r=8, improving
  val loss monotonically (K=10,H=25: 2.05 → **1.64**) but **not reaching** conventional
  (1.52); Phase-2's effective rank of the true update is ~180–230.
- ❌ **The learned operator overfits and does not generalize.** Reconstruction MSE
  falls to 0.44 on the 6 meta-train trajectories but the held-out prediction error is
  ~1.0 (≈ predicting zero): on unseen seeds it emits a near-zero update and its quality
  ≈ no-update. Classic overfitting on ~150 (trajectory × layer) examples.
- 🧭 **The bottleneck is learning/generalization, not the family and not compute** —
  opposite of Phase 3. Direct application FLOPs are far below conventional.
- 📋 **Recommendation: MODIFY** — more meta-training data, a more sample-efficient
  structured generation (compressed-basis / hypernetwork), and meta-objectives that
  optimize held-out quality; plus accepting much larger effective rank for full quality.

Full details: [`results/phase4/phase4_report.md`](results/phase4/phase4_report.md) ·
plots in `results/phase4/plots/`, raw data in `results/phase4/metrics|oracle|ablations`.

---

## 🔭 What Phase 5 investigates

Phase 5 asks one question: **can a learned update operator generalize** its
parameter-transformation strategy to unseen trajectories? Phase 4's operator
overfit its 6 meta-training trajectories; Phase 5 scales the data (6→16→32),
uses a structured **compressed-basis generator** (shared encoder → latent `z_l`
→ small coefficients applied onto learned per-layer bases), compares objectives
(A parameter-reconstruction / B behavioural / C combined), and tests four
generalization settings (A unseen seed, B unseen batch ordering, C unseen
initialization, D unseen corpus).

1. **Meta-training data** — 32 train / 8 val / 8 test trajectories (unseen seeds)
   plus a second corpus B (different word distribution, same charset/vocab).
2. **Oracle** — best rank-r of the true update (family ceiling).
3. **Generalization metrics** — % of oracle improvement recovered,
   cos(ΔW_pred, ΔW_target) vs cos(observed-gradient, ΔW_target), per test.
4. **Fairness + compute accounting** — observation / generator inference /
   U V^T / update / meta-training / amortized totals.

### 📌 Headline findings

- ❌ **Definitive negative result: the learned operator does NOT generalize.** Across
  all data sizes (6/16/32), objectives (A/B/C with λ 0.01–1.0), architectures
  (compressed vs flat), ranks (4/8), horizons, and generalization tests A–D, the
  learned update stays at the **no-update baseline**, recovering only **~1–4% of the
  oracle improvement**; cos(ΔW_pred, ΔW_target) ≈ 0.03 (essentially orthogonal).
- 🎯 More data did **not** help; the behavioural objective did **not** help; the
  structured generator did **not** help — the bottleneck is that early-training
  statistics do not determine the future low-rank factors the generator must emit.
- 🧭 The **oracle** still shows the low-rank family is *informative* (rank-8 captures
  ~56% energy and gets close to conventional on short horizons), so the failure is
  the **learned prediction**, not the family.
- 📋 **Recommendation: STOP THIS APPROACH** for the learned-update-operator
  methodology as tested; reconsider the broader hypothesis (the update is a
  high-rank, step-dependent accumulation that small early-statistics-driven
  generators cannot reproduce).

Full details: [`results/phase5/phase5_report.md`](results/phase5/phase5_report.md) ·
plots in `results/phase5/plots/`, raw data in
`results/phase5/metrics|ablations|generalization|oracle`.

---

## 🧩 Project roadmap

- [x] **Phase 1** — CPU baseline + full trajectory `W0…W100`
- [x] **Phase 2** — audit, oracle, structure analysis, direct-method benchmark
- [x] **Phase 3** — meta-learned direct update predictor, parameterisation ablations, compute analysis, reproducible negative result
- [x] **Phase 4** — generated low-rank directions: oracle, learned update operator, generalization-failure analysis, MODIFY recommendation
- [x] **Phase 5** — generalizable update operator: structured generator, objectives A/B/C, generalization tests A–D, second corpus; STOP recommendation
- [x] **Project archived** — `results/FINAL_REPORT.md`; approach **CONCLUDED — APPROACH ABANDONED**
- [ ] *Later* — learned optimizers, model editing, LoRA, GPU scaling (explicitly out of scope for now)

---

## 🛡️ Shared-VPS safety rules

- Default **4 CPU threads** (configurable via `threads` in the YAML) — never auto-grab all cores.
- **No CUDA, no distributed training, no large downloads.**
- Memory footprint is tiny: the whole model is ~7 MB; peak RSS typically < 0.6 GiB.
- One training job at a time; no background processes; everything stops/resumes cleanly.

---

## 📄 License

[MIT](LICENSE) © 2026 Akane Sakuramori.

> **Final reminder:** Direct Learning remains a *research hypothesis*. The baseline and
> experiments in this repo exist to test it honestly — negative results included.
