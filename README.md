<div align="center">

# 🧠 Direct Learning

### A CPU-first research framework for testing whether neural-network training can be made dramatically more compute-efficient.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch CPU](https://img.shields.io/badge/pytorch-2.x%20CPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![No GPU required](https://img.shields.io/badge/GPU-none_required-6B46C1?style=for-the-badge)
![Tests](https://img.shields.io/badge/tests-40%20passing-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-yellowgreen?style=for-the-badge)

**Research status: hypothesis under investigation — nothing here is claimed to work yet.**

</div>

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
| ✅ **`tests/`** | 40 tests covering model, data, training, checkpoints, and Phase-2 smoke tests |

### 📁 Project layout

```
direct-learning/
├── configs/baseline.yaml      # every knob, fully reproducible
├── src/
│   ├── model.py · tokenizer.py · dataset.py · train.py
│   ├── evaluate.py · checkpoint.py · utils.py
│   └── phase2/                # oracle · analysis · methods · plots · reports
├── tests/                     # 40 pytest cases
├── results/                   # config.json · metrics.json · training.log · trajectory/
│   └── phase2/                # baseline_audit.md · phase2_summary.md · plots/ · *.json
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

# Run the tests (all 40 should pass)
.venv/bin/pytest

# Run the Phase-1 baseline (100 AdamW steps, ~1 min)
.venv/bin/python -m src.train --config configs/baseline.yaml

# Run the Phase-2 Direct-Update research (audit + oracle + analysis + methods)
.venv/bin/python -m src.phase2.run --phase1-config configs/baseline.yaml
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

## 🧩 Project roadmap

- [x] **Phase 1** — CPU baseline + full trajectory `W0…W100`
- [x] **Phase 2** — audit, oracle, structure analysis, direct-method benchmark
- [ ] **Phase 3** — small *learned* Direct-Update predictor, benchmarked against Phase-2 numbers
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
