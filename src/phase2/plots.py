"""Plot generation for Phase-2 analysis (matplotlib, Agg backend)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIGSIZE = (6.2, 4.2)


def _save(fig, path) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_loss_trajectory(data: dict, outdir: str) -> str:
    ema = data["train_emas"]
    steps = np.arange(len(ema))
    steps = steps[1:]
    ema = ema[1:]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(steps, ema, label="train loss (EMA)")
    vs = sorted(data["val_records"].keys())
    ax.plot(vs, [data["val_records"][s]["loss"] for s in vs], "o-",
            label="val loss")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Phase-1 AdamW baseline trajectory")
    ax.legend(); _save(fig, os.path.join(outdir, "01_loss_trajectory.png"))
    return "01_loss_trajectory.png"


def plot_grad_norms(data: dict, outdir: str) -> str:
    uc = data["grad_norms_unclipped"][1:]
    cc = data["grad_norms_clipped"][1:]
    steps = np.arange(1, len(uc) + 1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(steps, uc, label="unclipped")
    ax.plot(steps, cc, label="clipped (cap=1.0)")
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("global grad L2 norm")
    ax.set_title("Gradient norm: unclipped vs clipped")
    ax.legend(); _save(fig, os.path.join(outdir, "02_grad_norms.png"))
    return "02_grad_norms.png"


def plot_delta_growth(analysis: dict, outdir: str) -> str:
    hs = analysis["horizons"]
    norm = [analysis["deltas"][str(h)]["delta_l2"] for h in hs]
    frac = [analysis["deltas"][str(h)]["delta_w0_ratio"] for h in hs]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(hs, norm, "o-", label="||ΔW_N||₂")
    ax.set_xlabel("horizon N"); ax.set_ylabel("cumulative update L2 norm")
    ax.set_title("Cumulative update magnitude vs horizon")
    ax.legend(); _save(fig, os.path.join(outdir, "03_delta_norm_growth.png"))
    return "03_delta_norm_growth.png"


def plot_cosine_alignment(analysis: dict, outdir: str) -> str:
    hs = analysis["horizons"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for key, label in [("cos_sim_first_grad", "cos(ΔW_N, first grad)"),
                       ("cos_sim_avg_grad", "cos(ΔW_N, avg grad N)"),
                       ("cos_sim_momentum", "cos(ΔW_N, momentum N)")]:
        vals = [analysis["deltas"][str(h)][key] for h in hs]
        ax.plot(hs, vals, "o-", label=label)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("horizon N"); ax.set_ylabel("cosine")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Alignment of cumulative update with candidate directions")
    ax.legend(fontsize=7); _save(fig, os.path.join(outdir, "04_cosine_alignment.png"))
    return "04_cosine_alignment.png"


def plot_direction_change(analysis: dict, outdir: str) -> str:
    dc = analysis["direction_change"]
    xs = [f"{d['horizon_a']}→{d['horizon_b']}" for d in dc]
    ys = [d["cosine"] for d in dc]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(xs, ys)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("cosine between consecutive cumulative deltas")
    ax.set_title("How the cumulative direction changes during training")
    _save(fig, os.path.join(outdir, "05_direction_change.png"))
    return "05_direction_change.png"


def plot_per_layer_magnitudes(analysis: dict, outdir: str) -> str:
    layer = analysis["deltas"]["100"]["per_layer"]
    names = list(layer.keys())
    up = [layer[n]["update_l2"] for n in names]
    gr = [layer[n]["grad_l2"] if layer[n]["grad_l2"] else 0.0 for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(x - 0.2, up, width=0.4, label="ΔW L2 (cumulative)")
    ax.bar(x + 0.2, gr, width=0.4, label="first gradient L2")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("L2 norm (log)")
    ax.set_title("Per-layer update vs first-gradient magnitudes (horizon 100)")
    ax.legend(fontsize=7)
    _save(fig, os.path.join(outdir, "06_per_layer_magnitudes.png"))
    return "06_per_layer_magnitudes.png"


def plot_svd_energy(analysis: dict, outdir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, h in zip(axes, ("5", "100")):
        svd = analysis["svd"].get(h, {})
        for name in list(svd.keys())[:6]:
            re = svd[name]["rank_energy"]
            ranks = [int(k) for k in re.keys()]
            ax.plot(ranks, [re[str(k)] for k in ranks], "o-", label=name.split(".")[-1][:22])
        ax.set_xlabel("rank k"); ax.set_ylabel("fraction of Frobenius energy")
        ax.set_title(f"Low-rank energy of ΔW (horizon {h})")
        ax.legend(fontsize=6)
        ax.set_xscale("log")
    _save(fig, os.path.join(outdir, "07_svd_energy.png"))
    return "07_svd_energy.png"


def plot_svd_spectrum(analysis: dict, outdir: str) -> str:
    svd = analysis["svd"].get("100", {})
    names = ["blocks.0.attn.qkv.weight", "blocks.0.mlp.fc1.weight",
             "blocks.1.mlp.fc2.weight", "lm_head.weight"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name in names:
        if name in svd:
            ax.plot(np.arange(1, len(svd[name]["sv_top20"]) + 1),
                    np.maximum(svd[name]["sv_top20"], 1e-9), "o-", label=name.split(".")[-1])
    ax.set_yscale("log")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("σᵢ (log)")
    ax.set_title("Singular-value spectrum of ΔW at horizon 100")
    ax.legend(fontsize=7)
    _save(fig, os.path.join(outdir, "08_svd_spectrum.png"))
    return "08_svd_spectrum.png"


def plot_method_comparison(results: dict, outdir: str) -> str:
    notes = results["direct"]
    hs = sorted({r["horizon"] for r in notes})
    method_labels = ["DirectGradient", "DirectAverageGradient", "DirectMomentum"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    ax = axes[0]
    for m in method_labels:
        ys = [next(r["final_val_loss"] for r in notes
                   if r["name"] == m and r["horizon"] == h) for h in hs]
        ax.plot(hs, ys, "o-", label=m)
    bl = next(r["final_val_loss"] for r in results["baseline"]) if results.get("baseline") else None
    if bl is not None:
        ax.axhline(bl, color="k", ls="--", label=f"BaselineAdamW val ({bl:.3f})")
    ax.set_xlabel("horizon N"); ax.set_ylabel("final val loss")
    ax.set_title("Direct approximations vs baseline")
    ax.legend(fontsize=7)
    ax2 = axes[1]
    fl = [r["compute"]["flops_est"] / 1e12 for r in notes
          if r["name"] in method_labels and r["horizon"] == 100]
    labs = [r["name"] for r in notes if r["name"] in method_labels and r["horizon"] == 100]
    ax2.bar(labs, fl)
    if results.get("baseline"):
        ax2.axhline(results["baseline"][0]["compute"]["flops_est"] / 1e12,
                    color="k", ls="--", label="baseline")
    ax2.set_ylabel("estimated FLOPs (×10¹²)")
    ax2.tick_params(axis="x", rotation=45, labelsize=6)
    ax2.set_title("Compute used (horizon 100 methods)")
    ax2.legend(fontsize=7)
    _save(fig, os.path.join(outdir, "09_method_comparison.png"))
    return "09_method_comparison.png"


def generate_all(data: dict, outdir: str) -> list:
    made = [
        plot_loss_trajectory(data, outdir),
        plot_grad_norms(data, outdir),
        plot_delta_growth(data["analysis"], outdir),
        plot_cosine_alignment(data["analysis"], outdir),
        plot_direction_change(data["analysis"], outdir),
        plot_per_layer_magnitudes(data["analysis"], outdir),
        plot_svd_energy(data["analysis"], outdir),
        plot_svd_spectrum(data["analysis"], outdir),
    ]
    if "results" in data:
        made.append(plot_method_comparison(data["results"], outdir))
    return made