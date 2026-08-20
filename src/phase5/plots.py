"""Phase-5 plots."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_all(ab, oracle, gen_res, p5, out_dir):
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    files = []

    # 1. data-size ablation: val loss and % oracle recovered
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sizes = sorted(ab["data_size"].keys())
    ax.plot(sizes, [ab["data_size"][n]["mean_val_loss"] for n in sizes],
            marker="o", label="learned (compressed, combined λ=0.1)")
    for n in sizes:
        ax.axhline(ab["data_size"][n]["no_update"], color="gray", ls=":", alpha=.5)
    ax.axhline(ab["data_size"][sizes[0]]["oracle_r4"], color="purple", ls="--",
               label="oracle r4")
    ax.set_xlabel("# meta-training trajectories")
    ax.set_ylabel("mean val loss (held-out test)")
    ax.set_title("Phase 5: does more data improve generalization?")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(plot_dir, "01_data_size.png")
    fig.savefig(p, dpi=110); plt.close(fig); files.append(p)

    # 2. objective ablation
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(ab["objective"].keys())
    vals = [ab["objective"][k]["mean_val_loss"] for k in labels]
    ax.bar(labels, vals, color="#4C72B0")
    ax.set_ylabel("mean val loss (held-out test)")
    ax.set_title("Objective ablation (16 train, r4, compressed)")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    p = os.path.join(plot_dir, "02_objective.png")
    fig.savefig(p, dpi=110); plt.close(fig); files.append(p)

    # 3. generalization: bar chart of no_update / oracle / learned per test
    fig, ax = plt.subplots(figsize=(9, 5))
    cats = list(gen_res.keys())
    x = range(len(cats))
    w = 0.25
    for i, (key, color, lab) in enumerate(
            [("no_update", "#9e9e9e", "no update"),
             ("oracle_r4", "#8e24aa", "oracle r4"),
             ("mean_val_loss", "#c62828", "learned")]):
        ax.bar([xx + (i - 1) * w for xx in x],
               [gen_res[c][key] for c in cats], w, label=lab, color=color)
    ax.set_xticks(list(x)); ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("val loss"); ax.set_title("Generalization tests (K=10, H=25, r=4)")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    p = os.path.join(plot_dir, "03_generalization.png")
    fig.savefig(p, dpi=110); plt.close(fig); files.append(p)

    # 4. oracle energy vs rank
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (K, H), rows in sorted(oracle.items()):
        ax.plot([r["rank"] for r in rows], [r["mean_explained_energy"] for r in rows],
                marker="o", label=f"K{K} H{H}")
    ax.set_xlabel("rank"); ax.set_ylabel("explained energy (2-D)")
    ax.set_title("Oracle: low-rank energy capture")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(plot_dir, "04_oracle_energy.png")
    fig.savefig(p, dpi=110); plt.close(fig); files.append(p)

    return files
