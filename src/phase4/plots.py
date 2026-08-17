"""Phase-4 plots: oracle, quality by pair, ablations, learned-vs-rank."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_all(baselines, oracle_results, learned, ablations, p4, out_dir) -> list:
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    files = []
    pairs = sorted(baselines.keys())

    # --- 1. oracle: energy and quality vs rank -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for (K, H), rows in sorted(oracle_results.items()):
        ranks = [r["rank"] for r in rows]
        axes[0].plot(ranks, [r["mean_explained_energy"] for r in rows],
                     marker="o", label=f"K{K} H{H}")
        axes[1].plot(ranks, [r["mean_val_loss"] for r in rows],
                     marker="o", label=f"K{K} H{H}")
    axes[0].set_xlabel("rank r"); axes[0].set_ylabel("explained energy (2-D weights)")
    axes[0].set_title("Oracle low-rank: captured update energy"); axes[0].grid(alpha=.3)
    axes[1].set_xlabel("rank r"); axes[1].set_ylabel("mean val loss")
    axes[1].set_title("Oracle low-rank: model quality"); axes[1].grid(alpha=.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(plot_dir, "01_oracle_energy_and_quality.png")
    fig.savefig(p, dpi=110); plt.close(fig); files.append(p)

    # --- 2. quality by pair (grouped bars) -----------------------------------
    labels = [f"K{K}\nH{H}" for (K, H) in pairs]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.19
    x = range(len(pairs))

    def _oracle_rank(pair, rank):
        for r in oracle_results[pair]:
            if r["rank"] == rank:
                return r["mean_val_loss"]
        return 0.0

    def _learned(pair, pref=4):
        ranks = sorted(learned[pair].keys())
        if not ranks:
            return 0.0
        r = min(ranks, key=lambda r: abs(r - pref))
        return learned[pair][r]["summary"]["mean_val_loss"]

    series = [
        ("no update (W_K)", [baselines[p]["no_update"]["mean_val_loss"] for p in pairs], "#9e9e9e"),
        ("AdamW W_H", [baselines[p]["conventional"]["mean_val_loss"] for p in pairs], "#2e7d32"),
        ("oracle low-rank r4", [_oracle_rank(p, 4) for p in pairs], "#8e24aa"),
        ("learned low-rank", [_learned(p, 4) for p in pairs], "#c62828"),
    ]
    for i, (name, vals, color) in enumerate(series):
        ax.bar([xx + (i - 1.5) * width for xx in x], vals, width, label=name, color=color)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("mean validation loss (test trajectories)")
    ax.set_title("Phase 4: generated low-rank update vs baselines")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(plot_dir, "02_quality_by_pair.png")
    fig.savefig(p, dpi=110); plt.close(fig); files.append(p)

    # --- 3. ablations --------------------------------------------------------
    for (K, H), ab in ablations.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fs_names = list(ab["feature_sets"].keys())
        vals = [ab["feature_sets"][f]["summary"]["mean_val_loss"] for f in fs_names]
        axes[0].bar(fs_names, vals, color="#4C72B0")
        axes[0].axhline(baselines[(K, H)]["conventional"]["mean_val_loss"],
                        color="green", ls="--", label="AdamW W_H")
        axes[0].axhline(baselines[(K, H)]["no_update"]["mean_val_loss"],
                        color="gray", ls=":", label="no update W_K")
        axes[0].set_title(f"Feature-set ablation (K={K}, H={H})")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
        # learned vs rank (incl. behavior pilot)
        ranks = sorted(learned[(K, H)].keys())
        axes[1].plot(ranks, [learned[(K, H)][r]["summary"]["mean_val_loss"] for r in ranks],
                     marker="o", label="learned operator")
        axes[1].axhline(baselines[(K, H)]["conventional"]["mean_val_loss"],
                        color="green", ls="--", label="AdamW W_H")
        axes[1].axhline(baselines[(K, H)]["no_update"]["mean_val_loss"],
                        color="gray", ls=":", label="no update W_K")
        if ab["behavior"]["summary"]["mean_val_loss"]:
            axes[1].axhline(ab["behavior"]["summary"]["mean_val_loss"],
                            color="purple", ls="-.", label="+ behavior obj.")
        axes[1].set_xlabel("rank r"); axes[1].set_ylabel("mean val loss")
        axes[1].set_title(f"Learned operator vs rank (K={K}, H={H})")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
        fig.tight_layout()
        p = os.path.join(plot_dir, f"03_ablations_K{K}_H{H}.png")
        fig.savefig(p, dpi=110); plt.close(fig); files.append(p)

    return files