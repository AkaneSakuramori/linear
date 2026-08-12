"""Phase-3 plots: quality vs horizon, prediction error, ablations."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _hs_pair(pair_results):
    hs = sorted({p[1] for p in pair_results})
    return hs


def _mean_val(pair_results, kind, H):
    vals = []
    for (K, h), r in pair_results.items():
        if h == H and r[kind]["mean_val_loss"] is not None:
            vals.append(r[kind]["mean_val_loss"])
    return sum(vals) / len(vals) if vals else None


def generate_all(pair_results, ablations, p3, out_dir) -> list:
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    files = []

    hs = sorted({p[1] for p in pair_results})

    # --- 1. quality vs target horizon ---------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, label, marker in (("conventional", "AdamW (conventional)", "o"),
                                ("oracle", "oracle (per-layer ceiling)", "s"),
                                ("direct", "Direct Update Predictor", "D"),
                                ("no_update", "no update (stay at W_K)", "^")):
        xs, ys = [], []
        for h in hs:
            v = _mean_val(pair_results, kind, h)
            if v is not None:
                xs.append(h)
                ys.append(v)
        if xs:
            ax.plot(xs, ys, marker=marker, label=label)
    ax.set_xlabel("target horizon H")
    ax.set_ylabel("mean validation loss (test trajectories)")
    ax.set_title("Phase 3: direct vs conventional quality (K = 5 / 10)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plot_dir, "01_quality_vs_horizon.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    files.append(p)

    # --- 2. prediction error vs horizon -------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs, y1, y2 = [], [], []
    for (K, h), r in sorted(pair_results.items()):
        xs.append(f"K{K} H{h}")
        y1.append(r["direct"].get("mean_rel_delta_error"))
        y2.append(r["direct"].get("mean_rel_param_dist_to_WH"))
    idx = range(len(xs))
    ax.plot(idx, y1, marker="o", label="rel ΔW prediction error")
    ax.plot(idx, y2, marker="s", label="rel param distance to W_H")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(xs, rotation=30, ha="right")
    ax.set_ylabel("relative error")
    ax.set_title("Phase 3: prediction quality in parameter space")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plot_dir, "02_prediction_error.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    files.append(p)

    # --- 3. feature-set ablation --------------------------------------------
    for (K, H), ab in ablations.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        labels = list(ab["feature_sets"].keys())
        vals = [ab["feature_sets"][fs]["summary"]["mean_val_loss"] for fs in labels]
        ax.bar(labels, vals, color="#4C72B0")
        conv = pair_results[(K, H)]["conventional"]["mean_val_loss"]
        ax.axhline(conv, color="green", ls="--", label=f"AdamW W_{H}")
        ax.axhline(pair_results[(K, H)]["no_update"]["mean_val_loss"],
                   color="gray", ls=":", label=f"stay at W_{K}")
        ax.set_ylabel("mean validation loss")
        ax.set_title(f"Feature ablation (K={K}, H={H})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(plot_dir, f"03_ablation_features_K{K}_H{H}.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        files.append(p)

        if ab["bases"]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            labels = [p3.basis] + list(ab["bases"].keys())
            vals = [pair_results[(K, H)]["direct"]["mean_val_loss"]]
            for b in ab["bases"].keys():
                vals.append(ab["bases"][b]["summary"]["mean_val_loss"])
            ax.bar(labels, vals, color="#55A868")
            ax.axhline(conv, color="green", ls="--", label=f"AdamW W_{H}")
            ax.axhline(pair_results[(K, H)]["oracle"]["mean_val_loss"],
                       color="purple", ls=":", label="oracle ceiling")
            ax.set_ylabel("mean validation loss")
            ax.set_title(f"Parameterisation ablation (K={K}, H={H})")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            p = os.path.join(plot_dir, f"04_ablation_basis_K{K}_H{H}.png")
            fig.savefig(p, dpi=110)
            plt.close(fig)
            files.append(p)

    return files
