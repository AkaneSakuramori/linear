"""Phase-3 research driver.

Usage:
    python -m src.phase3.run --phase1-config configs/baseline.yaml

Flow (all deterministic, CPU-only, ≤4 threads):
  1. generate AdamW trajectories for meta-train/val/test seeds,
  2. build compact meta-datasets (features + least-squares targets) per (K,H),
  3. meta-train the Direct Update Predictor per (K,H),
  4. evaluate direct application on held-out test trajectories (no future info),
  5. run feature-set and parameterisation ablations,
  6. write plots and results/phase3/phase3_report.md.

Trajectories are cached to results/phase3/checkpoints/ so stages can rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import torch

from src.dataset import build_corpora
from src.utils import (configure_threads, load_config, print_hardware_report,
                       save_json, set_seed)
from src.phase3.config import Phase3Config
from src.phase3.features import (build_features, compute_alpha_star,
                                 feature_dim, param_names, rel_param_distance)
from src.phase3.predictor import Predictor, train_predictor
from src.phase3.evaluate import (Phase3Eval, direct_compute, direct_result,
                                 oracle_result, conventional_result,
                                 predictor_train_flops, predictor_inference_flops)

RECORD_STEPS = [5, 10, 25, 50, 100]


# ---------------------------------------------------------------------------
# Trajectory generation (with on-disk cache)
# ---------------------------------------------------------------------------
def load_or_generate_trajectory(phase1_cfg, corpora, seed: int, cache_dir: str,
                                max_steps: int, k_max: int) -> dict:
    path = os.path.join(cache_dir, f"trajectory_seed{seed}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    from src.phase3.trajectory import generate_trajectory
    rec = generate_trajectory(phase1_cfg, corpora, seed, max_steps,
                              RECORD_STEPS, k_max,
                              eval_max_windows=phase1_cfg.eval_max_windows)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(rec, path)
    return rec


# ---------------------------------------------------------------------------
# Meta-dataset building
# ---------------------------------------------------------------------------
def build_meta_dataset(records: dict, p3: Phase3Config, phase1_cfg,
                       feature_set: str, basis: str, K: int, H: int):
    """Stack per-(trajectory, layer) features and alpha-star targets."""
    layers = param_names(next(iter(records.values())))
    rows_x, rows_y = [], []
    for seed, rec in records.items():
        for layer in layers:
            rows_x.append(build_features(rec, K, layer, feature_set, phase1_cfg.lr))
            rows_y.append(compute_alpha_star(rec, K, H, basis, layer))
    X = torch.stack(rows_x)
    Y = torch.stack(rows_y)
    return X, Y, layers


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def no_update_quality(rec: dict, K: int, eval_h: Phase3Eval) -> dict:
    return {"train_loss": None, "val_loss": rec["val_records"][K]["loss"],
            "val_ppl": rec["val_records"][K]["ppl"]}


def _avg(seq):
    vals = [v for v in seq if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def summarize_runs(runs: list) -> dict:
    return {
        "mean_val_loss": _avg([r["quality"]["val_loss"] for r in runs]),
        "mean_val_ppl": _avg([r["quality"]["val_ppl"] for r in runs]),
        "mean_rel_param_dist_to_WH": _avg([r.get("rel_param_dist_to_WH") for r in runs]),
        "mean_rel_delta_error": _avg([r.get("rel_delta_error") for r in runs]),
        "n": len(runs),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-3 Direct Update Predictor")
    p.add_argument("--phase1-config", default="configs/baseline.yaml")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main(argv=None) -> dict:
    args = parse_args() if argv is None else parse_args(argv)
    p1 = load_config(args.phase1_config)
    if args.threads:
        p1.threads = args.threads

    p3 = Phase3Config()
    if args.out_dir:
        p3.out_dir = args.out_dir

    configure_threads(p1.threads)
    print_hardware_report(p1.threads)
    set_seed(42)
    t_start = time.time()

    out = p3.out_dir
    for sub in ("configs", "checkpoints", "metrics", "predictions", "plots", "ablations"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    cache_dir = os.path.join(out, "checkpoints")
    save_json(os.path.join(out, "configs", "phase3_config.json"), p3.resolved_dict())

    corpora = build_corpora(p1.data_dir, p1.corpus_train_chars, p1.corpus_val_chars,
                            p1.corpus_seed, p1.corpus_val_seed, force=p1.regenerate)

    split_info = {
        "meta_train_seeds": p3.meta_train_seeds,
        "meta_val_seeds": p3.meta_val_seeds,
        "meta_test_seeds": p3.meta_test_seeds,
        "reference_seed": p3.reference_seed,
        "corpus": "shared deterministic corpus (same for every trajectory); "
                  "trajectories differ by init + batch sampling seed",
    }
    save_json(os.path.join(out, "configs", "split.json"), split_info)

    # ---------------- 1. trajectories --------------------------------------
    print("Generating trajectories ...")
    split_records = {}
    for split, seeds in (("train", p3.meta_train_seeds),
                         ("val", p3.meta_val_seeds),
                         ("test", p3.meta_test_seeds + [p3.reference_seed])):
        split_records[split] = {
            seed: load_or_generate_trajectory(p1, corpora, seed, cache_dir,
                                              p3.max_steps, p3.k_max,
                                              )
            for seed in seeds
        }

    # ---------------- 2. meta datasets + predictors -------------------------
    eval_h = Phase3Eval(p1, corpora, torch.device("cpu"))
    n_params = sum(v.numel() for v in split_records["test"][p3.meta_test_seeds[0]]
                   ["w_states"][0].values())

    print("Meta-training predictors ...")
    predictors = {}
    pair_results = {}
    meta_train_flops = sum(rec["flops_per_step"] * 100.0
                           for rec in split_records["train"].values())
    meta_train_time = sum(rec["elapsed_sec"] for rec in split_records["train"].values())

    for (K, H) in p3.all_pairs():
        X, Y, layers = build_meta_dataset(split_records["train"], p3, p1,
                                          p3.feature_set, p3.basis, K, H)
        Xv, Yv, _ = build_meta_dataset(split_records["val"], p3, p1,
                                       p3.feature_set, p3.basis, K, H)
        predictor, hist = train_predictor(X, Y, Xv, Yv, hidden=p3.hidden,
                                          steps=p3.train_steps, lr=p3.lr,
                                          batch_size=p3.batch_size,
                                          seed=p3.predictor_seed)
        path = os.path.join(cache_dir, f"predictor_K{K}_H{H}_{p3.feature_set}_{p3.basis}.pt")
        predictor.save(path)
        predictors[(K, H)] = predictor
        save_json(os.path.join(out, "metrics", f"train_{K}_{H}.json"),
                  {"K": K, "H": H, "feature_set": p3.feature_set, "basis": p3.basis,
                   "history": hist, "layers": layers})

        # ---- 4. evaluation on held-out test trajectories -------------------
        runs_direct, runs_oracle, runs_conv, runs_none = [], [], [], []
        for seed, rec in split_records["test"].items():
            runs_direct.append(direct_result(rec, K, H, p3.basis, predictor, eval_h,
                                             p3.feature_set, p1.lr))
            runs_oracle.append(oracle_result(rec, K, H, p3.basis, eval_h))
            runs_conv.append(conventional_result(rec, K, H))
            runs_none.append({"quality": no_update_quality(rec, K, eval_h),
                              "rel_param_dist_to_WH": rel_param_distance(
                                  rec["w_states"][K], rec["w_states"][H])})
        comp = direct_compute(rec, K, H, predictor_inference_flops(
            [feature_dim(p3.feature_set, K), p3.hidden, p3.hidden, Y.shape[1]]),
            n_params)
        pair_results[(K, H)] = {
            "K": K, "H": H,
            "direct": summarize_runs(runs_direct),
            "oracle": summarize_runs(runs_oracle),
            "conventional": summarize_runs(runs_conv),
            "no_update": summarize_runs(runs_none),
            "compute": comp,
            "predictor_history": hist,
            "per_test_seed": {
                seed: {"direct": runs_direct[i], "oracle": runs_oracle[i],
                       "conventional": runs_conv[i]}
                for i, seed in enumerate(split_records["test"])
            },
        }
        save_json(os.path.join(out, "metrics", f"pair_K{K}_H{H}.json"),
                  pair_results[(K, H)])
        save_json(os.path.join(out, "predictions", f"test_predictions_K{K}_H{H}.json"),
                  pair_results[(K, H)]["per_test_seed"])
        print(f"  K={K} H={H}: direct val {pair_results[(K,H)]['direct']['mean_val_loss']:.4f} "
              f"| oracle {pair_results[(K,H)]['oracle']['mean_val_loss']:.4f} "
              f"| conv {pair_results[(K,H)]['conventional']['mean_val_loss']:.4f}")

    # ---------------- 5. ablations ------------------------------------------
    ablations = run_ablations(split_records, p3, p1, eval_h, n_params, cache_dir)

    # ---------------- 6. compute ledger -------------------------------------
    def _basis_out_dim(basis, K):
        return {"mean": 1, "first_last": 2, "first_last_mean": 3}.get(basis, K)

    predictor_train_cost = sum(
        predictor_train_flops(p3.train_steps, p3.batch_size,
                              [feature_dim(p3.feature_set, K), p3.hidden, p3.hidden,
                               _basis_out_dim(p3.basis, K)])
        for (K, H) in p3.all_pairs())
    ledger = {
        "meta_training": {
            "trajectory_gen_flops": meta_train_flops,
            "trajectory_gen_wall_sec": meta_train_time,
            "predictor_train_flops": predictor_train_cost,
            "total_meta_flops": meta_train_flops + predictor_train_cost,
        },
        "application": {
            "direct_total_flops": pair_results[(5, 100)]["compute"]["direct_total_flops"],
            "conventional_total_flops": pair_results[(5, 100)]["compute"]["conventional_total_flops"],
        },
        "amortization": {},
    }
    direct_f = pair_results[(5, 100)]["compute"]["direct_total_flops"]
    conv_f = pair_results[(5, 100)]["compute"]["conventional_total_flops"]
    meta_f = ledger["meta_training"]["total_meta_flops"]
    for n_app in (1, 10, 100, 1000):
        ledger["amortization"][str(n_app)] = {
            "direct_total": meta_f + n_app * direct_f,
            "conventional_total": n_app * conv_f,
        }
    save_json(os.path.join(out, "metrics", "compute_ledger.json"), ledger)

    # ---------------- 7. plots + report -------------------------------------
    from src.phase3 import plots as plots_mod
    plot_files = plots_mod.generate_all(pair_results, ablations, p3, out)

    from src.phase3.report import write_report
    report_path = write_report(p3, pair_results, ablations, ledger, split_info,
                               out, predictor_hist=predictors[(5, 100)],
                               reference_seed=p3.reference_seed)

    total_sec = time.time() - t_start
    print(f"\nPhase-3 complete in {total_sec:.1f}s -> {os.path.abspath(out)}")
    print("plots:", ", ".join(plot_files))
    print("report:", report_path)
    return {"pair_results": pair_results, "ablations": ablations, "ledger": ledger}


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------
def run_ablations(split_records, p3: Phase3Config, p1, eval_h: Phase3Eval,
                  n_params: int, cache_dir: str) -> dict:
    out = os.path.join(p3.out_dir, "ablations")
    ablations = {}
    for (K, H) in p3.ablation_pairs:
        feats = {}
        for fs in p3.ablation_feature_sets:
            X, Y, layers = build_meta_dataset(split_records["train"], p3, p1, fs,
                                              p3.basis, K, H)
            Xv, Yv, _ = build_meta_dataset(split_records["val"], p3, p1, fs,
                                           p3.basis, K, H)
            predictor, hist = train_predictor(X, Y, Xv, Yv, hidden=p3.hidden,
                                              steps=p3.train_steps, lr=p3.lr,
                                              batch_size=p3.batch_size,
                                              seed=p3.predictor_seed)
            predictor.save(os.path.join(cache_dir, f"abl_{fs}_K{K}_H{H}.pt"))
            runs = [direct_result(rec, K, H, p3.basis, predictor, eval_h, fs, p1.lr)
                    for rec in split_records["test"].values()]
            feats[fs] = {"summary": summarize_runs(runs),
                         "per_test_seed": [r["quality"]["val_loss"] for r in runs],
                         "history": hist}
        bases = {}
        for b in p3.ablation_bases:
            if b == p3.basis:
                continue
            X, Y, layers = build_meta_dataset(split_records["train"], p3, p1,
                                              p3.feature_set, b, K, H)
            Xv, Yv, _ = build_meta_dataset(split_records["val"], p3, p1,
                                           p3.feature_set, b, K, H)
            predictor, hist = train_predictor(X, Y, Xv, Yv, hidden=p3.hidden,
                                              steps=p3.train_steps, lr=p3.lr,
                                              batch_size=p3.batch_size,
                                              seed=p3.predictor_seed)
            predictor.save(os.path.join(cache_dir, f"abl_basis_{b}_K{K}_H{H}.pt"))
            runs = [direct_result(rec, K, H, b, predictor, eval_h,
                                  p3.feature_set, p1.lr)
                    for rec in split_records["test"].values()]
            bases[b] = {"summary": summarize_runs(runs),
                        "per_test_seed": [r["quality"]["val_loss"] for r in runs]}
        ablations[(K, H)] = {"feature_sets": feats, "bases": bases}
        save_json(os.path.join(out, f"ablation_K{K}_H{H}.json"), ablations[(K, H)])
        print(f"  ablations K={K} H={H}: "
              + ", ".join(f"{fs}={feats[fs]['summary']['mean_val_loss']:.3f}"
                          for fs in feats))
    return ablations


if __name__ == "__main__":
    main()
