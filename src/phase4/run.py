"""Phase-4 research driver.

Usage:
    python -m src.phase4.run --phase1-config configs/baseline.yaml

Flow (deterministic, CPU-only, ≤4 threads):
  1. generate AdamW trajectories (meta-train/val/test) with records to K=25,
  2. oracle experiment: best rank-r of ΔW_target per layer (future answer),
  3. meta-train the generated low-rank update operator per (K, H, rank),
  4. evaluate direct application on held-out test trajectories,
  5. ablations (feature sets, ranks, behavioural objective pilot),
  6. compute accounting + plots + results/phase4/phase4_report.md.

Trajectories are cached under results/phase4/checkpoints/.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

import torch

from src.dataset import build_corpora
from src.utils import (configure_threads, load_config, print_hardware_report,
                       save_json, set_seed)
from src.phase3.evaluate import Phase3Eval
from src.phase4.config import Phase4Config
from src.phase4.features import build_features, feature_dim
from src.phase4.operator import (build_segments, generate_deltas, offsets_of,
                                 total_gen, train_operator)
from src.phase4.oracle import oracle_result
from src.phase4.evaluate import (conventional_result, learned_direct_compute,
                                 learned_result, no_update_result,
                                 operator_train_flops,
                                 phase3_predictor_result)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-4 Generated Parameter Directions")
    p.add_argument("--phase1-config", default="configs/baseline.yaml")
    p.add_argument("--phase4-config", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--threads", type=int, default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Trajectory loading (reuses the Phase-3 generator; broader records)
# ---------------------------------------------------------------------------
def load_or_generate_trajectory(phase1_cfg, corpora, seed: int, cache_dir: str,
                                p4: Phase4Config) -> dict:
    path = os.path.join(cache_dir, f"trajectory_seed{seed}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    from src.phase3.trajectory import generate_trajectory
    rec = generate_trajectory(phase1_cfg, corpora, seed, p4.max_steps,
                              p4.record_steps, p4.k_max,
                              eval_max_windows=phase1_cfg.eval_max_windows)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(rec, path)
    return rec


def build_examples(train_records, p4: Phase4Config, phase1_cfg, K: int, H: int,
                   segs, feature_set: str = None):
    layers = [s["name"] for s in segs]
    fs = feature_set or p4.feature_set
    X, idx, targets = [], [], []
    for rec in train_records.values():
        for i, name in enumerate(layers):
            X.append(build_features(rec, K, name, fs, phase1_cfg.lr))
            idx.append(i)
            targets.append(rec["w_states"][H][name].float()
                           - rec["w_states"][K][name].float())
    return torch.stack(X), torch.tensor(idx, dtype=torch.long), targets


def _avg(seq):
    vals = [v for v in seq if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def summarize(runs, quality_key="quality"):
    return {
        "mean_val_loss": _avg([r[quality_key]["val_loss"] for r in runs]),
        "mean_val_ppl": _avg([r[quality_key]["val_ppl"] for r in runs]),
        "n": len(runs),
    }


def save_operator(operator, path, segs, rank, in_dim, hidden, layer_emb_dim,
                  history=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": operator.state_dict(), "segs": segs, "rank": rank,
                "in_dim": in_dim, "hidden": hidden,
                "layer_emb_dim": layer_emb_dim, "history": history or []},
               path)


def load_operator(path, rank, segs):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    from src.phase4.operator import UpdateOperator
    in_dim, hidden, emb = payload["in_dim"], payload["hidden"], payload["layer_emb_dim"]
    op = UpdateOperator(in_dim, hidden, emb, len(segs), total_gen(segs, rank),
                        torch.zeros(in_dim), torch.ones(in_dim))
    op.load_state_dict(payload["state"])
    return op, payload.get("history") or []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> dict:
    args = parse_args() if argv is None else parse_args(argv)
    p1 = load_config(args.phase1_config)
    if args.threads:
        p1.threads = args.threads

    p4 = Phase4Config()
    if args.phase4_config:
        import yaml
        with open(args.phase4_config, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for k, v in raw.items():
            if hasattr(p4, k):
                setattr(p4, k, v)
    if args.out_dir:
        p4.out_dir = args.out_dir

    configure_threads(p1.threads)
    print_hardware_report(p1.threads)
    set_seed(42)
    t_start = time.time()

    out = p4.out_dir
    for sub in ("configs", "oracle", "checkpoints", "metrics", "predictions",
                "ablations", "plots"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    cache_dir = os.path.join(out, "checkpoints")
    save_json(os.path.join(out, "configs", "phase4_config.json"), p4.resolved_dict())

    corpora = build_corpora(p1.data_dir, p1.corpus_train_chars,
                            p1.corpus_val_chars, p1.corpus_seed,
                            p1.corpus_val_seed, force=p1.regenerate)
    split_info = {
        "meta_train_seeds": p4.meta_train_seeds,
        "meta_val_seeds": p4.meta_val_seeds,
        "meta_test_seeds": p4.meta_test_seeds,
        "reference_seed": p4.reference_seed,
        "note": "shared deterministic corpus; trajectories differ by init + batch sampling",
    }
    save_json(os.path.join(out, "configs", "split.json"), split_info)

    print("Generating trajectories (K up to 25) ...")
    records = {}
    for split, seeds in (("train", p4.meta_train_seeds),
                         ("val", p4.meta_val_seeds),
                         ("test", p4.meta_test_seeds + [p4.reference_seed])):
        records[split] = {seed: load_or_generate_trajectory(p1, corpora, seed,
                                                            cache_dir, p4)
                          for seed in seeds}

    eval_h = Phase3Eval(p1, corpora, torch.device("cpu"))
    segs = build_segments(records["train"][p4.meta_train_seeds[0]]["w_states"][0])
    n_params = sum(v.numel() for v in
                   records["test"][p4.meta_test_seeds[0]]["w_states"][0].values())
    n_layers = len(segs)
    print(f"  segments/layers={n_layers} params={n_params:,}")

    test_seeds = p4.meta_test_seeds + [p4.reference_seed]

    # ---------------- oracle experiment -------------------------------------
    print("Oracle (best rank-r of ΔW_target) ...")
    oracle_results = {}
    for (K, H) in p4.pairs:
        rows = []
        for rank in p4.oracle_ranks:
            runs = [oracle_result(records["test"][s], K, H, rank, segs, eval_h)
                    for s in test_seeds]
            rows.append({
                "K": K, "H": H, "rank": rank,
                "mean_val_loss": _avg([r["quality"]["val_loss"] for r in runs]),
                "mean_val_ppl": _avg([r["quality"]["val_ppl"] for r in runs]),
                "mean_explained_energy": _avg([r["explained_energy"] for r in runs]),
                "mean_rel_param_dist": _avg([r["rel_param_dist_to_WH"] for r in runs]),
                "per_seed": [r["quality"]["val_loss"] for r in runs],
            })
        oracle_results[(K, H)] = rows
        print(f"  K={K} H={H}: " + ", ".join(
            f"r{r['rank']} energy={r['mean_explained_energy']:.3f} "
            f"val={r['mean_val_loss']:.3f}" for r in rows))
    save_json(os.path.join(out, "oracle", "oracle_results.json"),
              {f"{K}_{H}": rows for (K, H), rows in oracle_results.items()})

    # ---------------- baselines ---------------------------------------------
    baselines = {}
    for (K, H) in p4.pairs:
        no_runs = [no_update_result(records["test"][s], K, eval_h) for s in test_seeds]
        conv_runs = [conventional_result(records["test"][s], K, H) for s in test_seeds]
        p3_runs = [phase3_predictor_result(records["test"][s], K, H, eval_h,
                                           "results/phase3/checkpoints", p1.lr)
                   for s in test_seeds]
        p3_avail = p3_runs[0].get("available", False)
        baselines[(K, H)] = {
            "no_update": summarize(no_runs),
            "conventional": summarize(conv_runs),
            "phase3_predictor": (summarize([r for r in p3_runs if r.get("available", False)])
                                 if p3_avail else {"available": False}),
            "phase3_available": p3_avail,
        }
    save_json(os.path.join(out, "metrics", "baselines.json"),
              {f"{K}_{H}": v for (K, H), v in baselines.items()})

    # ---------------- learned operator --------------------------------------
    print("Meta-training generated low-rank operators ...")
    learned = {}
    meta_train_flops = sum(rec["flops_per_step"] * 100.0
                           for rec in records["train"].values())
    operator_train_total = 0.0

    for (K, H) in p4.pairs:
        X, layer_idx, targets = build_examples(records["train"], p4, p1, K, H, segs)
        in_dim = feature_dim(p4.feature_set, K)
        learned[(K, H)] = {}
        for rank in p4.learned_ranks:
            path = os.path.join(cache_dir, f"operator_K{K}_H{H}_r{rank}.pt")
            if os.path.exists(path):
                operator, hist = load_operator(path, rank, segs)
                wall = 0.0
                cached = True
            else:
                t0 = time.time()
                operator, hist = train_operator(
                    X, layer_idx, targets, segs, rank, in_dim,
                    hidden=p4.hidden, layer_emb_dim=p4.layer_emb_dim,
                    steps=p4.train_steps, lr=p4.lr, batch_size=p4.batch_size,
                    seed=p4.operator_seed)
                wall = time.time() - t0
                save_operator(operator, path, segs, rank, in_dim, p4.hidden,
                              p4.layer_emb_dim, history=hist)
                cached = False
            operator_train_total += operator_train_flops(
                p4.train_steps, X.shape[0], in_dim, p4.hidden,
                p4.layer_emb_dim, total_gen(segs, rank)) if not cached else 0.0
            runs = [learned_result(records["test"][s], K, H, rank, operator, segs,
                                   eval_h, p4.feature_set, p1.lr)
                    for s in test_seeds]
            learned[(K, H)][rank] = {
                "summary": summarize(runs),
                "mean_rel_param_dist": _avg([r["rel_param_dist_to_WH"] for r in runs]),
                "mean_rel_delta_error": _avg([r["rel_delta_error"] for r in runs]),
                "mean_update_norm": _avg([r["update_norm"] for r in runs]),
                "train_wall_sec": wall,
                "cached": cached,
                "history": hist,
                "per_seed": [r["quality"]["val_loss"] for r in runs],
            }
            tag = "cached" if cached else f"{wall:.1f}s"
            print(f"  K={K} H={H} r={rank}: val {learned[(K,H)][rank]['summary']['mean_val_loss']:.3f} "
                  f"rel_err {learned[(K,H)][rank]['mean_rel_delta_error']:.3f} "
                  f"({tag})")

    save_json(os.path.join(out, "metrics", "learned.json"),
              {f"{K}_{H}": {str(r): v for r, v in ranks.items()}
               for (K, H), ranks in learned.items()})

    # ---------------- ablations ---------------------------------------------
    ablations = run_ablations(records, p4, p1, eval_h, segs, cache_dir,
                              corpora, n_params, n_layers)
    save_json(os.path.join(out, "ablations", "ablations.json"),
              {f"{K}_{H}": v for (K, H), v in ablations.items()})

    # ---------------- compute ledger ----------------------------------------
    primary = (10, 25)
    rank_p = p4.default_rank
    gen_p = total_gen(segs, rank_p)
    in_p = feature_dim(p4.feature_set, primary[0])
    c = learned_direct_compute(records["test"][test_seeds[0]], primary[0],
                               primary[1], rank_p, in_p, p4.hidden,
                               p4.layer_emb_dim, gen_p, n_layers, n_params, segs)
    meta_f = meta_train_flops + operator_train_total
    ledger = {
        "meta_training": {
            "trajectory_gen_flops": meta_train_flops,
            "operator_train_flops": operator_train_total,
            "total_meta_flops": meta_f,
        },
        "application": {
            "direct_total_flops": c["direct_total_flops"],
            "conventional_total_flops": c["conventional_total_flops"],
            "components": {k: v for k, v in c.items()},
        },
        "amortization": {},
    }
    for n_app in ("1", "10", "100", "1000"):
        n = int(n_app)
        ledger["amortization"][n_app] = {
            "direct_total": meta_f + n * c["direct_total_flops"],
            "conventional_total": n * c["conventional_total_flops"],
        }
    save_json(os.path.join(out, "metrics", "compute_ledger.json"), ledger)

    # ---------------- plots + report ----------------------------------------
    from src.phase4 import plots as plots_mod
    plot_files = plots_mod.generate_all(baselines, oracle_results, learned,
                                        ablations, p4, out)
    from src.phase4.report import write_report
    report_path = write_report(p4, baselines, oracle_results, learned, ablations,
                               ledger, split_info, out)

    total_sec = time.time() - t_start
    print(f"\nPhase-4 complete in {total_sec:.1f}s -> {os.path.abspath(out)}")
    print("plots:", ", ".join(plot_files))
    print("report:", report_path)
    return {"oracle": oracle_results, "learned": learned, "baselines": baselines,
            "ledger": ledger}


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------
def run_ablations(records, p4: Phase4Config, p1, eval_h: Phase3Eval, segs,
                  cache_dir, corpora, n_params, n_layers) -> dict:
    out = os.path.join(p4.out_dir, "ablations")
    test_seeds = p4.meta_test_seeds + [p4.reference_seed]
    ablations = {}

    for (K, H) in p4.ablation_pairs:
        entry = {"feature_sets": {}, "behavior": {}, "structure": {}}
        # feature sets
        for fs in p4.ablation_feature_sets:
            rank = p4.default_rank
            ab_path = os.path.join(cache_dir, f"abl_{fs}_K{K}_H{H}_r{rank}.pt")
            if os.path.exists(ab_path):
                op, _ = load_operator(ab_path, rank, segs)
            else:
                X, idx, targets = build_examples(records["train"], p4, p1, K, H,
                                                 segs, feature_set=fs)
                in_dim = feature_dim(fs, K)
                op, _ = train_operator(X, idx, targets, segs, rank, in_dim,
                                       hidden=p4.hidden,
                                       layer_emb_dim=p4.layer_emb_dim,
                                       steps=p4.train_steps, lr=p4.lr,
                                       batch_size=p4.batch_size,
                                       seed=p4.operator_seed)
                save_operator(op, ab_path, segs, rank, in_dim, p4.hidden,
                              p4.layer_emb_dim)
            runs = [learned_result(records["test"][s], K, H, rank, op, segs,
                                   eval_h, fs, p1.lr) for s in test_seeds]
            entry["feature_sets"][fs] = {
                "summary": summarize(runs),
                "per_seed": [r["quality"]["val_loss"] for r in runs],
            }
        # behavioural-objective pilot (extra_loss = train-corpus CE of W_pred)
        rank = p4.default_rank
        beh_path = os.path.join(cache_dir, f"abl_behavior_K{K}_H{H}_r{rank}.pt")
        if os.path.exists(beh_path):
            op_b, hist_b = load_operator(beh_path, rank, segs)
        else:
            X, idx, targets = build_examples(records["train"], p4, p1, K, H, segs)
            in_dim = feature_dim(p4.feature_set, K)
            from src.phase4.functional import make_behavior_loss_fn
            train_ids = corpora["train_ids"].to(torch.device("cpu"))
            rec0 = records["train"][p4.meta_train_seeds[0]]
            extra = make_behavior_loss_fn(rec0, K, H, rank, segs, p4.feature_set,
                                          p1.lr, p1, train_ids,
                                          windows=p4.behavior_windows, batch=32)
            op_b, hist_b = train_operator(X, idx, targets, segs, rank, in_dim,
                                          hidden=p4.hidden,
                                          layer_emb_dim=p4.layer_emb_dim,
                                          steps=p4.behavior_steps, lr=p4.lr,
                                          batch_size=p4.batch_size,
                                          seed=p4.operator_seed,
                                          extra_loss_fn=extra,
                                          extra_weight=0.1, extra_frac=0.0)
            save_operator(op_b, beh_path, segs, rank, in_dim, p4.hidden,
                          p4.layer_emb_dim, history=hist_b)
        runs_b = [learned_result(records["test"][s], K, H, rank, op_b, segs,
                                 eval_h, p4.feature_set, p1.lr) for s in test_seeds]
        entry["behavior"] = {"summary": summarize(runs_b),
                             "per_seed": [r["quality"]["val_loss"] for r in runs_b],
                             "weight": 0.1, "steps": p4.behavior_steps}
        ablations[(K, H)] = entry
        print(f"  ablations K={K} H={H}: "
              + ", ".join(f"{fs}={entry['feature_sets'][fs]['summary']['mean_val_loss']:.3f}"
                          for fs in entry["feature_sets"])
              + f" | behavior={entry['behavior']['summary']['mean_val_loss']:.3f}")
    return ablations


if __name__ == "__main__":
    main()