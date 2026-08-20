"""Phase-5 research driver.

Usage:
    python -m src.phase5.run --phase1-config configs/baseline.yaml

Flow (CPU-only, ≤4 threads):
  1. build corpus A (+ tokenizer) and a second corpus B,
  2. generate/cache trajectories (train/val/test + decoupled + corpus B),
  3. oracle: best rank-r of the true update,
  4. train the structured generator under data-size / objective / arch / rank
     ablations and multiple horizons,
  5. generalization tests A-D on the primary generator,
  6. compute accounting + plots + results/phase5/phase5_report.md.
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from src.dataset import build_corpora
from src.utils import (configure_threads, load_config, print_hardware_report,
                       save_json, set_seed)
from src.phase3.evaluate import Phase3Eval
from src.phase5.config import Phase5Config
from src.phase5.features import feature_dim
from src.phase5.trajectory import load_or_generate
from src.phase5.generator import train_generator, generate_deltas
from src.phase5 import evaluate as ev


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-5 Generalizable Update Operator")
    p.add_argument("--phase1-config", default="configs/baseline.yaml")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--threads", type=int, default=None)
    return p.parse_args(argv)


def build_examples(train_records, p5, p1, K, H, segs, feature_set=None):
    from src.phase4.features import build_features
    layers = [s["name"] for s in segs]
    fs = feature_set or p5.feature_set
    X, idx, targets = [], [], []
    for rec in train_records.values():
        for i, name in enumerate(layers):
            X.append(build_features(rec, K, name, fs, p1.lr))
            idx.append(i)
            targets.append(rec["w_states"][H][name].float()
                           - rec["w_states"][K][name].float())
    return torch.stack(X), torch.tensor(idx, dtype=torch.long), targets


def _avg(seq):
    vals = [v for v in seq if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def summarize(runs, key="quality"):
    return {"mean_val_loss": _avg([r[key]["val_loss"] for r in runs]),
            "mean_val_ppl": _avg([r[key]["val_ppl"] for r in runs]),
            "mean_cos_pred_target": _avg([r.get("cos_pred_target") for r in runs]),
            "mean_cos_gradmean_target": _avg([r.get("cos_gradmean_target") for r in runs]),
            "mean_rel_delta_error": _avg([r.get("rel_delta_error") for r in runs]),
            "mean_update_norm": _avg([r.get("update_norm") for r in runs]),
            "n": len(runs)}


def save_gen(gen, path, p5, rank, in_dim, objective, lambda_b, history=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": gen.state_dict(), "rank": rank, "in_dim": in_dim,
                "hidden": getattr(gen, "hidden", p5.hidden),
                "latent_dim": getattr(gen, "latent_dim", p5.latent_dim),
                "m_basis": getattr(gen, "m_basis", p5.m_basis),
                "layer_emb_dim": getattr(gen, "layer_emb_dim", p5.layer_emb_dim),
                "objective": objective, "lambda_b": lambda_b,
                "history": history or []}, path)


def load_gen(path, p5, segs, rank, in_dim):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    from src.phase5.generator import StructuredGenerator
    g = StructuredGenerator(in_dim, payload["hidden"], payload["latent_dim"],
                            payload["layer_emb_dim"], payload["m_basis"], rank,
                            segs, torch.zeros(in_dim), torch.ones(in_dim))
    g.load_state_dict(payload["state"])
    return g, payload.get("history") or []


def main(argv=None) -> dict:
    args = parse_args() if argv is None else parse_args(argv)
    p1 = load_config(args.phase1_config)
    if args.threads:
        p1.threads = args.threads
    p5 = Phase5Config()
    if args.out_dir:
        p5.out_dir = args.out_dir

    configure_threads(p1.threads)
    print_hardware_report(p1.threads)
    set_seed(42)
    t_start = time.time()

    out = p5.out_dir
    for sub in ("configs", "checkpoints", "metrics", "predictions",
                "generalization", "ablations", "plots"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    cache = os.path.join(out, "checkpoints")
    save_json(os.path.join(out, "configs", "phase5_config.json"), p5.resolved_dict())

    # ---------------- corpora A and B ---------------------------------------
    corpora_a = build_corpora(p1.data_dir, p1.corpus_train_chars,
                              p1.corpus_val_chars, p1.corpus_seed,
                              p1.corpus_val_seed, force=p1.regenerate)
    from src.phase5.corpus2 import build_corpus_b
    b_dir = os.path.join(out, "configs", "corpus_b")
    corpora_b = build_corpus_b(b_dir, corpora_a["tokenizer"],
                               30000, 10000, 999, 888, force=True)

    # ---------------- trajectories -------------------------------------------
    print("Generating trajectories ...")
    rec = {}
    train16 = p5.meta_train_seeds[:16]
    train32 = p5.meta_train_seeds[:32]
    rec["train"] = {s: load_or_generate(p1, corpora_a, s, cache, p5)
                    for s in train32}
    rec["val"] = {s: load_or_generate(p1, corpora_a, s, cache, p5)
                  for s in p5.meta_val_seeds}
    rec["test"] = {s: load_or_generate(p1, corpora_a, s, cache, p5)
                   for s in p5.meta_test_seeds}
    rec["ref"] = load_or_generate(p1, corpora_a, p5.reference_seed, cache, p5)
    # decoupled generalization tests
    rec["test_batch_order"] = {
        d: load_or_generate(p1, corpora_a, d, cache, p5,
                            init_seed=train16[0], data_seed=d)
        for d in p5.unseen_data_seeds}
    rec["test_init"] = {
        i: load_or_generate(p1, corpora_a, i, cache, p5,
                            init_seed=i, data_seed=train16[1])
        for i in p5.unseen_init_seeds}
    # corpus B trajectories
    rec["test_corpus_b"] = {
        s: load_or_generate(p1, corpora_b, s, cache, p5) for s in p5.corpus_b_test_seeds}

    eval_a = Phase3Eval(p1, corpora_a, torch.device("cpu"))
    eval_b = Phase3Eval(p1, corpora_b, torch.device("cpu"))
    segs = [{"name": n, "out": int(t.shape[0]),
             "in": int(t.shape[1]) if t.ndim == 2 else int(t.numel()),
             "ndim": 2 if t.ndim == 2 else 1}
            for n, t in sorted(rec["train"][train32[0]]["w_states"][0].items())]
    n_params = sum(v.numel() for v in rec["train"][train32[0]]["w_states"][0].values())
    n_layers = len(segs)
    print(f"  layers={n_layers} params={n_params:,} "
          f"train={len(train32)} val={len(rec['val'])} "
          f"test_A={len(rec['test'])+1} corpus_B={len(rec['test_corpus_b'])}")

    # ---------------- oracle -------------------------------------------------
    print("Oracle ...")
    oracle = {}
    for (K, H) in p5.pairs:
        rows = []
        for rank in p5.oracle_ranks:
            runs = [ev.oracle_lowrank_result(rec["test"][s], K, H, rank, segs, eval_a)
                    for s in p5.meta_test_seeds]
            runs.append(ev.oracle_lowrank_result(rec["ref"], K, H, rank, segs, eval_a))
            rows.append({"K": K, "H": H, "rank": rank,
                         "mean_val_loss": _avg([r["quality"]["val_loss"] for r in runs]),
                         "mean_explained_energy": _avg([r["explained_energy"] for r in runs]),
                         "mean_rel_param_dist": _avg([r["rel_param_dist_to_WH"] for r in runs])})
        oracle[(K, H)] = rows
        print(f"  K{K} H{H}: " + ", ".join(f"r{r['rank']} en={r['mean_explained_energy']:.2f} "
                                           f"val={r['mean_val_loss']:.3f}" for r in rows))
    save_json(os.path.join(out, "oracle", "oracle_results.json"),
              {f"{K}_{H}": rows for (K, H), rows in oracle.items()})

    # ---------------- primary generator --------------------------------------
    # primary config: 32 train, combined λ=0.1, rank 4, compressed, (10,25)
    primary_key = "s32_c0.1_r4"
    gen, ghist, _ = train_or_load(
        primary_key, rec["train"], rec["val"], p5, p1, (10, 25), 4,
        "combined", 0.1, "compressed", eval_a, segs, cache,
        train_ids=corpora_a["train_ids"])
    print(f"Primary generator: val {gen_val(gen, rec, (10,25), 4, p5, p1, eval_a, segs)}")

    # ---------------- ablations ------------------------------------------------
    ab = run_ablations(rec, p5, p1, eval_a, eval_b, segs, cache,
                       train_ids=corpora_a["train_ids"])

    # ---------------- generalization (tests A-D) --------------------------------
    gen_res = generalization_eval(gen, rec, p5, p1, eval_a, eval_b, segs)
    save_json(os.path.join(out, "generalization", "generalization.json"), gen_res)

    # ---------------- baselines ------------------------------------------------
    bases = {}
    (K, H) = (10, 25)
    for name, rset in [("train", rec["train"]), ("val", rec["val"]),
                       ("test", rec["test"]), ("ref", {"7": rec["ref"]})]:
        runs = [ev.no_update_result(r, K, eval_a) for r in rset.values()]
        runs2 = [ev.conventional_result(r, K, H) for r in rset.values()]
        bases[name] = {"no_update": summarize(runs), "conventional": summarize(runs2)}
    save_json(os.path.join(out, "metrics", "baselines.json"),
              {k: v for k, v in bases.items()})

    # ---------------- compute ledger ------------------------------------------
    meta_f = sum(r["flops_per_step"] * 50.0 for r in rec["train"].values())
    c = ev.generator_compute(rec["test"][p5.meta_test_seeds[0]], 10, 25, 4,
                             feature_dim(p5.feature_set, 10), p5.hidden,
                             p5.latent_dim, p5.layer_emb_dim, 2 * p5.m_basis * 4,
                             n_layers, n_params, segs)
    ledger = {"meta_training": {"trajectory_gen_flops": meta_f,
                                "generator_train_flops": 0.0,
                                "total_meta_flops": meta_f},
              "application": {"direct_total_flops": c["direct_total_flops"],
                              "conventional_total_flops": c["conventional_total_flops"],
                              "components": {k: v for k, v in c.items()}},
              "amortization": {}}
    for n in ("1", "10", "100", "1000"):
        ledger["amortization"][n] = {
            "direct_total": meta_f + int(n) * c["direct_total_flops"],
            "conventional_total": int(n) * c["conventional_total_flops"]}
    save_json(os.path.join(out, "metrics", "compute_ledger.json"), ledger)

    # ---------------- plots + report -------------------------------------------
    from src.phase5 import plots as plots_mod
    plots_mod.generate_all(ab, oracle, gen_res, p5, out)
    from src.phase5.report import write_report
    write_report(p5, ab, oracle, gen_res, ledger, out)

    print(f"\nPhase-5 complete in {time.time()-t_start:.1f}s -> {os.path.abspath(out)}")


# ---------------------------------------------------------------------------
def train_or_load(name, train_records, val_records, p5, p1, pair, rank,
                  objective, lambda_b, arch, eval_h, segs, cache, train_ids=None):
    """Train a generator (or load cached), returning it."""
    K, H = pair
    in_dim = feature_dim(p5.feature_set, K)
    path = os.path.join(cache, f"gen_{name}.pt")

    if arch == "direct":
        from src.phase4.operator import UpdateOperator, train_operator
        if os.path.exists(path):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            op = UpdateOperator(in_dim, payload["hidden"], payload["layer_emb_dim"],
                                len(segs), payload["total_gen"],
                                torch.zeros(in_dim), torch.ones(in_dim))
            op.load_state_dict(payload["state"])
            return op, payload.get("history") or [], None
        X, idx, targets = build_examples(train_records, p5, p1, K, H, segs)
        op, hist = train_operator(X, idx, targets, segs, rank, in_dim,
                                  hidden=p5.hidden, layer_emb_dim=p5.layer_emb_dim,
                                  steps=p5.train_steps, lr=p5.lr,
                                  batch_size=p5.batch_size, seed=p5.seed)
        from src.phase4.operator import total_gen
        torch.save({"state": op.state_dict(), "rank": rank, "in_dim": in_dim,
                    "hidden": p5.hidden, "layer_emb_dim": p5.layer_emb_dim,
                    "total_gen": total_gen(segs, rank),
                    "history": hist}, path)
        return op, hist, None

    if os.path.exists(path):
        gen, hist = load_gen(path, p5, segs, rank, in_dim)
        return gen, hist, None
    X, idx, targets = build_examples(train_records, p5, p1, K, H, segs)
    Xv, idxv, tv = build_examples(val_records, p5, p1, K, H, segs)
    behavior = None
    if objective in ("behavior", "combined") and lambda_b > 0 and train_ids is not None:
        behavior = {"records": list(train_records.values()), "K": K,
                    "feature_set": p5.feature_set, "lr": p1.lr,
                    "phase1_cfg": p1, "windows": p5.behavior_windows,
                    "train_ids": train_ids}
    gen, hist, best_val = train_generator(
        X, idx, targets, segs, rank, in_dim,
        hidden=p5.hidden, latent_dim=p5.latent_dim,
        layer_emb_dim=p5.layer_emb_dim, m_basis=p5.m_basis,
        steps=p5.train_steps, lr=p5.lr, weight_decay=p5.weight_decay,
        batch_size=p5.batch_size, seed=p5.seed, objective=objective,
        lambda_b=lambda_b, behavior=behavior, update_norm_reg=p5.update_norm_reg,
        val_X=Xv, val_idx=idxv, val_targets=tv)
    save_gen(gen, path, p5, rank, in_dim, objective, lambda_b, history=hist)
    return gen, hist, best_val


def gen_val(gen, rec, pair, rank, p5, p1, eval_h, segs):
    K, H = pair
    runs = [ev.learned_result(gen, rec["test"][s], K, H, rank, segs, eval_h,
                              p5.feature_set, p1.lr) for s in p5.meta_test_seeds]
    return summarize(runs)["mean_val_loss"]


def run_ablations(rec, p5, p1, eval_a, eval_b, segs, cache, train_ids=None):
    ab = {"data_size": {}, "objective": {}, "arch": {}, "rank": {}, "horizon": {}}
    (K, H) = (10, 25)

    # data-size ablation
    for n in p5.data_size_ablation:
        tr = {s: rec["train"][s] for s in p5.meta_train_seeds[:n]}
        name = f"s{n}_c0.1_r4"
        g, hist, bv = train_or_load(name, tr, rec["val"], p5, p1, (K, H), 4,
                                    "combined", 0.1, "compressed", eval_a, segs,
                                    cache, train_ids=train_ids)
        runs = [ev.learned_result(g, rec["test"][s], K, H, 4, segs, eval_a,
                                  p5.feature_set, p1.lr) for s in p5.meta_test_seeds]
        no = _avg([ev.no_update_result(rec["test"][s], K, eval_a)["quality"]["val_loss"]
                   for s in p5.meta_test_seeds])
        orc = _avg([ev.oracle_lowrank_result(rec["test"][s], K, H, 4, segs, eval_a)["quality"]["val_loss"]
                    for s in p5.meta_test_seeds])
        ab["data_size"][n] = {**summarize(runs), "no_update": no, "oracle_r4": orc,
                              "pct_oracle_recovered": ev.pct_oracle_recovered(
                                  no, summarize(runs)["mean_val_loss"], orc),
                              "train_rel_mse": (hist[-1]["train_rel_mse"] if hist else None)}
        print(f"  data_size {n}: val {ab['data_size'][n]['mean_val_loss']:.3f} "
              f"pct_rec {ab['data_size'][n]['pct_oracle_recovered']:.1f}")

    # objective ablation (16 train)
    tr16 = {s: rec["train"][s] for s in p5.meta_train_seeds[:16]}
    for (obj, lb) in p5.objective_ablation:
        name = f"obj_{obj}_{lb}_r4"
        g, hist, bv = train_or_load(name, tr16, rec["val"], p5, p1, (K, H), 4,
                                    obj, lb, "compressed", eval_a, segs, cache, train_ids=train_ids)
        runs = [ev.learned_result(g, rec["test"][s], K, H, 4, segs, eval_a,
                                  p5.feature_set, p1.lr) for s in p5.meta_test_seeds]
        no = _avg([ev.no_update_result(rec["test"][s], K, eval_a)["quality"]["val_loss"]
                   for s in p5.meta_test_seeds])
        orc = _avg([ev.oracle_lowrank_result(rec["test"][s], K, H, 4, segs, eval_a)["quality"]["val_loss"]
                    for s in p5.meta_test_seeds])
        ab["objective"][f"{obj}_{lb}"] = {**summarize(runs),
                                          "pct_oracle_recovered": ev.pct_oracle_recovered(
                                              no, summarize(runs)["mean_val_loss"], orc)}
        print(f"  objective {obj} λ={lb}: val {ab['objective'][f'{obj}_{lb}']['mean_val_loss']:.3f} "
              f"pct {ab['objective'][f'{obj}_{lb}']['pct_oracle_recovered']:.1f}")

    # architecture ablation (16 train, combined 0.1)
    for arch in ("compressed", "direct"):
        name = f"arch_{arch}_r4"
        g, hist, bv = train_or_load(name, tr16, rec["val"], p5, p1, (K, H), 4,
                                    "combined", 0.1, arch, eval_a, segs, cache, train_ids=train_ids)
        runs = [ev.learned_result(g, rec["test"][s], K, H, 4, segs, eval_a,
                                  p5.feature_set, p1.lr) for s in p5.meta_test_seeds]
        ab["arch"][arch] = summarize(runs)
        print(f"  arch {arch}: val {summarize(runs)['mean_val_loss']:.3f}")

    # rank ablation (16 train, combined 0.1)
    for rank in p5.learned_ranks:
        name = f"rank_r{rank}"
        g, hist, bv = train_or_load(name, tr16, rec["val"], p5, p1, (K, H), rank,
                                    "combined", 0.1, "compressed", eval_a, segs, cache, train_ids=train_ids)
        runs = [ev.learned_result(g, rec["test"][s], K, H, rank, segs, eval_a,
                                  p5.feature_set, p1.lr) for s in p5.meta_test_seeds]
        ab["rank"][rank] = summarize(runs)
        print(f"  rank {rank}: val {summarize(runs)['mean_val_loss']:.3f}")

    # horizon (16 train, combined 0.1, rank 4)
    for (k, h) in p5.pairs:
        name = f"hor_K{k}_H{h}"
        g, hist, bv = train_or_load(name, tr16, rec["val"], p5, p1, (k, h), 4,
                                    "combined", 0.1, "compressed", eval_a, segs,
                                    cache, train_ids=train_ids)
        runs = [ev.learned_result(g, rec["test"][s], k, h, 4, segs, eval_a,
                                  p5.feature_set, p1.lr) for s in p5.meta_test_seeds]
        ab["horizon"][f"{k}_{h}"] = summarize(runs)
        print(f"  horizon K{k} H{h}: val {summarize(runs)['mean_val_loss']:.3f}")

    save_json(os.path.join(p5.out_dir, "ablations", "ablations.json"),
              {k: {str(kk): v for kk, v in v.items()}
               for k, v in ab.items()})
    return ab


def generalization_eval(gen, rec, p5, p1, eval_a, eval_b, segs):
    """Evaluate the primary generator on tests A (seed), B (batch order),
    C (init), D (corpus)."""
    K, H = (10, 25)
    rank = 4
    out = {}
    sets = {
        "A_unseen_seed": [(s, rec["test"][s], eval_a) for s in p5.meta_test_seeds],
        "A_ref_seed": [("7", rec["ref"], eval_a)],
        "B_unseen_batch_order": [(str(d), r, eval_a) for d, r in rec["test_batch_order"].items()],
        "C_unseen_init": [(str(i), r, eval_a) for i, r in rec["test_init"].items()],
        "D_unseen_corpus": [(str(s), r, eval_b) for s, r in rec["test_corpus_b"].items()],
    }
    for name, items in sets.items():
        learned, no_up, conv, orc = [], [], [], []
        for _, r, eh in items:
            learned.append(ev.learned_result(gen, r, K, H, rank, segs, eh,
                                             p5.feature_set, p1.lr))
            no_up.append(ev.no_update_result(r, K, eh)["quality"]["val_loss"])
            conv.append(ev.conventional_result(r, K, H)["quality"]["val_loss"])
            orc.append(ev.oracle_lowrank_result(r, K, H, rank, segs, eh)["quality"]["val_loss"])
        no, co, orcv = _avg(no_up), _avg(conv), _avg(orc)
        out[name] = {**summarize(learned), "no_update": no, "conventional": co,
                     "oracle_r4": orcv,
                     "pct_oracle_recovered": ev.pct_oracle_recovered(
                         no, summarize(learned)["mean_val_loss"], orcv)}
        print(f"  gen test {name}: no_up {no:.3f} conv {co:.3f} orc {orcv:.3f} "
              f"learned {out[name]['mean_val_loss']:.3f} "
              f"pct_rec {out[name]['pct_oracle_recovered']:.1f} "
              f"cos_pred {out[name]['mean_cos_pred_target']:.3f}")
    return out


if __name__ == "__main__":
    main()
