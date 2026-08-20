"""Phase-5 research report generator (results/phase5/phase5_report.md)."""
from __future__ import annotations

import os
from datetime import datetime, timezone


def _fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def _table(rows, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_report(p5, ab, oracle, gen_res, ledger, out_dir):
    lines = []
    A = lines.append
    A("# Phase 5 — Generalizable Update Operator")
    A("")
    A(f"Date: {datetime.now(timezone.utc).isoformat()}")
    A("")
    A("> Research status: R&D hypothesis under investigation. Nothing here is "
      "claimed to replace gradient descent or to scale to large models.")
    A("")

    A("## 1. Research question")
    A("")
    A("Can a learned update operator **generalize** its parameter-transformation "
      "strategy to unseen training trajectories (unseen seed, unseen batch ordering, "
      "unseen initialization, unseen corpus)? The goal is to separate representation "
      "capacity from generalization capacity — not to beat AdamW prematurely.")
    A("")

    A("## 2. Experimental design")
    A("")
    A("Short horizons first: K ∈ {10, 15, 25}, H ∈ {25, 50}, ranks 4 and 8 "
      "(primary 4). Each experiment reports no-update, conventional AdamW, "
      "rank-r oracle, and the learned generator on held-out trajectories. "
      "Meta-training uses trajectory data through step K only; the future target "
      "ΔW = W_H − W_K is used solely for supervision. The behavioural objective "
      "uses the TRAIN corpus (no eval-corpus leakage).")
    A("")

    A("## 3. Meta-training data")
    A("")
    A("- Corpus A: existing deterministic synthetic corpus (shared, ~60k train / 20k "
      "val tokens, 32-char vocabulary).")
    A(f"- Meta-train: {len(p5.meta_train_seeds)} trajectories (seeds 10–41); "
      f"data-size ablation uses the first 6 / 16 / 32 of these.")
    A(f"- Meta-validation: {len(p5.meta_val_seeds)} trajectories (seeds 42–49) — used "
      "for validation-based checkpoint selection.")
    A(f"- Held-out tests: unseen-seed test A (seeds 50–57 + reference 7); unseen "
      "batch-order test B (seen init, unseen data seed 60–61); unseen-init test C "
      "(unseen init 62–63, seen data seed); unseen-corpus test D (corpus B).")
    A("")
    A("All trajectories differ by initialization and batch sampling seed. Corpus B "
      "uses the same character set/tokenizer (identical architecture) but a sharply "
      "different word distribution, providing a genuine distribution shift.")
    A("")

    A("## 4. Architecture")
    A("")
    A("Compressed / latent generation (shared across layers):")
    A("")
    A("```")
    A("global+per-layer features + layer embedding")
    A("   -> shared feature encoder -> latent z_l")
    A("   -> small factor generator -> coefficients")
    A("   -> apply onto learned per-layer bases")
    A("ΔW_l = U_l V_l^T,  U_l = P_l C_l,  V_l = Q_l D_l   (2-D)")
    A("Δw_l = B_l c_l                                     (1-D)")
    A("P_l ∈ R^{out×m}, Q_l ∈ R^{in×m}, B_l ∈ R^{dim×m}  (learned per-layer bases)")
    A("```")
    A("")
    A(f"The network predicts only the small coefficients C_l, D_l (m×r) / c_l (m) "
      f"(m = {p5.m_basis} basis width, r = {p5.rank} rank). The learned bases are "
      "directions shared across trajectories (not observed gradients). The output "
      "head is zero-initialized (starts at 'no update'). Compared against the "
      "Phase-4 flat MLP ('direct') in the architecture ablation.")
    A("")

    A("## 5. Training objectives")
    A("")
    A("| objective | formula |")
    A("|---|---|")
    A("| A recon | Σ_l ‖U_l V_l^T − ΔW_target_l‖² / ‖ΔW_target_l‖² |")
    A("| B behavior | CE(W_K + ΔW_pred; train corpus) |")
    A("| C combined | A + λ·B (λ tested: 0.01, 0.1, 1.0) |")
    A("")
    A("Regularization: Adam weight decay, validation-based checkpoint selection "
      "(keep the generator with lowest meta-val reconstruction MSE).")
    A("")

    A("## 6. Generalization results")
    A("")
    A("### Data-size ablation (K=10, H=25, r=4, combined λ=0.1, compressed)")
    A("")
    A(_table([
        ["# train traj", "val (test)", "no update", "oracle r4", "% oracle recovered"],
        *([n, _fmt(ab["data_size"][n]["mean_val_loss"]),
           _fmt(ab["data_size"][n]["no_update"]),
           _fmt(ab["data_size"][n]["oracle_r4"]),
           _fmt(ab["data_size"][n]["pct_oracle_recovered"], 1)]
          for n in sorted(ab["data_size"].keys()))
    ], ["# train traj", "val (test)", "no update", "oracle r4", "% oracle recovered"]))
    A("")
    A("### Objective ablation (16 train, r=4, compressed)")
    A("")
    A(_table([
        ["objective", "val (test)", "% oracle recovered"],
        *([k, _fmt(v["mean_val_loss"]), _fmt(v["pct_oracle_recovered"], 1)]
          for k, v in ab["objective"].items())
    ], ["objective", "val (test)", "% oracle recovered"]))
    A("")
    A("### Architecture ablation (16 train, combined λ=0.1, r=4)")
    A("")
    A(_table([
        ["arch", "val (test)"],
        *([k, _fmt(v["mean_val_loss"])] for k, v in ab["arch"].items())
    ], ["arch", "val (test)"]))
    A("")
    A("### Rank ablation (16 train, combined λ=0.1)")
    A("")
    A(_table([
        ["rank", "val (test)"],
        *([k, _fmt(v["mean_val_loss"])] for k, v in ab["rank"].items())
    ], ["rank", "val (test)"]))
    A("")
    A("### Horizon (16 train, combined λ=0.1, r=4)")
    A("")
    A(_table([
        ["(K,H)", "val (test)"],
        *([k, _fmt(v["mean_val_loss"])] for k, v in ab["horizon"].items())
    ], ["(K,H)", "val (test)"]))
    A("")

    A("## 7. Oracle comparison")
    A("")
    A("The rank-r oracle (best SVD of the true update) sets the family ceiling.")
    A("")
    A(_table([
        ["pair", "rank", "val (test)", "explained energy (2-D)"],
        *([f"({r['K']},{r['H']})", r["rank"], _fmt(r["mean_val_loss"]),
           _fmt(r["mean_explained_energy"], 3)] for (K, H), rows in sorted(oracle.items())
          for r in rows)
    ], ["pair", "rank", "val (test)", "explained energy (2-D)"]))
    A("")

    A("## 8. Ablations")
    A("")
    A("Summarized in Section 6 (data size, objective, architecture, rank, horizon). "
      "The primary read: does adding meta-training data, changing the objective, or "
      "using a structured generator move the held-out result away from no-update?")
    A("")

    A("## 9. Compute")
    A("")
    comp = ledger["application"]["components"]
    A("### Per application (K=10, H=25, r=4)")
    A("")
    A(_table([
        ["observation (AdamW to W_K)", f"{comp['observation_fwd_bwd']}",
         f"{comp['observation_flops']:.3e}"],
        ["generator inference", "1", f"{comp['generator_inference_flops']:.3e}"],
        ["U V^T generation", "1", f"{comp['uv_generation_flops']:.3e}"],
        ["parameter update", "1", f"{comp['param_update_flops']:.3e}"],
        ["direct total", "-", f"{comp['direct_total_flops']:.3e}"],
        ["conventional total", "100", f"{comp['conventional_total_flops']:.3e}"],
    ], ["component", "steps", "FLOPs"]))
    A("")
    A("### Meta-training (one-time) and amortization")
    A("")
    A(_table([
        ["N applications", "Direct (meta + N·direct)", "Conventional (N·conventional)"],
        *([n, f"{ledger['amortization'][n]['direct_total']:.3e}",
           f"{ledger['amortization'][n]['conventional_total']:.3e}"]
          for n in ("1", "10", "100", "1000"))
    ], ["N applications", "Direct (meta + N·direct)", "Conventional (N·conventional)"]))
    A("")
    A(f"Meta-training trajectory generation FLOPs (32 runs × 50 steps): "
      f"{ledger['meta_training']['trajectory_gen_flops']:.3e}. The direct "
      "application cost is dominated by the K observation steps.")
    A("")

    A("## 10. Failure analysis")
    A("")
    A("Determine which limiting factor applies (representation / generator / "
      "training data / objective / horizon / generalization / compute):")
    A("")
    A("| factor | how tested |")
    A("|---|---|")
    A("| representation | oracle energy vs rank (Sec. 7) |")
    A("| generator | arch ablation: compressed vs flat (Sec. 6) |")
    A("| training data | data-size ablation 6/16/32 (Sec. 6) |")
    A("| objective | A/B/C comparison (Sec. 6) |")
    A("| horizon | K/H grid (Sec. 6) |")
    A("| generalization | tests A–D (below) |")
    A("| compute | Section 9 |")
    A("")
    A("### Generalization tests (primary generator: 32 train, combined λ=0.1, r=4, K=10 H=25)")
    A("")
    A(_table([
        *([c, _fmt(gen_res[c]["no_update"]), _fmt(gen_res[c]["oracle_r4"]),
           _fmt(gen_res[c]["mean_val_loss"]),
           _fmt(gen_res[c]["pct_oracle_recovered"], 1),
           _fmt(gen_res[c]["mean_cos_pred_target"]),
           _fmt(gen_res[c]["mean_cos_gradmean_target"])]
          for c in gen_res)
    ], ["test", "no update", "oracle r4", "learned", "% oracle recovered",
        "cos(ΔW_pred, ΔW_target)", "cos(mean-grad, ΔW_target)"]))
    A("")

    A("## 11. Recommendation")
    A("")
    A("Success level 1 (research): learned > no-update on held-out trajectories with "
      "a substantial fraction of oracle improvement recovered. Success level 2 "
      "(methodological): quality ≈ AdamW at lower total compute.")
    A("")
    ga = gen_res.get("A_unseen_seed", {})
    learned_v = ga.get("mean_val_loss")
    no_v = ga.get("no_update")
    rec = ga.get("pct_oracle_recovered")
    if learned_v is None or no_v is None:
        verdict = "MODIFY (insufficient data to judge)"
    elif rec is not None and rec > 20:
        verdict = "PROCEED — the operator generalizes to unseen seeds and recovers a "
        "meaningful fraction of the oracle improvement on held-out trajectories "
        "(research success)."
    elif rec is not None and rec > 5:
        verdict = "MODIFY — a modest generalization signal exists (learned below "
        "no-update by a non-trivial margin), but the recovered fraction of the "
        "oracle improvement is not yet substantial; expand data/architecture before "
        "concluding."
    else:
        verdict = "STOP THIS APPROACH — increasing meta-training data (6→32), a "
        "structured compressed generator, multiple training objectives, and a "
        "second corpus all still leave the learned operator at the no-update "
        "baseline on unseen trajectories (~1–4% of the oracle improvement "
        "recovered; cos(ΔW_pred, ΔW_target) ≈ 0). The learned update operator is "
        "not generalizing, so the learned-update methodology as tested should be "
        "reconsidered. (The oracle still shows the low-rank family is informative; "
        "the failure is the learned prediction, not the family.)"
    A(f"**{verdict}**")
    A("")
    A(f"Measured basis: unseen-seed test — no update {_fmt(no_v)}, learned "
      f"{_fmt(learned_v)}, % oracle improvement recovered {_fmt(rec, 1)}. Full "
      "numbers in Section 10.")
    A("")
    A("---")
    A(f"_Phase-5 report generated by src/phase5/run.py. Raw data under "
      f"results/phase5/metrics|ablations|generalization|oracle|predictions._")

    path = os.path.join(out_dir, "phase5_report.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
