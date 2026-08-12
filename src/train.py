"""Phase 1 baseline training: conventional AdamW on a tiny decoder-only LM.

Usage:
    python -m src.train --config configs/baseline.yaml

CLI flags override the YAML config for the most common knobs. Everything is
CPU-first: threads are constrained, no CUDA, small memory footprint, and each
optimizer step is fully deterministic given a seed so runs can be resumed.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import resource
import time
from typing import Optional

import torch
from torch.optim import AdamW

from src.checkpoint import (load_resume, load_trajectory_step_last,
                            save_resume, save_trajectory_step)
from src.dataset import build_corpora, make_window_batch
from src.evaluate import evaluate_loss, grad_and_param_norms
from src.model import Transformer
from src.utils import (Config, configure_threads, count_parameters,
                       estimate_flops, load_config, print_hardware_report,
                       resolve_device, save_json, set_seed)


def build_scheduler(optimizer, cfg: Config):
    if cfg.lr_schedule == "cosine":
        def lr_fn(step: int) -> float:
            if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
                return step / cfg.warmup_steps
            progress = (step - cfg.warmup_steps) / max(1, cfg.n_steps - cfg.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def build_model(cfg: Config, vocab_size: int) -> Transformer:
    from src.model import ModelConfig

    mcfg = ModelConfig(
        vocab_size=vocab_size,
        n_layer=cfg.n_layer,
        d_model=cfg.d_model,
        n_head=cfg.n_head,
        context_length=cfg.context_length,
        ffn_mult=cfg.ffn_mult,
        dropout=cfg.dropout,
        tie_embeddings=cfg.tie_embeddings,
    )
    return Transformer(mcfg)


class TrainingLog:
    HEADER = [
        "step", "phase", "epochs_seen", "tokens_seen",
        "train_loss", "train_loss_ema",
        "val_loss", "val_ppl",
        "grad_norm", "param_norm", "lr",
        "elapsed_sec", "tokens_per_sec", "approx_flops_total",
    ]

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.file = open(path, "w", encoding="utf-8", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.HEADER)
        self.file.flush()

    def write_row(self, row: dict) -> None:
        self.writer.writerow([row.get(k, "") for k in self.HEADER])
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 baseline training")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--grad-accumulation", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main(argv: Optional[list] = None) -> dict:
    t_start = time.time()
    args = vars(parse_args() if argv is None else parse_args(argv))

    cfg = load_config(args["config"])
    for k, v in args.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.regenerate = cfg.regenerate or args.get("regenerate", False)
    cfg.resume = cfg.resume or args.get("resume", False)

    configure_threads(cfg.threads)
    print_hardware_report(cfg.threads)

    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    print(f"\nConfig: {cfg.experiment_name} | steps={cfg.n_steps} "
          f"lr={cfg.lr} wd={cfg.weight_decay} batch={cfg.batch_size} "
          f"accum={cfg.grad_accumulation} seed={cfg.seed} device={device}")
    print(f"Trajectory steps: {cfg.trajectory_steps}\n")

    corpora = build_corpora(
        cfg.data_dir, cfg.corpus_train_chars, cfg.corpus_val_chars,
        cfg.corpus_seed, cfg.corpus_val_seed, force=cfg.regenerate,
    )
    train_ids = corpora["train_ids"].to(device)
    val_ids = corpora["val_ids"].to(device)
    vocab_size = corpora["vocab_size"]

    model = build_model(cfg, vocab_size).to(device)
    total_params, trainable = count_parameters(model)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                      betas=tuple(cfg.betas))
    scheduler = build_scheduler(optimizer, cfg)

    eff_batch = cfg.batch_size * cfg.grad_accumulation
    tokens_per_step = eff_batch * cfg.context_length
    train_tokens = int(train_ids.numel())
    epochs_total = tokens_per_step * cfg.n_steps / train_tokens
    flops = estimate_flops(model, cfg.context_length, cfg.n_steps, eff_batch)

    os.makedirs(cfg.out_dir, exist_ok=True)
    trajectory_dir = os.path.join(cfg.out_dir, "trajectory")
    os.makedirs(trajectory_dir, exist_ok=True)
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    save_json(os.path.join(cfg.out_dir, "config.json"), cfg.resolved_dict(vocab_size))
    resume_path = os.path.join(cfg.checkpoints_dir, "resume_latest.pt")

    print(f"Corpus: train={train_tokens:,} tokens, val={corpora['n_val_tokens']:,} tokens, "
          f"vocab={vocab_size} | params {total_params:,} (trainable {trainable:,})")
    print(f"Token exposure: ~{epochs_total:.1f} epochs | "
          f"approx_flops_total={flops['approx_flops_total']:.3e}")

    step = 0
    if cfg.resume and os.path.exists(resume_path):
        info = load_resume(resume_path, model, optimizer, scheduler)
        step = info["step"]
        print(f"Resumed from step {step} ({resume_path})")

    model.train()
    log = TrainingLog(os.path.join(cfg.out_dir, "training.log"))
    ema_loss: Optional[float] = None
    tokens_seen = 0
    trajectory_metrics: list = []
    peak_rss_gib = 0.0

    for step in range(step, cfg.n_steps + 1):
        step_loss = 0.0
        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            for mb in range(cfg.grad_accumulation):
                gi = (step - 1) * cfg.grad_accumulation + mb
                x, y = make_window_batch(train_ids, cfg.context_length, gi,
                                         cfg.seed, cfg.batch_size)
                logits = model(x)
                loss = model.loss(logits, y)
                (loss / cfg.grad_accumulation).backward()
                step_loss += float(loss.item())
            step_loss /= cfg.grad_accumulation
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            tokens_seen = tokens_per_step * step
            ema_loss = step_loss if ema_loss is None else ema_loss * 0.9 + step_loss * 0.1
        else:
            val0 = evaluate_loss(model, val_ids, cfg.context_length,
                                 batch_size=cfg.batch_size,
                                 max_windows=cfg.eval_max_windows)
            step_loss = val0["loss"]
            ema_loss = val0["loss"]

        norms = grad_and_param_norms(model)
        is_trajectory = step in cfg.trajectory_steps
        elapsed = time.time() - t_start

        val_loss = None
        val_ppl = None
        metrics = None
        if is_trajectory:
            val = evaluate_loss(model, val_ids, cfg.context_length,
                                batch_size=cfg.batch_size,
                                max_windows=cfg.eval_max_windows)
            val_loss = val["loss"]
            val_ppl = val["ppl"]
            metrics = {
                "step": step,
                "train_loss": round(step_loss, 6),
                "train_loss_ema": round(ema_loss or 0.0, 6),
                "val_loss": round(val_loss, 6),
                "val_ppl": round(val_ppl, 4),
                "grad_norm": round(norms["grad_norm"], 6) if norms["grad_norm"] is not None else None,
                "param_norm": round(norms["param_norm"], 4),
                "lr": float(scheduler.get_last_lr()[0]),
                "elapsed_sec": round(elapsed, 2),
                "tokens_seen": tokens_seen,
                "approx_flops_so_far": round(flops["approx_flops_per_step"] * (step + 1), 2),
            }
            trajectory_metrics.append(metrics)
            save_trajectory_step(trajectory_dir, step, model, metrics)
            save_resume(resume_path, model, optimizer, scheduler, step, cfg.seed,
                        extra={"metrics": metrics})
            print(f"  [trajectory] step {step:>3}: train={step_loss:.4f} "
                  f"ema={ema_loss:.4f} val={val_loss:.4f} ppl={val_ppl:.3f} "
                  f"grad_norm={metrics['grad_norm']}")

        peak_rss_gib = max(peak_rss_gib,
                           resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0)
        log.write_row({
            "step": step,
            "phase": "trajectory" if is_trajectory else ("init" if step == 0 else "train"),
            "epochs_seen": tokens_seen / max(1, train_tokens),
            "tokens_seen": tokens_seen,
            "train_loss": round(step_loss, 6),
            "train_loss_ema": round(ema_loss or 0.0, 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else "",
            "val_ppl": round(val_ppl, 4) if val_ppl is not None else "",
            "grad_norm": round(norms["grad_norm"], 6) if norms["grad_norm"] is not None else "",
            "param_norm": round(norms["param_norm"], 4),
            "lr": float(scheduler.get_last_lr()[0]),
            "elapsed_sec": round(elapsed, 2),
            "tokens_per_sec": round(tokens_seen / max(1e-9, elapsed), 1),
            "approx_flops_total": round(flops["approx_flops_total"], 3),
        })

        if step > 0 and not is_trajectory and step % cfg.log_every == 0:
            print(f"  step {step:>3}: loss={step_loss:.4f} ema={ema_loss:.4f} "
                  f"grad_norm={norms['grad_norm'] and round(norms['grad_norm'], 4)}")

    log.close()

    final_val = evaluate_loss(model, val_ids, cfg.context_length,
                              batch_size=cfg.batch_size,
                              max_windows=cfg.eval_max_windows)
    total_elapsed = time.time() - t_start
    first = trajectory_metrics[0]
    last = trajectory_metrics[-1]

    summary = {
        "experiment": cfg.experiment_name,
        "param_count": total_params,
        "trainable_params": trainable,
        "n_train_tokens": train_tokens,
        "n_val_tokens": int(val_ids.numel()),
        "vocab_size": vocab_size,
        "epochs": round(epochs_total, 2),
        "steps": cfg.n_steps,
        "threads": cfg.threads,
        "device": str(device),
        "initial_train_loss": first["train_loss"],
        "initial_val_loss": first["val_loss"],
        "initial_val_ppl": first["val_ppl"],
        "final_train_loss": last["train_loss"],
        "final_val_loss": final_val["loss"],
        "final_val_ppl": final_val["ppl"],
        "training_time_sec": round(total_elapsed, 2),
        "tokens_per_sec": round(tokens_seen / max(1e-9, total_elapsed), 1),
        "peak_rss_gib": round(peak_rss_gib, 3),
        "approx_flops_total": flops["approx_flops_total"],
        "flops_estimate_method": flops["estimate_method"],
        "trajectory_steps": cfg.trajectory_steps,
    }
    import glob

    cp_files = glob.glob(os.path.join(trajectory_dir, "*.pt"))
    sizes = [os.path.getsize(f) for f in cp_files]
    if sizes:
        summary["trajectory_checkpoint_bytes"] = sizes
        summary["max_checkpoint_size_bytes"] = int(max(sizes))

    save_json(os.path.join(cfg.out_dir, "metrics.json"), {
        "config": cfg.resolved_dict(vocab_size),
        "hardware": {
            "cpu_cores": os.cpu_count(),
            "threads": cfg.threads,
            "gpu": "cuda" if torch.cuda.is_available() else "none",
        },
        "flops": flops,
        "summary": summary,
        "trajectory": trajectory_metrics,
    })

    print("\n" + "=" * 70)
    print("BASELINE COMPLETE")
    print(f"  parameters:          {total_params:,}")
    print(f"  dataset (tokens):    train={train_tokens:,} val={int(val_ids.numel()):,}")
    print(f"  threads:             {cfg.threads}")
    print(f"  steps:               {cfg.n_steps}")
    print(f"  initial train loss:  {first['train_loss']:.4f}   val: {first['val_loss']:.4f} (ppl {first['val_ppl']:.2f})")
    print(f"  final   train loss:  {last['train_loss']:.4f}   val: {final_val['loss']:.4f} (ppl {final_val['ppl']:.2f})")
    print(f"  time:                {total_elapsed:.1f} s ({summary['tokens_per_sec']:.0f} tok/s)")
    print(f"  peak RSS:            {peak_rss_gib:.2f} GiB")
    print(f"  approx FLOPs:        {flops['approx_flops_total']:.3e}")
    print(f"  results:             {os.path.abspath(cfg.out_dir)}")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    main()