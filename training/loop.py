"""Training orchestrator: alternate self-play and SGD, export ONNX at the
wall-clock milestones (15m / 1h / 4h by default), then exit."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from net import ChessNet, N_BLOCKS, N_FILTERS, n_params
from selfplay import play_batch
from train import ReplayBuffer, train_step
from export import export


def parse_milestones(spec):
    out = []
    for s in spec.split(","):
        s = s.strip()
        if not s:
            continue
        if s.endswith("h"):
            out.append((s, float(s[:-1]) * 3600))
        elif s.endswith("m"):
            out.append((s, float(s[:-1]) * 60))
        elif s.endswith("s"):
            out.append((s, float(s[:-1])))
        else:
            out.append((s, float(s)))
    return sorted(out, key=lambda kv: kv[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--milestones", default="15m,1h,4h",
                    help="Comma-separated wall-clock milestones. Loop exits "
                         "after the last one.")
    ap.add_argument("--games-per-batch", type=int, default=32)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--buffer-capacity", type=int, default=200_000)
    ap.add_argument("--min-buffer-for-train", type=int, default=2_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-steps-per-cycle", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=float, default=30.0, help="seconds")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    net = ChessNet().to(device)
    print(f"net: blocks={N_BLOCKS} filters={N_FILTERS} params={n_params(net):,}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buf = ReplayBuffer(args.buffer_capacity)
    milestones = parse_milestones(args.milestones)
    print(f"milestones: {milestones}", flush=True)

    start = time.time()
    last_log = start
    next_milestone = 0
    total_games = 0
    total_train_steps = 0
    rng = np.random.default_rng(args.seed)

    def elapsed():
        return time.time() - start

    def save_checkpoint(tag):
        ckpt_path = out_dir / f"ckpt_{tag}.pt"
        torch.save(
            {
                "weights": net.state_dict(),
                "config": {"n_blocks": N_BLOCKS, "n_filters": N_FILTERS},
                "stats": {
                    "elapsed_sec": elapsed(),
                    "games": total_games,
                    "train_steps": total_train_steps,
                    "buffer_size": len(buf),
                },
            },
            ckpt_path,
        )
        onnx_path = out_dir / f"ckpt_{tag}.onnx"
        net.eval()
        export(str(ckpt_path), str(onnx_path))
        net.train()
        print(f"[{tag}] saved {ckpt_path} and {onnx_path}", flush=True)

    net.train()
    while True:
        if next_milestone >= len(milestones):
            print("all milestones hit, exiting", flush=True)
            break
        tag, deadline = milestones[next_milestone]
        if elapsed() >= deadline:
            save_checkpoint(tag)
            next_milestone += 1
            continue

        cycle_start = time.time()
        results, stats = play_batch(
            net, device, args.games_per_batch, args.sims, rng=rng
        )
        buf.add_many(results)
        total_games += stats["games"]
        cycle_sp_time = time.time() - cycle_start

        train_loss = None
        if len(buf) >= args.min_buffer_for_train:
            train_start = time.time()
            losses = []
            for _ in range(args.train_steps_per_cycle):
                batch = buf.sample(args.batch_size, rng=rng)
                if batch is None:
                    break
                losses.append(train_step(net, opt, batch, device))
                total_train_steps += 1
            cycle_tr_time = time.time() - train_start
            if losses:
                train_loss = {
                    "loss": float(np.mean([l["loss"] for l in losses])),
                    "policy": float(np.mean([l["policy_loss"] for l in losses])),
                    "value": float(np.mean([l["value_loss"] for l in losses])),
                }
        else:
            cycle_tr_time = 0.0

        if time.time() - last_log >= args.log_every:
            log = {
                "t": round(elapsed(), 1),
                "games": total_games,
                "buf": len(buf),
                "tr_steps": total_train_steps,
                "sp_sec": round(cycle_sp_time, 2),
                "tr_sec": round(cycle_tr_time, 2),
                "avg_plies": round(stats["avg_plies"], 1),
                "loss": train_loss,
                "next_ms": milestones[next_milestone][0],
            }
            print(json.dumps(log), flush=True)
            last_log = time.time()


if __name__ == "__main__":
    main()
