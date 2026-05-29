"""Training orchestrator: alternate self-play and SGD.

Two modes:
  - Legacy milestone mode (--milestones): exits after all milestones hit.
  - Indefinite mode (--snapshot-every, default): runs forever, saving an
    immutable timestamped snapshot every N hours of cumulative training time.
    Resumes automatically from latest.pt on restart (pm2-safe).
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from net import ChessNet, N_BLOCKS, N_FILTERS, n_params
from selfplay import play_batch
from train import ReplayBuffer, train_step
from export import export


def parse_duration(spec):
    """Parse a single duration string like '12h', '30m', '300s' into seconds."""
    s = spec.strip()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    elif s.endswith("m"):
        return float(s[:-1]) * 60
    elif s.endswith("s"):
        return float(s[:-1])
    return float(s)


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
    ap.add_argument("--milestones", default=None,
                    help="Legacy: comma-separated wall-clock milestones. "
                         "Loop exits after the last one. Mutually exclusive "
                         "with --snapshot-every.")
    ap.add_argument("--snapshot-every", default="4h",
                    help="Indefinite mode: save a snapshot every this interval "
                         "of cumulative training time (e.g. 4h, 30m). "
                         "Ignored when --milestones is set.")
    ap.add_argument("--init-from", default=None,
                    help="Warm-start a fresh run: load weights+optimizer from "
                         "this checkpoint but keep the clock/counters/snapshot "
                         "schedule at zero. Only used on a cold start (when no "
                         "latest.pt exists in --out-dir).")
    ap.add_argument("--save-latest-every", default="300s",
                    help="How often to write latest.pt (weights + optimizer + "
                         "RNG + elapsed). Default 300s. Used for crash recovery.")
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

    snapshot_interval = parse_duration(args.snapshot_every)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    net = ChessNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buf = ReplayBuffer(args.buffer_capacity)
    rng = np.random.default_rng(args.seed)

    # --- Resume from latest.pt if present ---
    base_elapsed = 0.0
    total_games = 0
    total_train_steps = 0
    next_snapshot_at = 0.0  # cumulative seconds until next snapshot (indefinite mode)
    loaded_interval = None  # snapshot interval recorded in the resumed checkpoint

    latest_path = out_dir / "latest.pt"
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["weights"])
        opt.load_state_dict(ckpt["opt"])
        base_elapsed = ckpt.get("elapsed_sec", 0.0)
        total_games = ckpt.get("games", 0)
        total_train_steps = ckpt.get("train_steps", 0)
        next_snapshot_at = ckpt.get("next_snapshot_at", 0.0)
        loaded_interval = ckpt.get("snapshot_interval")
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"].cpu())
        if "np_rng" in ckpt:
            rng.bit_generator.state = ckpt["np_rng"]
        print(f"resumed from {latest_path} "
              f"(elapsed={base_elapsed/3600:.2f}h games={total_games} "
              f"steps={total_train_steps})", flush=True)
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["weights"])
        if "opt" in ckpt:
            try:
                opt.load_state_dict(ckpt["opt"])
            except (ValueError, KeyError) as e:
                print(f"warn: skipped optimizer state from --init-from ({e})", flush=True)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"warm start from {args.init_from} "
              f"(weights+optimizer, fresh clock; seed={args.seed})", flush=True)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"cold start (seed={args.seed})", flush=True)

    print(f"net: blocks={N_BLOCKS} filters={N_FILTERS} params={n_params(net):,}", flush=True)

    start = time.time()
    last_log = start
    last_latest_save = start

    def elapsed():
        return base_elapsed + (time.time() - start)

    def _atomic_save(path, obj):
        tmp = path.with_suffix(".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

    def save_latest():
        _atomic_save(latest_path, {
            "weights": net.state_dict(),
            "opt": opt.state_dict(),
            "elapsed_sec": elapsed(),
            "games": total_games,
            "train_steps": total_train_steps,
            "next_snapshot_at": next_snapshot_at,
            "snapshot_interval": snapshot_interval,
            "torch_rng": torch.get_rng_state(),
            "np_rng": rng.bit_generator.state,
            "config": {"n_blocks": N_BLOCKS, "n_filters": N_FILTERS},
        })

    def save_snapshot():
        snap_dir = out_dir / "snapshots"
        snap_dir.mkdir(exist_ok=True)
        hours = int(elapsed() / 3600)
        utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%MZ")
        stem = f"snap_h{hours:05d}_{utc}"
        pt_path = snap_dir / f"{stem}.pt"
        onnx_path = snap_dir / f"{stem}.onnx"
        _atomic_save(pt_path, {
            "weights": net.state_dict(),
            "config": {"n_blocks": N_BLOCKS, "n_filters": N_FILTERS},
            "stats": {
                "elapsed_sec": elapsed(),
                "games": total_games,
                "train_steps": total_train_steps,
                "buffer_size": len(buf),
            },
        })
        net.eval()
        export(str(pt_path), str(onnx_path))
        net.train()
        print(f"[snapshot] {pt_path.name}  {onnx_path.name}", flush=True)
        return pt_path

    def _pid_alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def launch_eval(pt_path):
        """Fire evaluate.py as a detached subprocess. Overlap-guarded by
        eval.lock so a hung eval never spawns a second one."""
        lock = out_dir / "eval.lock"
        if lock.exists():
            try:
                prev_pid = int(lock.read_text().split()[0])
                if _pid_alive(prev_pid):
                    print(f"[eval] prior eval (pid {prev_pid}) still running; "
                          f"skipping {pt_path.name}", flush=True)
                    return
            except (ValueError, IndexError, OSError):
                pass  # stale/garbled lock — overwrite below
        training_dir = Path(__file__).resolve().parent
        logf = open(out_dir / "eval.log", "a")
        try:
            proc = subprocess.Popen(
                [sys.executable, "evaluate.py",
                 "--run-dir", str(out_dir), "--candidate", str(pt_path)],
                cwd=str(training_dir), stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            logf.close()
        lock.write_text(f"{proc.pid} {time.time()}")
        print(f"[eval] launched evaluate.py pid={proc.pid} for {pt_path.name}",
              flush=True)

    # -------------------------------------------------------------------------
    # Legacy milestone mode
    # -------------------------------------------------------------------------
    if args.milestones is not None:
        milestones = parse_milestones(args.milestones)
        print(f"milestone mode: {milestones}", flush=True)
        next_milestone = 0

        def save_checkpoint(tag):
            ckpt_path = out_dir / f"ckpt_{tag}.pt"
            _atomic_save(ckpt_path, {
                "weights": net.state_dict(),
                "config": {"n_blocks": N_BLOCKS, "n_filters": N_FILTERS},
                "stats": {
                    "elapsed_sec": elapsed(),
                    "games": total_games,
                    "train_steps": total_train_steps,
                    "buffer_size": len(buf),
                },
            })
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
        return

    # -------------------------------------------------------------------------
    # Indefinite mode (default)
    # -------------------------------------------------------------------------
    save_latest_interval = parse_duration(args.save_latest_every)

    # If the snapshot interval changed across a restart, realign to the new grid
    # so the new cadence takes effect immediately rather than honoring a stale
    # next_snapshot_at computed under the old interval.
    if loaded_interval is not None and abs(loaded_interval - snapshot_interval) > 1e-6:
        next_snapshot_at = (int(elapsed() // snapshot_interval) + 1) * snapshot_interval
        print(f"snapshot interval changed {loaded_interval}s -> {snapshot_interval}s; "
              f"realigned next snapshot to elapsed={next_snapshot_at/3600:.2f}h", flush=True)

    # Advance next_snapshot_at past already-elapsed intervals (e.g. after resume)
    while next_snapshot_at <= elapsed():
        next_snapshot_at += snapshot_interval

    print(f"indefinite mode: snapshot every {args.snapshot_every} "
          f"(next at elapsed={next_snapshot_at/3600:.2f}h)", flush=True)

    net.train()
    while True:
        # Fire snapshot if due
        if elapsed() >= next_snapshot_at:
            pt_path = save_snapshot()
            save_latest()
            last_latest_save = time.time()
            next_snapshot_at += snapshot_interval
            launch_eval(pt_path)
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

        # Periodically save latest.pt for crash recovery
        if time.time() - last_latest_save >= save_latest_interval:
            save_latest()
            last_latest_save = time.time()

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
                "next_snap_in": round(next_snapshot_at - elapsed(), 1),
            }
            print(json.dumps(log), flush=True)
            last_log = time.time()


if __name__ == "__main__":
    main()
