"""Parallel self-play orchestrator (Path B).

Runs the trainer in this (main) process and spawns ONE in-process self-play
process running a native engine that holds the net on the GPU and does its own
batched in-process forwards — no separate inference server, no shared-memory
channels. Select the engine via CHESS_BACKEND:
  - rust_pipeline : decoupled non-blocking pipeline (workers never block on the
                    GPU; two inference buckets feed one consumer).
  - rust_async    : bulk-synchronous multi-threaded engine (A/B baseline).

The self-play process pushes finished (state, pi, z) tuples onto a bounded queue;
the trainer drains that queue into a ring ReplayBuffer, runs SGD, and periodically
publishes fresh weights to a file the self-play side reloads on mtime change.
Snapshots / latest.pt / detached eval are unchanged from loop.py (same checkpoint
format → resumes the run1 checkpoint).

IMPORTANT: module-level imports are kept torch-free. `spawn` re-imports this
module in every child (to set up __main__), so importing torch here would pull
torch into the CPU-only self-play process. All torch imports live inside main().
"""

import argparse
import datetime
import json
import multiprocessing as mp
import os
import queue as pyqueue
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from chess_backend import USE_ASYNC_ENGINE, USE_PIPELINE_ENGINE
from selfplay import run_selfplay_async, run_selfplay_pipeline


def parse_duration(spec):
    """Parse a duration string like '12h', '30m', '300s' into seconds."""
    s = str(spec).strip()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--init-from", default=None,
                    help="Warm-start a fresh run from this checkpoint (weights "
                         "[+optimizer], fresh clock). Only on cold start.")
    ap.add_argument("--snapshot-every", default="4h")
    ap.add_argument("--save-latest-every", default="300s")
    ap.add_argument("--publish-every", default="15s",
                    help="How often the trainer republishes weights for the "
                         "self-play process to reload.")
    # self-play workers
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--games-per-worker", type=int, default=16)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--queue-max", type=int, default=64)
    # rust_async knobs (CPU parallelism is decoupled from batch width)
    ap.add_argument("--async-n-games", type=int, default=None,
                    help="rust_async: total game slots in flight (NN batch-width "
                         "ceiling). Defaults to workers*games-per-worker.")
    ap.add_argument("--async-n-threads", type=int, default=None,
                    help="rust_async: MCTS worker threads. Defaults to cpu_count-2.")
    ap.add_argument("--async-max-batch", type=int, default=None,
                    help="rust_async: max rows per GPU forward. Defaults to "
                         "async-n-games.")
    # rust_pipeline knobs (decoupled non-blocking engine; one batch-width knob)
    ap.add_argument("--pipeline-bucket-size", type=int, default=512,
                    help="rust_pipeline: GPU batch width per forward (the single "
                         "tuning knob; in-flight games self-regulate to ~2x this).")
    ap.add_argument("--pipeline-n-threads", type=int, default=16,
                    help="rust_pipeline: MCTS worker threads.")
    # training
    ap.add_argument("--buffer-capacity", type=int, default=200_000)
    ap.add_argument("--min-buffer-for-train", type=int, default=2_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--target-reuse", type=float, default=4.0,
                    help="Avg times each generated sample is used in training. "
                         "Paces SGD to the self-play data rate so the trainer "
                         "doesn't starve self-play of GPU, and bounds "
                         "overfitting. 4 follows KataGo's tuned cap (<=4/row); "
                         "AlphaZero used ~0.5 (compute-rich), Leela ~10 (flagged "
                         "as over-sampling). loop.py's implicit ratio was ~6.7.")
    ap.add_argument("--max-steps-per-cycle", type=int, default=64,
                    help="Cap on SGD steps per loop iteration (smooths bursts).")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=float, default=30.0)
    return ap.parse_args()


def main():
    args = parse_args()

    # torch-heavy imports live here so `spawn`-reimported workers stay torch-free.
    import torch
    from net import ChessNet, N_BLOCKS, N_FILTERS, n_params
    from train import ReplayBuffer, train_step
    from export import export

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_interval = parse_duration(args.snapshot_every)
    save_latest_interval = parse_duration(args.save_latest_every)
    publish_interval = parse_duration(args.publish_every)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    net = ChessNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    buf = ReplayBuffer(args.buffer_capacity)
    rng = np.random.default_rng(args.seed)

    # ----- Resume from latest.pt, else warm-start from --init-from, else cold -----
    base_elapsed = 0.0
    total_games = 0
    total_train_steps = 0
    next_snapshot_at = 0.0
    loaded_interval = None
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
        print(f"resumed from {latest_path} (elapsed={base_elapsed/3600:.2f}h "
              f"games={total_games} steps={total_train_steps})", flush=True)
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["weights"])
        if "opt" in ckpt:
            try:
                opt.load_state_dict(ckpt["opt"])
            except (ValueError, KeyError) as e:
                print(f"warn: skipped optimizer state from --init-from ({e})",
                      flush=True)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"warm start from {args.init_from} (fresh clock; seed={args.seed})",
              flush=True)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"cold start (seed={args.seed})", flush=True)

    print(f"net: blocks={N_BLOCKS} filters={N_FILTERS} params={n_params(net):,}",
          flush=True)

    # ----- Checkpoint / snapshot / eval helpers (mirror loop.py) -----
    def _atomic_save(path, obj):
        tmp = path.with_suffix(".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

    serving_path = out_dir / "serving_weights.pt"

    def publish_weights():
        _atomic_save(serving_path, {
            "weights": net.state_dict(),
            "config": {"n_blocks": N_BLOCKS, "n_filters": N_FILTERS},
        })

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
            "stats": {"elapsed_sec": elapsed(), "games": total_games,
                      "train_steps": total_train_steps, "buffer_size": len(buf)},
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
        # A finished-but-unreaped child lingers as a zombie whose PID-table
        # entry keeps os.kill(pid, 0) succeeding even though it has exited.
        # Treat zombies as dead, otherwise a stale lock blocks every future
        # eval forever.
        try:
            with open(f"/proc/{pid}/stat") as f:
                data = f.read()
            state = data[data.rindex(")") + 1:].split()[0]
            if state == "Z":
                return False
        except (OSError, ValueError, IndexError):
            pass
        return True

    def launch_eval(pt_path):
        # Reap the previous eval if it has finished, so it doesn't linger as a
        # zombie (parent never exits) and trip the staleness check below.
        prev = launch_eval.proc
        if prev is not None and prev.poll() is not None:
            launch_eval.proc = None
        lock = out_dir / "eval.lock"
        if lock.exists():
            try:
                prev_pid = int(lock.read_text().split()[0])
                if _pid_alive(prev_pid):
                    print(f"[eval] prior eval (pid {prev_pid}) still running; "
                          f"skipping {pt_path.name}", flush=True)
                    return
            except (ValueError, IndexError, OSError):
                pass
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
        launch_eval.proc = proc
        lock.write_text(f"{proc.pid} {time.time()}")
        print(f"[eval] launched evaluate.py pid={proc.pid} for {pt_path.name}",
              flush=True)

    launch_eval.proc = None

    # ----- Spawn the in-process self-play engine -----
    publish_weights()  # initial: the net source the self-play side loads/reloads

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue(maxsize=args.queue_max)
    stop = ctx.Event()

    if USE_PIPELINE_ENGINE:
        # rust_pipeline: ONE self-play process runs the decoupled non-blocking
        # engine — workers never block on the GPU; two inference buckets feed one
        # consumer. Dynamic game count; bucket size is the only batch-width knob.
        def start_selfplay():
            p = ctx.Process(
                target=run_selfplay_pipeline,
                args=(out_queue, stop, args.seed + 1000, args.sims,
                      str(serving_path), args.pipeline_n_threads,
                      args.pipeline_bucket_size, str(device)),
            )
            p.start()
            return p

        print(f"spawned 1 pipeline self-play process "
              f"(threads={args.pipeline_n_threads} "
              f"bucket={args.pipeline_bucket_size} sims={args.sims}); "
              f"no inference server", flush=True)
    elif USE_ASYNC_ENGINE:
        # rust_async: ONE self-play process runs the native multi-threaded engine
        # (n_threads MCTS workers + 1 batching consumer doing in-process forwards)
        # — no inference server, no shm. CPU parallelism (n_threads) is decoupled
        # from batch width (n_games), so we can keep the GPU fed AND use all cores.
        n_games = args.async_n_games or (args.workers * args.games_per_worker)
        n_threads = args.async_n_threads or max(1, (os.cpu_count() or 2) - 2)
        max_batch = args.async_max_batch or n_games

        def start_selfplay():
            p = ctx.Process(
                target=run_selfplay_async,
                args=(out_queue, stop, args.seed + 1000, n_games, args.sims,
                      str(serving_path), n_threads, max_batch, str(device)),
            )
            p.start()
            return p

        print(f"spawned 1 async self-play process (games={n_games} "
              f"threads={n_threads} max_batch={max_batch} sims={args.sims}); "
              f"no inference server", flush=True)
    else:
        raise SystemExit(
            "orchestrator requires an in-process self-play backend; set "
            "CHESS_BACKEND=rust_pipeline (or rust_async)."
        )

    selfplay_proc = start_selfplay()

    # ----- Clock / snapshot schedule (mirror loop.py) -----
    start = time.time()
    last_log = start
    last_latest_save = start
    last_publish = start

    def elapsed():
        return base_elapsed + (time.time() - start)

    if loaded_interval is not None and abs(loaded_interval - snapshot_interval) > 1e-6:
        next_snapshot_at = (int(elapsed() // snapshot_interval) + 1) * snapshot_interval
        print(f"snapshot interval changed; realigned next snapshot to "
              f"elapsed={next_snapshot_at/3600:.2f}h", flush=True)
    while next_snapshot_at <= elapsed():
        next_snapshot_at += snapshot_interval
    print(f"snapshot every {args.snapshot_every} "
          f"(next at elapsed={next_snapshot_at/3600:.2f}h)", flush=True)

    last_avg_plies = 0.0
    last_loss = None      # most recent training loss (paced SGD logs sparsely)
    gen_samples = 0       # tuples generated this session (for reuse pacing)
    session_steps = 0     # SGD steps taken this session (for reuse pacing)
    net.train()
    try:
        while True:
            # Supervise the self-play process — restart if it died.
            if not selfplay_proc.is_alive():
                print("warn: self-play process died; restarting", flush=True)
                selfplay_proc = start_selfplay()

            # Snapshot if due.
            if elapsed() >= next_snapshot_at:
                pt_path = save_snapshot()
                save_latest()
                last_latest_save = time.time()
                next_snapshot_at += snapshot_interval
                launch_eval(pt_path)
                continue

            # Drain finished games from workers into the ring buffer (bounded).
            got_any = False
            for _ in range(args.workers * 2):
                try:
                    results, n_games, avg_plies = out_queue.get_nowait()
                except pyqueue.Empty:
                    break
                buf.add_many(results)
                total_games += n_games
                gen_samples += len(results)
                last_avg_plies = avg_plies
                got_any = True
            # Block briefly for data while the buffer is still filling.
            if len(buf) < args.min_buffer_for_train and not got_any:
                try:
                    results, n_games, avg_plies = out_queue.get(timeout=1.0)
                    buf.add_many(results)
                    total_games += n_games
                    gen_samples += len(results)
                    last_avg_plies = avg_plies
                except pyqueue.Empty:
                    pass

            # SGD, paced to the data rate: keep cumulative samples-trained near
            # target_reuse x samples-generated. This stops the trainer from
            # hogging the GPU self-play needs, and bounds overfitting.
            train_loss = None
            cycle_tr_time = 0.0
            allowed = int(gen_samples * args.target_reuse / args.batch_size) - session_steps
            n_steps = max(0, min(allowed, args.max_steps_per_cycle))
            if len(buf) >= args.min_buffer_for_train and n_steps > 0:
                train_start = time.time()
                losses = []
                for _ in range(n_steps):
                    batch = buf.sample(args.batch_size, rng=rng)
                    if batch is None:
                        break
                    losses.append(train_step(net, opt, batch, device))
                    total_train_steps += 1
                    session_steps += 1
                cycle_tr_time = time.time() - train_start
                if losses:
                    train_loss = {
                        "loss": float(np.mean([l["loss"] for l in losses])),
                        "policy": float(np.mean([l["policy_loss"] for l in losses])),
                        "value": float(np.mean([l["value_loss"] for l in losses])),
                    }
                    last_loss = train_loss
            elif not got_any:
                # Caught up (paced out) with no new data — avoid hot-spinning
                # the trainer core; stay responsive to incoming games.
                time.sleep(0.05)

            now = time.time()
            if now - last_publish >= publish_interval:
                publish_weights()
                last_publish = now
            if now - last_latest_save >= save_latest_interval:
                save_latest()
                last_latest_save = now
            if now - last_log >= args.log_every:
                print(json.dumps({
                    "t": round(elapsed(), 1),
                    "games": total_games,
                    "buf": len(buf),
                    "tr_steps": total_train_steps,
                    "reuse": round(session_steps * args.batch_size / max(1, gen_samples), 2),
                    "qsize": out_queue.qsize(),
                    "tr_sec": round(cycle_tr_time, 2),
                    "avg_plies": round(last_avg_plies, 1),
                    "loss": last_loss,
                    "next_snap_in": round(next_snapshot_at - elapsed(), 1),
                }), flush=True)
                last_log = now
    finally:
        print("shutting down: signaling self-play", flush=True)
        stop.set()
        selfplay_proc.join(timeout=10)
        if selfplay_proc.is_alive():
            selfplay_proc.terminate()


if __name__ == "__main__":
    main()
