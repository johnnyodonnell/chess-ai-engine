"""Isolated games/sec A/B for the rust_pipeline self-play engine.

Spawns run_selfplay_pipeline exactly like orchestrator.py, drains the output
queue, and reports completed games/sec over a measurement window (after a warmup
window that lets torch.compile trace and the pipeline reach steady state).

Reads CHESS_COMPILE / CHESS_BF16 from the environment (inherited by the spawned
child). Run once per arm:
  CHESS_COMPILE=1 CHESS_BF16=0 python bench_games_sec.py --weights ... (baseline)
  CHESS_COMPILE=1 CHESS_BF16=1 python bench_games_sec.py --weights ... (treatment)
"""

import argparse
import multiprocessing as mp
import os
import queue as _queue
import time

from selfplay import run_selfplay_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--bucket", type=int, default=512)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--run", type=float, default=420.0)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    print(
        f"A/B arm: CHESS_COMPILE={os.environ.get('CHESS_COMPILE', '0')}"
        f" CHESS_BF16={os.environ.get('CHESS_BF16', '0')}"
        f" bucket={args.bucket} threads={args.threads} sims={args.sims}",
        flush=True,
    )
    print(f"run={args.run}s interval={args.interval}s", flush=True)

    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=512)
    stop = ctx.Event()
    p = ctx.Process(
        target=run_selfplay_pipeline,
        args=(q, stop, args.seed, args.sims, args.weights,
              args.threads, args.bucket, "cuda"),
    )
    p.start()

    interval = args.interval
    try:
        start = time.time()
        deadline = start + args.run
        win_games = win_plies = 0
        win_start = start
        while time.time() < deadline:
            try:
                rows, n_games, avg_plies = q.get(timeout=0.5)
                win_games += n_games
                win_plies += len(rows)
            except _queue.Empty:
                pass
            now = time.time()
            if now - win_start >= interval:
                dt = now - win_start
                print(
                    f"[+{now - start:5.0f}s] {win_games / dt:6.3f} games/sec  "
                    f"({win_plies / max(1, win_games):5.1f} avg plies, "
                    f"{win_plies / dt:5.0f} rows/sec)",
                    flush=True,
                )
                win_games = win_plies = 0
                win_start = now
    finally:
        stop.set()
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    print("DONE (read steady-state from the last few intervals)", flush=True)


if __name__ == "__main__":
    main()
