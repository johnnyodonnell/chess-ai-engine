"""Strict-synchronous REINFORCE self-play / train orchestrator.

Loop (no overlap -> zero GPU contention):
  1. run the Rust worker (current weights) -> a cohort file
  2. ONE full-batch REINFORCE SGD step over the cohort (sgd_batch = n)
  3. publish new weights (serving_weights.safetensors) for the next self-play
  4. save latest checkpoint (resume)
  5. every --eval-every: snapshot (safetensors + onnx) + pool-Elo evaluation

Resumes from <out-dir>/latest.pt. Runs indefinitely until killed.
"""

import argparse
import datetime
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
# Shared AlphaZero modules (encode.py, net.py, export.py) live in ../training.
# Append (not insert) so our local train.py/cohort.py win over the AlphaZero
# trainer's same-named modules there.
sys.path.append(str(HERE.parent / "training"))

from net import ChessNet, N_BLOCKS, N_FILTERS, n_params  # noqa: E402
from export import export_module  # noqa: E402

from cohort import read_cohort  # noqa: E402
from train import train_on_cohort  # noqa: E402
from weights_io import save_weights_st  # noqa: E402

SELFPLAY_BIN = HERE / "selfplay_rs" / "target" / "release" / "selfplay_reinforce"
EVAL_BIN = HERE / "selfplay_rs" / "target" / "release" / "evaluate_reinforce"

# A cohort takes seconds; a late ack means the worker is wedged, not slow.
ACK_TIMEOUT_SEC = 600


def parse_duration(spec: str) -> float:
    s = str(spec).strip()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


def worker_env() -> dict:
    """libtorch + CUDA libs on LD_LIBRARY_PATH for the Rust subprocess."""
    import glob

    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    sp = os.path.dirname(os.path.dirname(torch.__file__))  # site-packages
    libs = [torch_lib] + sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(libs) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def read_ack(worker, timeout, what):
    ready, _, _ = select.select([worker.stdout], [], [], timeout)
    if not ready:
        raise RuntimeError(f"selfplay worker {what} timeout ({timeout:.0f}s)")
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError("selfplay worker died")
    return line


def start_selfplay_worker(env):
    if not SELFPLAY_BIN.exists():
        raise SystemExit(f"self-play binary not found: {SELFPLAY_BIN}\n"
                         f"build it: cd selfplay_rs && cargo build --release")
    p = subprocess.Popen(
        [str(SELFPLAY_BIN), "selfplay-serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        cwd=str(HERE), env=env, text=True, bufsize=1,
    )
    line = read_ack(p, ACK_TIMEOUT_SEC, "ready")
    if not json.loads(line).get("ready"):
        raise RuntimeError(f"selfplay worker failed to start (got {line!r})")
    return p


def run_cohort(worker, **cmd):
    worker.stdin.write(json.dumps(cmd) + "\n")
    worker.stdin.flush()
    line = read_ack(worker, ACK_TIMEOUT_SEC, "ack")
    ack = json.loads(line)
    if not ack.get("done"):
        raise RuntimeError(f"selfplay worker bad ack: {line!r}")
    return ack


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/run1")
    ap.add_argument("--eval-every", default="30m")
    # Cohort size = rows per cohort = rows in the single SGD step. Push this as
    # high as one full-batch step fits in GPU memory (see README tuning notes).
    ap.add_argument("--cohort-rows", type=int, default=8192, help="decision rows per cohort")
    ap.add_argument("--selfplay-batch", type=int, default=512, help="max GPU forward width")
    ap.add_argument("--selfplay-threads", type=int, default=16, help="CPU game-worker threads")
    ap.add_argument("--selfplay-concurrency", type=int, default=0,
                    help="games kept in flight (0 = 2x batch); overlaps CPU game logic with the GPU forward")
    ap.add_argument("--max-plies", type=int, default=200, help="ply cap; over it a game is a draw")
    ap.add_argument("--micro-batch", type=int, default=32768,
                    help="rows per forward in the accumulated SGD step (bounds GPU memory; "
                         "the step is still one optimizer update over the whole cohort)")
    ap.add_argument("--temperature", type=float, default=1.0, help="opening sampling temperature")
    ap.add_argument("--temp-end", type=float, default=0.35, help="annealed-to temperature")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--c-value", type=float, default=1.0)
    ap.add_argument("--c-entropy", type=float, default=0.05, help="entropy bonus coefficient")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-cohorts", type=int, default=0, help="0 = run forever")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--eval-games", type=int, default=60, help="games per opponent")
    ap.add_argument("--n-top", type=int, default=2, help="top-rated snapshots kept as opponents")
    ap.add_argument("--n-anchors", type=int, default=3, help="frozen snapshots spread across Elo range")
    ap.add_argument("--opening-plies", type=int, default=8,
                    help="eval: opening plies sampled with temperature for game diversity")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    eval_interval = parse_duration(args.eval_every)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    net = ChessNet().to(device)
    # Train through a compiled wrapper (fuses the tiny launch-bound ChessNet);
    # keep the raw module for state_dict / safetensors / ONNX so the saved keys
    # stay plain FQNs (no `_orig_mod.` prefix) that the Rust forward loads.
    # NB: mode="reduce-overhead" (CUDA graphs) is incompatible with our
    # micro-batch gradient accumulation — each chunk's backward needs its forward
    # activations to persist, but cudagraphs reuse that static memory across
    # chunks (verified to error; see git history). Default mode it is.
    net_train = torch.compile(net)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)

    base_elapsed, total_cohorts, total_games, total_steps = 0.0, 0, 0, 0
    next_eval_at = eval_interval
    latest = out_dir / "latest.pt"
    if latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["weights"])
        opt.load_state_dict(ckpt["opt"])
        base_elapsed = ckpt.get("elapsed_sec", 0.0)
        total_cohorts = ckpt.get("cohorts", 0)
        total_games = ckpt.get("games", 0)
        total_steps = ckpt.get("train_steps", 0)
        next_eval_at = ckpt.get("next_eval_at", eval_interval)
        if "np_rng" in ckpt:
            rng.bit_generator.state = ckpt["np_rng"]
        print(f"resumed from {latest} (elapsed={base_elapsed/3600:.2f}h "
              f"cohorts={total_cohorts} games={total_games})", flush=True)
    else:
        torch.manual_seed(args.seed)
        print(f"cold start (seed={args.seed})", flush=True)

    print(f"net: blocks={N_BLOCKS} filters={N_FILTERS} params={n_params(net):,}", flush=True)

    serving_st = out_dir / "serving_weights.safetensors"
    # The cohort is large (rows * 5826 f32) and rewritten every cycle; keep it on
    # tmpfs (RAM) when available so the round-trip is RAM-speed and we don't grind
    # the SSD. It's transient (consumed immediately by the train step).
    shm = Path("/dev/shm")
    cohort_path = (shm if shm.is_dir() else out_dir) / f"reinforce_cohort_{out_dir.name}.bin"
    save_weights_st(net, str(serving_st))  # initial weights for the first self-play

    start = time.time()

    def elapsed():
        return base_elapsed + (time.time() - start)

    def save_latest():
        tmp = latest.with_suffix(".tmp")
        torch.save({
            "weights": net.state_dict(),
            "opt": opt.state_dict(),
            "elapsed_sec": elapsed(),
            "cohorts": total_cohorts,
            "games": total_games,
            "train_steps": total_steps,
            "next_eval_at": next_eval_at,
            "np_rng": rng.bit_generator.state,
            "config": {"n_blocks": N_BLOCKS, "n_filters": N_FILTERS},
        }, tmp)
        tmp.replace(latest)

    def save_snapshot():
        hours = int(elapsed() / 3600)
        utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"snap_h{hours:05d}_{utc}"
        st_path = out_dir / "snapshots" / f"{stem}.safetensors"
        onnx_path = out_dir / "snapshots" / f"{stem}.onnx"
        save_weights_st(net, str(st_path))
        export_module(net, str(onnx_path))
        print(f"[snapshot] {st_path.name}", flush=True)
        return st_path

    def run_eval(st_path):
        # The Rust evaluator owns the pool/Elo bookkeeping: it picks the active
        # opponent set (top-N + random + spread anchors) from pool.json, plays
        # greedy raw-policy matches, refits a global Bradley-Terry Elo, and writes
        # pool.json. It streams its own `[eval]` logs to our stdout.
        if args.no_eval or not EVAL_BIN.exists():
            if not args.no_eval:
                print(f"[eval] skipped: {EVAL_BIN} not built", flush=True)
            return
        try:
            subprocess.run(
                [str(EVAL_BIN), "--run-dir", str(out_dir), "--candidate", str(st_path),
                 "--games", str(args.eval_games), "--n-top", str(args.n_top),
                 "--n-anchors", str(args.n_anchors), "--opening-plies", str(args.opening_plies),
                 "--max-plies", str(args.max_plies), "--seed", str(total_cohorts)],
                cwd=str(HERE), env=env, check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"[eval] failed (exit {e.returncode})", flush=True)

    env = worker_env()
    worker = start_selfplay_worker(env)

    def selfplay(seed):
        """One cohort via the persistent worker; respawn + retry once on death."""
        nonlocal worker
        cmd = dict(
            weights=str(serving_st), out=str(cohort_path),
            target_rows=args.cohort_rows, batch=args.selfplay_batch,
            threads=args.selfplay_threads, concurrency=args.selfplay_concurrency,
            temperature=args.temperature, temp_end=args.temp_end,
            max_plies=args.max_plies, seed=seed,
        )
        try:
            return run_cohort(worker, **cmd)
        except (RuntimeError, BrokenPipeError, ValueError) as e:
            print(f"[selfplay] worker error ({e}); respawning", flush=True)
            try:
                worker.kill()
            except ProcessLookupError:
                pass
            worker = start_selfplay_worker(env)
            return run_cohort(worker, **cmd)

    while True:
        t0 = time.time()
        seed = args.seed + total_cohorts * 7919 + 1
        ack = selfplay(seed)
        sp_sec = time.time() - t0

        cohort = read_cohort(str(cohort_path))
        t1 = time.time()
        # Exactly one optimizer step per cycle, accumulated over the whole cohort
        # in --micro-batch chunks so a huge cohort's single step stays within GPU
        # memory (and coexists with other GPU jobs on the box).
        stats = train_on_cohort(
            net_train, opt, cohort, device,
            micro_batch=args.micro_batch,
            c_value=args.c_value, c_entropy=args.c_entropy, rng=rng,
        )
        tr_sec = time.time() - t1

        save_weights_st(net, str(serving_st))
        total_cohorts += 1
        total_games += int(ack.get("rows", cohort["n"]))  # games unknown; track rows
        total_steps += stats["steps"]
        save_latest()

        did_eval = False
        if elapsed() >= next_eval_at:
            st_path = save_snapshot()
            next_eval_at += eval_interval
            run_eval(st_path)
            did_eval = True

        print(json.dumps({
            "t": round(elapsed(), 1),
            "cohorts": total_cohorts,
            "rows": cohort["n"],
            "tr_steps": total_steps,
            "z_mean": round(float(cohort["z"].mean()), 3),
            "sp_sec": round(sp_sec, 2),
            "tr_sec": round(tr_sec, 2),
            "loss": round(stats["loss"], 4),
            "policy": round(stats["policy"], 4),
            "value": round(stats["value"], 4),
            "entropy": round(stats["entropy"], 4),
            "eval": did_eval,
            "next_eval_in": round(next_eval_at - elapsed(), 1),
        }), flush=True)

        if args.max_cohorts and total_cohorts >= args.max_cohorts:
            print("reached --max-cohorts; stopping", flush=True)
            break


if __name__ == "__main__":
    main()
