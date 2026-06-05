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
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

# The self-play worker is now a standalone Rust binary (selfplay_rs) spawned via
# subprocess; it talks to this trainer over a stdout frame stream. No Python
# self-play backend / CHESS_BACKEND selection anymore.
SELFPLAY_BIN = Path(__file__).resolve().parent / "selfplay_rs" / "target" / "release" / "selfplay_rs"
# Evaluation is now the standalone Rust binary too (port of evaluate.py), built
# from the same crate. Launched as a detached subprocess after each snapshot.
EVAL_BIN = Path(__file__).resolve().parent / "selfplay_rs" / "target" / "release" / "evaluate_rs"
_ROW_FLOATS = 18 * 8 * 8 + 4672 + 1  # state[1152] + pi[4672] + z


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
                    help="How often the trainer republishes weights: serving_weights.pt "
                         "(tooling) + serving_weights.safetensors (the raw bf16 weights "
                         "the Rust worker swaps into the live AOTI constant buffer).")
    ap.add_argument("--pt2-every", default="300s",
                    help="Deprecated/unused: the AOTInductor serving_model.pt2 is now "
                         "compiled once at startup; weights refresh via the safetensors "
                         "sidecar (see --publish-every), no periodic recompile.")
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
    # rust_pipeline knobs (decoupled non-blocking engine). The GPU batch width is a
    # fixed architectural constant (net.SERVING_BATCH = pipeline.rs BATCH); the .pt2
    # is exported at it and the worker is hard-coded to it, so it is not a flag.
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
    from net import ChessNet, N_BLOCKS, N_FILTERS, SERVING_BATCH, n_params
    from train import ReplayBuffer, train_step
    from export import export_module

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_interval = parse_duration(args.snapshot_every)
    save_latest_interval = parse_duration(args.save_latest_every)
    publish_interval = parse_duration(args.publish_every)
    pt2_interval = parse_duration(args.pt2_every)

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
    # The AOTInductor package (Inductor's fused Triton cubins, no Python) defines the
    # worker's forward graph + holds the initial weights. It is compiled ONCE (compiling
    # is ~5s of GPU); fresh weights are then pushed into the live runner's constant
    # buffer every 15s via the safetensors sidecar below (no recompile). Compiled with
    # use_runtime_constant_folding=True so the runner re-derives folded constants
    # (BN-into-conv etc.) from the pushed primaries — see selfplay_rs maybe_reload.
    serving_pt2_path = out_dir / "serving_model.pt2"
    # Raw weights the worker swaps in every publish (bf16, keyed by state_dict FQN).
    serving_st_path = out_dir / "serving_weights.safetensors"

    def _export_pt2(path):
        import torch._inductor  # noqa: F401  (registers aoti_compile_and_package)
        # Keep folded constants updatable at runtime: the worker pushes raw primaries
        # and calls run_const_fold to re-derive the _FOLDED_CONST_* tensors.
        torch._inductor.config.aot_inductor.use_runtime_constant_folding = True
        ex = ChessNet(n_blocks=N_BLOCKS, n_filters=N_FILTERS).to(device).eval()
        ex.load_state_dict(net.state_dict())
        ex = ex.bfloat16()  # prod self-play runs bf16 (matches the parity baseline)
        x = torch.zeros(SERVING_BATCH, 18, 8, 8, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            ep = torch.export.export(ex, (x,))
            # aoti_compile_and_package requires the path to end in .pt2.
            tmp = str(path.with_suffix("")) + ".tmp.pt2"
            torch._inductor.aoti_compile_and_package(ep, package_path=tmp)
        os.replace(tmp, str(path))

    def publish_weights():
        _atomic_save(serving_path, {
            "weights": net.state_dict(),
            "config": {"n_blocks": N_BLOCKS, "n_filters": N_FILTERS},
        })

    def publish_safetensors():
        # The worker pushes these raw bf16 primaries into the AOTI constant buffer
        # (mangling FQN '.'->'_'). Drop num_batches_tracked (not an AOTI constant).
        from safetensors.torch import save_file
        sd = {k: v.detach().to("cpu", torch.bfloat16).contiguous()
              for k, v in net.state_dict().items() if "num_batches_tracked" not in k}
        tmp = str(serving_st_path) + ".tmp"
        save_file(sd, tmp)
        os.replace(tmp, str(serving_st_path))

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

    def _save_snapshot_st(path):
        # Snapshot weights as fp32 safetensors (the Rust evaluator reads these;
        # no pickled .pt). Drop num_batches_tracked (not a model parameter).
        from safetensors.torch import save_file
        sd = {k: v.detach().to("cpu", torch.float32).contiguous()
              for k, v in net.state_dict().items() if "num_batches_tracked" not in k}
        tmp = str(path) + ".tmp"
        save_file(sd, tmp)
        os.replace(tmp, str(path))

    def save_snapshot():
        snap_dir = out_dir / "snapshots"
        snap_dir.mkdir(exist_ok=True)
        hours = int(elapsed() / 3600)
        utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%MZ")
        stem = f"snap_h{hours:05d}_{utc}"
        st_path = snap_dir / f"{stem}.safetensors"
        onnx_path = snap_dir / f"{stem}.onnx"
        _save_snapshot_st(st_path)
        export_module(net, str(onnx_path))  # ONNX from the in-memory net (no .pt round-trip)
        print(f"[snapshot] {st_path.name}  {onnx_path.name}", flush=True)
        return st_path

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

    def launch_eval(st_path):
        # Reap the previous eval if it has finished, so it doesn't linger as a
        # zombie (parent never exits) and trip the staleness check below.
        prev = launch_eval.proc
        if prev is not None and prev.poll() is not None:
            launch_eval.proc = None
        if not EVAL_BIN.exists():
            raise SystemExit(f"eval binary not found: {EVAL_BIN} "
                             f"(build it: cd selfplay_rs && cargo build --release)")
        lock = out_dir / "eval.lock"
        if lock.exists():
            try:
                prev_pid = int(lock.read_text().split()[0])
                if _pid_alive(prev_pid):
                    print(f"[eval] prior eval (pid {prev_pid}) still running; "
                          f"skipping {st_path.name}", flush=True)
                    return
            except (ValueError, IndexError, OSError):
                pass
        training_dir = Path(__file__).resolve().parent
        logf = open(out_dir / "eval.log", "a")
        try:
            proc = subprocess.Popen(
                [str(EVAL_BIN),
                 "--run-dir", str(out_dir), "--candidate", str(st_path)],
                cwd=str(training_dir), stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            logf.close()
        launch_eval.proc = proc
        lock.write_text(f"{proc.pid} {time.time()}")
        print(f"[eval] launched evaluate_rs pid={proc.pid} for {st_path.name}",
              flush=True)

    launch_eval.proc = None

    # ----- Spawn the standalone Rust self-play worker -----
    publish_weights()              # serving_weights.pt (tooling / warm-start)
    publish_safetensors()          # serving_weights.safetensors (worker swaps these in)
    _export_pt2(serving_pt2_path)  # the .pt2 forward graph — compiled ONCE (arch is fixed)

    out_queue = pyqueue.Queue(maxsize=args.queue_max)
    stop = threading.Event()

    def _read_exact(f, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = f.read(n - len(buf))
            if not chunk:
                return None  # EOF: worker exited
            buf += chunk
        return bytes(buf)

    def _reader_loop(stdout):
        # Decode length-prefixed game frames from the worker into (rows, 1, n_rows)
        # on out_queue — the same trainer contract the Python worker used.
        while True:
            hdr = _read_exact(stdout, 4)
            if hdr is None:
                return
            (n_rows,) = struct.unpack("<I", hdr)
            payload = _read_exact(stdout, n_rows * _ROW_FLOATS * 4)
            if payload is None:
                return
            flat = np.frombuffer(payload, dtype="<f4").reshape(n_rows, _ROW_FLOATS)
            rows = [(r[:1152].reshape(18, 8, 8).copy(), r[1152:1152 + 4672].copy(), float(r[-1]))
                    for r in flat]
            while not stop.is_set():
                try:
                    out_queue.put((rows, 1, n_rows), timeout=0.5)
                    break
                except pyqueue.Full:
                    pass

    def _worker_env():
        import glob
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        sp = os.path.dirname(os.path.dirname(torch.__file__))  # site-packages
        libs = [torch_lib] + sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join(libs) + ":" + env.get("LD_LIBRARY_PATH", "")
        return env

    def start_selfplay():
        if not SELFPLAY_BIN.exists():
            raise SystemExit(f"self-play binary not found: {SELFPLAY_BIN} "
                             f"(build it: cd selfplay_rs && cargo build --release)")
        proc = subprocess.Popen(
            [str(SELFPLAY_BIN), "serve",
             "--pt2", str(serving_pt2_path),
             "--weights-st", str(serving_st_path),
             "--threads", str(args.pipeline_n_threads),
             "--sims", str(args.sims),
             "--seed", str(args.seed + 1000)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=_worker_env(),
        )
        t = threading.Thread(target=_reader_loop, args=(proc.stdout,), daemon=True)
        t.start()
        proc.reader_thread = t
        print(f"spawned selfplay_rs worker pid={proc.pid} "
              f"(threads={args.pipeline_n_threads} batch={SERVING_BATCH} "
              f"sims={args.sims})", flush=True)
        return proc

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
            # Supervise the self-play worker — restart if it died.
            if selfplay_proc.poll() is not None:
                print("warn: self-play worker died; restarting", flush=True)
                selfplay_proc = start_selfplay()

            # Snapshot if due.
            if elapsed() >= next_snapshot_at:
                st_path = save_snapshot()
                save_latest()
                last_latest_save = time.time()
                next_snapshot_at += snapshot_interval
                launch_eval(st_path)
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
                publish_safetensors()  # worker swaps these into the live AOTI buffer
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
        selfplay_proc.terminate()  # SIGTERM the Rust worker
        try:
            selfplay_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            selfplay_proc.kill()


if __name__ == "__main__":
    main()
