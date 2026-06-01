#!/usr/bin/env bash
# pm2 entrypoint for indefinite self-play training (parallel orchestrator, Path B).
# Run from the repo root: pm2 start training/run_daemon.sh --name chess-train \
#                           --kill-timeout 20000
# The orchestrator runs N CPU-only self-play workers feeding one GPU inference
# server, with the trainer pacing SGD to the data rate. It resumes runs/run1's
# latest.pt (checkpoint format is shared with the legacy loop.py).
# Rollback: swap `orchestrator.py ...` back to `loop.py --snapshot-every 4h
# --save-latest-every 300s --out-dir "$OUT_DIR"` and restart.
#
# Self-play backend: native Rust MCTS engine (chess_rs, CHESS_BACKEND=rust_mcts)
# by default — bit-exact with the Python MCTS (test_mcts_parity.py) but runs the
# whole search in Rust. Instant rollback: start with CHESS_BACKEND=rust (Rust
# board + Python MCTS) or CHESS_BACKEND=python (python-chess reference), e.g.
# `CHESS_BACKEND=rust pm2 restart chess-train --update-env`.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_DIR/training/.venv"
OUT_DIR="$REPO_DIR/runs/run1"

# One-time warm start: set INIT_FROM to a checkpoint to seed a fresh run from
# existing weights (clean clock/cadence). Ignored once latest.pt exists.
INIT_ARGS=()
if [[ -n "${INIT_FROM:-}" ]]; then
  INIT_ARGS=(--init-from "$INIT_FROM")
fi

export CHESS_BACKEND="${CHESS_BACKEND:-rust_mcts}"

# Self-play concurrency + IPC. rust_mcts makes a worker's per-sim CPU work
# negligible, so workers fall into a lockstep "convoy" on the central GPU
# inference server: it runs a batched forward, then idles waiting for the whole
# convoy to wake (mp.Event futex -> scheduler wakeup) and resubmit together. Two
# ways to fight the resulting GPU starvation:
#   1. Huge games-per-worker (gpw=96) to amortize the idle — works (~49k pos/s)
#      but couples gpw to GPU efficiency, hurting replay diversity / burstiness.
#   2. Spin-wait the round trip so a SMALL gpw saturates the GPU (below).
# Measured on run1's net (200 sims): gpw=96 no-spin -> 49k pos/s; gpw=16 no-spin
# -> 27k; gpw=16 + spin (12 workers) -> 67k pos/s (and small gpw = good replay
# diversity, no warmup). So rust_mcts uses small gpw + spin-wait; the slower
# paths (python / rust board) pipeline via their own CPU jitter and don't spin.
if [[ "$CHESS_BACKEND" == "rust_inproc" ]]; then
  # Option B: a single self-play process runs MctsEngine.run entirely in Rust
  # and does the forward IN-PROCESS (no inference server, no shm, no spin-wait).
  # games-in-flight (the NN batch width) = WORKERS * GAMES_PER_WORKER.
  DEFAULT_GPW=16
  DEFAULT_WORKERS=12
elif [[ "$CHESS_BACKEND" == "rust_mcts" ]]; then
  DEFAULT_GPW=16
  DEFAULT_WORKERS=12
  # Hybrid spin-wait: worker busy-checks the response for INFER_SPIN_US us before
  # blocking; server busy-polls (INFER_IDLE_SLEEP=0) so a returning convoy is
  # noticed immediately. Costs CPU (workers spin through the ~2ms forward) but
  # the cores would otherwise sit idle on a dedicated box. Disable by exporting
  # INFER_SPIN_US=0. The spin budget must exceed the forward time to catch the
  # response without sleeping; ~3ms suits run1's net at these batch sizes.
  export INFER_SPIN_US="${INFER_SPIN_US:-3000}"
  export INFER_IDLE_SLEEP="${INFER_IDLE_SLEEP:-0}"
else
  DEFAULT_GPW=16
  DEFAULT_WORKERS=20
fi

cd "$REPO_DIR/training"
exec "$VENV/bin/python" orchestrator.py \
  --snapshot-every "${SNAPSHOT_EVERY:-4h}" \
  --save-latest-every 300s \
  --out-dir "$OUT_DIR" \
  --workers "${WORKERS:-$DEFAULT_WORKERS}" \
  --games-per-worker "${GAMES_PER_WORKER:-$DEFAULT_GPW}" \
  --sims "${SIMS:-200}" \
  "${INIT_ARGS[@]}"
