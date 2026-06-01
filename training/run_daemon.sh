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

# Self-play concurrency. rust_mcts makes a worker's per-sim CPU work negligible,
# so all workers fall into a lockstep "convoy" on the central GPU inference
# server: it runs one big batched forward, then sits idle (~80-90%) waiting for
# the whole convoy to wake (mp.Event latency) and resubmit together. Each cycle
# is gated by that fixed round-trip latency, not the GPU — so the cure is to
# pack far more positions into every forward by raising games-per-worker.
# Measured on run1's net (20 workers, 200 sims): gpw=16 -> ~22k pos/s (HALF the
# Python-MCTS rate — the regression), gpw=96 -> ~49k pos/s (beats it, GPU still
# ~90% idle). The slower paths (python / rust board) pipeline naturally via
# their own CPU jitter and gain nothing from a big gpw, so keep their tuned 16.
if [[ "$CHESS_BACKEND" == "rust_mcts" ]]; then
  DEFAULT_GPW=96
else
  DEFAULT_GPW=16
fi

cd "$REPO_DIR/training"
exec "$VENV/bin/python" orchestrator.py \
  --snapshot-every "${SNAPSHOT_EVERY:-4h}" \
  --save-latest-every 300s \
  --out-dir "$OUT_DIR" \
  --workers "${WORKERS:-20}" \
  --games-per-worker "${GAMES_PER_WORKER:-$DEFAULT_GPW}" \
  --sims "${SIMS:-200}" \
  "${INIT_ARGS[@]}"
