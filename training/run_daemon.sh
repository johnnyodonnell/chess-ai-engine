#!/usr/bin/env bash
# pm2 entrypoint for indefinite self-play training (parallel orchestrator, Path B).
# Run from the repo root: pm2 start training/run_daemon.sh --name chess-train \
#                           --kill-timeout 20000
# The orchestrator runs N CPU-only self-play workers feeding one GPU inference
# server, with the trainer pacing SGD to the data rate. It resumes runs/run1's
# latest.pt (checkpoint format is shared with the legacy loop.py).
# Rollback: swap `orchestrator.py ...` back to `loop.py --snapshot-every 4h
# --save-latest-every 300s --out-dir "$OUT_DIR"` and restart.
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

cd "$REPO_DIR/training"
exec "$VENV/bin/python" orchestrator.py \
  --snapshot-every "${SNAPSHOT_EVERY:-4h}" \
  --save-latest-every 300s \
  --out-dir "$OUT_DIR" \
  --workers "${WORKERS:-20}" \
  --games-per-worker "${GAMES_PER_WORKER:-16}" \
  --sims "${SIMS:-200}" \
  "${INIT_ARGS[@]}"
