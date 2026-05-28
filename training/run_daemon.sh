#!/usr/bin/env bash
# pm2 entrypoint for indefinite self-play training.
# Run from the repo root: pm2 start training/run_daemon.sh --name chess-train
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_DIR/training/.venv"
OUT_DIR="$HOME/chess-runs/run1"

cd "$REPO_DIR/training"
exec "$VENV/bin/python" loop.py \
  --snapshot-every 12h \
  --save-latest-every 300s \
  --out-dir "$OUT_DIR"
