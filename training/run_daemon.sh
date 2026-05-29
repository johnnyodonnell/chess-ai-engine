#!/usr/bin/env bash
# pm2 entrypoint for indefinite self-play training.
# Run from the repo root: pm2 start training/run_daemon.sh --name chess-train
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_DIR/training/.venv"
OUT_DIR="$REPO_DIR/runs/run1"

# One-time warm start: set INIT_FROM to a checkpoint to seed a fresh run from
# existing weights (clean clock/cadence). Unset it after the first boot.
INIT_ARGS=()
if [[ -n "${INIT_FROM:-}" ]]; then
  INIT_ARGS=(--init-from "$INIT_FROM")
fi

cd "$REPO_DIR/training"
exec "$VENV/bin/python" loop.py \
  --snapshot-every 4h \
  --save-latest-every 300s \
  --out-dir "$OUT_DIR" \
  "${INIT_ARGS[@]}"
