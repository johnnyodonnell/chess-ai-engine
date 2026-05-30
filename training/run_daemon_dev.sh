#!/usr/bin/env bash
# pm2 entrypoint for the ISOLATED dev run of the parallel orchestrator (Path B).
# Runs from the dev checkout, writes to runs/dev1, and warm-starts (cold start
# only) from the live run1 checkpoint — leaving the production run1 untouched.
#   pm2 start training/run_daemon_dev.sh --name chess-train-dev
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Reuse the venv from the live checkout (deps already installed there).
VENV="/home/johnny/Workspace/chess-ai-engine/training/.venv"
OUT_DIR="$REPO_DIR/runs/dev1"
INIT_FROM="${INIT_FROM:-/home/johnny/Workspace/chess-ai-engine/runs/run1/latest.pt}"

cd "$REPO_DIR/training"
exec "$VENV/bin/python" orchestrator.py \
  --out-dir "$OUT_DIR" \
  --init-from "$INIT_FROM" \
  --snapshot-every "${SNAPSHOT_EVERY:-4h}" \
  --workers "${WORKERS:-12}" \
  --games-per-worker "${GAMES_PER_WORKER:-16}" \
  --sims "${SIMS:-200}"
