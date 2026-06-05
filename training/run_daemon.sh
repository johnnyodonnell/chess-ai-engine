#!/usr/bin/env bash
# pm2 entrypoint for indefinite self-play training.
#   pm2 start training/run_daemon.sh --name chess-train --kill-timeout 20000
#
# The orchestrator runs the trainer (PyTorch, paced SGD) in this process and
# spawns the standalone Rust self-play worker (selfplay_rs) as a subprocess. The
# worker runs MCTS + the AOTInductor forward 100% in Rust (no Python in the
# self-play path) and streams finished games back over stdout. The trainer
# recompiles serving_model.pt2 every --pt2-every and the worker reloads it.
# Resumes runs/run1's latest.pt (checkpoint format shared with loop.py).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_DIR/training/.venv"
OUT_DIR="${OUT_DIR:-$REPO_DIR/runs/run1}"

# One-time warm start: set INIT_FROM to seed a fresh run. Ignored once latest.pt exists.
INIT_ARGS=()
if [[ -n "${INIT_FROM:-}" ]]; then
  INIT_ARGS=(--init-from "$INIT_FROM")
fi

cd "$REPO_DIR/training"
exec "$VENV/bin/python" orchestrator.py \
  --snapshot-every "${SNAPSHOT_EVERY:-4h}" \
  --save-latest-every 300s \
  --out-dir "$OUT_DIR" \
  --workers "${WORKERS:-12}" \
  --sims "${SIMS:-200}" \
  --pipeline-n-threads "${PIPELINE_N_THREADS:-16}" \
  --pt2-every "${PT2_EVERY:-300s}" \
  "${INIT_ARGS[@]}"
