#!/usr/bin/env bash
# pm2 entrypoint for indefinite self-play training (parallel orchestrator, Path B).
# Run from the repo root: pm2 start training/run_daemon.sh --name chess-train \
#                           --kill-timeout 20000
# The orchestrator runs ONE in-process native self-play engine (Rust owns the
# games and does the GPU forward IN-PROCESS — no inference server, no shm) plus
# the trainer pacing SGD to the data rate. It resumes runs/run1's latest.pt
# (checkpoint format is shared with the legacy loop.py).
#
# Self-play backend (CHESS_BACKEND):
#   rust_pipeline (default) — decoupled non-blocking pipeline: workers never block
#     on the GPU; a global ply-priority queue feeds two inference buckets that one
#     consumer batches; dynamic game count (bucket size is the one batch-width knob).
#   rust_async — bulk-synchronous multi-threaded engine (the A/B baseline).
# Switch/rollback: `CHESS_BACKEND=rust_async pm2 restart chess-train --update-env`.
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

export CHESS_BACKEND="${CHESS_BACKEND:-rust_pipeline}"

# torch.compile the in-process self-play forward (rust_pipeline). The net is tiny
# (4x96), so eager mode is dominated by per-op dispatch overhead; compile fuses it
# into a few kernels. Measured +29% completed games/sec on the GB10. Default ON;
# rollback with `CHESS_COMPILE=0 pm2 restart chess-train --update-env`.
export CHESS_COMPILE="${CHESS_COMPILE:-1}"

# Backend-specific knobs. rust_pipeline: one batch-width knob (bucket size; the
# in-flight game pool self-regulates to ~2x it). rust_async: CPU threads are
# decoupled from batch width (keep N_GAMES >> N_THREADS so batches stay fat).
ENGINE_ARGS=()
if [[ "$CHESS_BACKEND" == "rust_pipeline" ]]; then
  ENGINE_ARGS=(
    --pipeline-bucket-size "${PIPELINE_BUCKET_SIZE:-512}"
    --pipeline-n-threads "${PIPELINE_N_THREADS:-16}"
  )
elif [[ "$CHESS_BACKEND" == "rust_async" ]]; then
  ENGINE_ARGS=(
    --async-n-games "${ASYNC_N_GAMES:-512}"
    --async-n-threads "${ASYNC_N_THREADS:-18}"
    --async-max-batch "${ASYNC_MAX_BATCH:-512}"
  )
else
  echo "run_daemon.sh: CHESS_BACKEND must be rust_pipeline or rust_async" \
       "(got '$CHESS_BACKEND')" >&2
  exit 1
fi

cd "$REPO_DIR/training"
exec "$VENV/bin/python" orchestrator.py \
  --snapshot-every "${SNAPSHOT_EVERY:-4h}" \
  --save-latest-every 300s \
  --out-dir "$OUT_DIR" \
  --workers "${WORKERS:-12}" \
  --games-per-worker "${GAMES_PER_WORKER:-16}" \
  --sims "${SIMS:-200}" \
  "${ENGINE_ARGS[@]}" \
  "${INIT_ARGS[@]}"
