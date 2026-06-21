#!/usr/bin/env bash
# Launch the REINFORCE self-play / train orchestrator.
#
#   nohup ./run_daemon.sh --out-dir runs/run1 > runs/run1.log 2>&1 &
#
# The orchestrator runs the PyTorch trainer in this process and drives the Rust
# self-play worker (selfplay_reinforce) as a persistent subprocess. Resumes
# <out-dir>/latest.pt if present.
#
# VENV defaults to the proven chess-ai-engine venv (torch cu128, python-chess,
# onnx, safetensors). Override with VENV=/path/to/venv.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$HOME/Workspace/chess-ai-engine/training/.venv}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "venv python not found at $VENV/bin/python (set VENV=...)" >&2
  exit 1
fi

# build.rs (if rebuilt) and the LD_LIBRARY_PATH the worker needs both key off this.
export SELFPLAY_PYTHON="$VENV/bin/python"

cd "$HERE"
exec "$VENV/bin/python" orchestrator.py "$@"
