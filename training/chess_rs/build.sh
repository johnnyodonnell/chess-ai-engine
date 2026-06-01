#!/usr/bin/env bash
# Build the chess_rs wheel (release) and install it into the training venv.
# Run on the aarch64 box; the wheel is arch-specific so it must be built there.
#   ./build.sh
# Override the target venv with VENV=/path/to/venv ./build.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-/home/johnny/Workspace/chess-ai-engine/training/.venv}"
export PATH="$HOME/.cargo/bin:$PATH"

cd "$HERE"
"$VENV/bin/maturin" build --release -i "$VENV/bin/python"
"$VENV/bin/pip" install --force-reinstall --no-deps target/wheels/chess_rs-*.whl
echo "installed chess_rs into $VENV"
