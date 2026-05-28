# Chess AI Engine

A chess web app (React + Vite) where you play White against a bot. The repo
is laid out so smarter engines can drop in later by satisfying the same
one-line contract.

## Running locally

```sh
npm install
npm run dev        # or: ./run-local.sh
```

Then open the printed URL. `npm run build` produces a production build in
`dist/`. Vite serves the trained `.onnx` model from `public/models/` and the
onnxruntime-web runtime assets through a small middleware in `vite.config.js`.

## The engine

Every engine lives in `src/engine/` and exposes the same function:

```js
bestMove(fen)  // fen: a chess position string  ->  a move object
```

Engines may be sync or async (the caller awaits either). The app picks one
with a single import in `src/App.jsx`.

| Engine | Approach | Result |
| --- | --- | --- |
| Random | uniform-random legal move | ~300–500 ELO; loses every game to Stockfish at its weakest setting |
| AlphaZero | small ResNet + MCTS, trained by self-play | visibly stronger than random; current checkpoint is 1 h of cold-start training |

### Random — `src/engine/random.js`

Five lines: ask `chess.js` for the legal moves in the current position, pick
one uniformly. No search, no evaluation, no learning.

### AlphaZero — `src/engine/alphazero/`

Small ResNet (4 blocks × 96 filters, ~1.3 M params) with the standard
AlphaZero policy-and-value heads, plus PUCT MCTS at inference time
(400 simulations or 1.5 s, whichever comes first). The net is trained from
random init by self-play on the box at `asus-nvidia` (code in `training/`);
each checkpoint is exported to ONNX and shipped at `public/models/current.onnx`
for the browser to load via `onnxruntime-web` (WebGPU when available, WASM
fallback).

The JS encoder (`encode.js`) is byte-for-byte mirrored from the Python one
(`training/encode.py`) — divergence in encoding silently destroys the model's
predictions, so any change in one must be matched in the other.

## Project layout

```
src/
  App.jsx                    game UI and turn logic
  components/                Board, Status
  engine/
    game.js                  thin chess.js wrapper — rules, status, FEN helpers
    random.js                engine 1 — uniform-random legal move
    alphazero/               engine 2 — ResNet + MCTS via onnxruntime-web
      encode.js              FEN → tensor; move ↔ policy index
      net.js                 ONNX session wrapper
      mcts.js                PUCT search
      index.js               bestMove(fen) entry point
  styles/app.css             visuals

training/                    Python self-play + training (run on asus-nvidia)
  encode.py                  mirror of src/engine/alphazero/encode.js
  net.py                     ChessNet (ResNet)
  mcts.py                    batched PUCT for self-play
  selfplay.py                concurrent self-play games
  train.py                   loss + optimizer step
  loop.py                    orchestrator (15m / 1h / 4h milestones)
  export.py                  PyTorch → ONNX
  eval_stockfish.py          arena vs. Stockfish for ELO estimation
  eval_random.py             pure-random baseline vs. Stockfish

public/
  models/current.onnx        the live AlphaZero checkpoint
```

## Ongoing training & snapshots

A long-running self-play process is running on `asus-nvidia` under pm2,
saving a snapshot every 12 hours of cumulative training time.

**Snapshots live at:** `~/chess-runs/run1/snapshots/` on `asus-nvidia`

```sh
# List available snapshots
ssh asus-nvidia 'ls ~/chess-runs/run1/snapshots/'

# Check training progress
ssh asus-nvidia 'pm2 logs chess-train --lines 30 --nostream'

# Try a snapshot locally — pick one from the list above, then:
scp asus-nvidia:~/chess-runs/run1/snapshots/snap_h00024_<UTC>.onnx public/models/current.onnx
npm run dev
```

Snapshot naming: `snap_h<NNNNN>_<YYYYMMDDTHHMMZ>.{pt,onnx}` where `h<NNNNN>`
is cumulative hours trained (5-digit padded, persists across restarts).
`latest.pt` in the same run dir holds the current weights + optimizer state
for crash recovery.
