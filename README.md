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
  loop.py                    orchestrator (indefinite self-play + snapshots)
  export.py                  PyTorch → ONNX
  evaluate.py                self-play Elo eval + anchor pool + model serving

public/
  models/current.onnx        the live AlphaZero checkpoint
```

## Ongoing training, evaluation & serving

A long-running self-play process runs on `asus-nvidia` under pm2, saving a
snapshot every **4 hours** of cumulative training time. After each snapshot a
detached `evaluate.py` process plays the new snapshot against the active pool,
fits a self-play Elo, and promotes the strongest model to `best.onnx`.

Everything for a run lives in one folder in the repo:
**`~/Workspace/chess-ai-engine/runs/run1/`** on `asus-nvidia`

```
runs/run1/
  snapshots/                 immutable snap_h<NNNNN>_<UTC>.{pt,onnx}
  latest.pt                  weights+optimizer+RNG+clock for crash recovery
  best.onnx / best.pt        current strongest model (by self-play Elo)
  pool.json                  models, ratings, accumulated match results
  eval.log                   evaluation subprocess output
```

```sh
# Check training progress
ssh asus-nvidia 'pm2 logs chess-train --lines 30 --nostream'

# Standings (ratings + which model is served)
ssh asus-nvidia 'cat ~/Workspace/chess-ai-engine/runs/run1/pool.json'

# Ship the current strongest model:
scp asus-nvidia:~/Workspace/chess-ai-engine/runs/run1/best.onnx public/models/current.onnx
npm run dev
```

**Evaluation:** the pool holds 8 active models — the 3 top performers plus 5
anchors (a fixed random player pinned at Elo 0 and 4 frozen ex-snapshots spread
across the strength range below the top 3). Ratings are a lightweight Elo fit
over accumulated match results with random pinned at 0 — purely self-play, no
external engine. The frozen anchors keep the scale from drifting and catch
regression. `best.onnx` only changes when a model beats the incumbent by a
confidence margin.

Snapshot naming: `snap_h<NNNNN>_<YYYYMMDDTHHMMZ>.{pt,onnx}` where `h<NNNNN>` is
cumulative hours trained (5-digit padded, persists across restarts).
