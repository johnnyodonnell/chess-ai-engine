# Stockfish-anchored Elo evaluation

The in-repo evaluator (`training/evaluate.py`) fits a **self-play** Elo with the
random player pinned at 0. That scale is internally consistent but has no
absolute anchor, so it can drift. This tool measures the **exact model + search
that ships in the web app** (`public/models/current.onnx`, sequential PUCT,
C_PUCT=1.5, 400 sims) against **Stockfish 17.1** for an absolute Elo.

## Setup (already done locally; not committed)

- `.venv/` — `uv venv` with `python-chess`, `numpy`, `onnxruntime` (CPU).
- `bin/stockfish` → Stockfish 17.1 (`x86-64-bmi2`) downloaded from the official
  GitHub release into `stockfish/`.

These are git-ignored (large binaries). Recreate with:

```sh
uv venv .venv --python 3.12
uv pip install --python .venv "python-chess" "numpy" "onnxruntime"
curl -L -o sf.tar https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-bmi2.tar
tar xf sf.tar && ln -sf "$PWD/stockfish/stockfish-ubuntu-x86-64-bmi2" bin/stockfish
```

## Files

- `az_agent.py` — faithful Python port of `src/engine/alphazero/mcts.js`,
  reusing `training/encode.py` so the net input is byte-identical to the browser.
- `match.py` — plays a match vs Stockfish (alternating colors, randomized
  openings, draw adjudication at the move cap) and converts the score to an Elo
  gap with a 95% CI.

## Running

```sh
# anchor against a calibrated Stockfish Elo (UCI_LimitStrength)
.venv/bin/python match.py --games 24 --sf-elo 1900 --sf-movetime 0.1
```

## Method notes / caveats

- Stockfish is throttled with `UCI_LimitStrength` + `UCI_Elo`. The reported
  AZ Elo = `UCI_Elo + gap`, where `gap = -400·log10(1/score − 1)`.
- Engine-anchored Elo is **opponent-dependent**: it is only trustworthy near a
  50% score. Far from 50% the logistic extrapolation is unreliable (and SF's own
  UCI_Elo calibration is not perfectly transitive against a net+MCTS opponent),
  so we bracket the 50% crossover rather than trusting a lopsided match.
- Search is fixed at 400 sims (no time cap) to measure full intended strength.
  The browser additionally caps at 1.5 s, so on slow clients live strength can be
  lower if 400 sims don't finish in time.
