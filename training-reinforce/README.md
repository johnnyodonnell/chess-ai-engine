# training-reinforce — vanilla REINFORCE chess trainer

A second, independent trainer for the chess engine that learns by **REINFORCE**
(policy-gradient self-play, *no search*), as opposed to the AlphaZero/MCTS trainer
in `../training`. Modeled on the REINFORCE setup in `fox-lite-ai-engine/training`.

It reuses the AlphaZero stack for everything board-related so models stay
interchangeable with the rest of the repo:
- **`../training/chess_core`** — Rust chess rules, 18×8×8 encoding, 4672-move indexing.
- **`../training/net.py` `ChessNet`** — the 4×96 ResNet (policy[4672] + value), unchanged.
- **`../training/encode.py`**, **`../training/export.py`** — encoding + ONNX export.

## The loop (`orchestrator.py`)

Strict-synchronous, zero GPU contention:

1. **Self-play** — the Rust worker (`selfplay_rs`) plays games current-vs-current,
   sampling each move from the temperature-scaled, legal-masked policy, until a
   cohort of `--cohort-rows` decisions is collected → `cohort.bin`.
2. **Train** — **one full-batch REINFORCE SGD step** over the whole cohort
   (`sgd_batch = n`, so it is *exactly one* update per cycle). Loss =
   `policy_gradient(adv = z - V) + c_value·MSE(V,z) − c_entropy·entropy`, where
   `z` is the terminal game outcome from the mover's POV (win +1 / draw 0 / loss −1).
3. **Publish** new weights (`serving_weights.safetensors`) for the next self-play.
4. **Checkpoint** `latest.pt` (resumable).
5. Every `--eval-every`, **snapshot** (safetensors + ONNX) and run the **Rust
   pool-Elo evaluator** (`evaluate_reinforce`, ported from the AlphaZero `eval.rs`
   but greedy raw-policy instead of MCTS): greedy games vs a random anchor + past
   snapshots, refitting Bradley-Terry Elo into `pool.json`.

## Build

```bash
cd selfplay_rs
SELFPLAY_PYTHON=/path/to/venv/bin/python cargo build --release
```

Requires a venv with torch (cu128) + python-chess + onnx + safetensors + numpy.
`build.rs` force-links the torch CUDA backend via `SELFPLAY_PYTHON`.

## Parity gate + smoke test

```bash
python make_fixture.py                              # writes parity/
./selfplay_rs/target/release/selfplay_reinforce forward-check parity   # FORWARD-CHECK OK
./selfplay_rs/target/release/selfplay_reinforce selfplay \
    --weights parity/fwd_weights.safetensors --out /tmp/c.bin --target-rows 2000
python cohort.py /tmp/c.bin                         # COHORT OK
```

## Run

```bash
nohup ./run_daemon.sh --out-dir runs/run1 > runs/run1.log 2>&1 &
```

`run_daemon.sh` defaults `VENV` to the chess-ai-engine venv; override with `VENV=...`.

## Cohort-size tuning

Because the trainer sets `sgd_batch = n`, the cycle is **always exactly one
full-batch SGD step**, regardless of cohort size. Push `--cohort-rows` as high as
that single forward+backward fits in GPU memory (back off on OOM); stop once
self-play time per cohort dominates the cycle. `8192` is a safe starting value, not
a target — ramp it up while watching `tr_sec` and memory in the logs.

## Key flags

## Self-play pipeline

Self-play is CPU-bound (legal-move gen + 4672-wide mask + sampling), not GPU.
`selfplay_rs` runs a **decoupled pipeline** (`pipeline.rs`): `--selfplay-threads`
CPU worker threads run game logic while one inference thread runs the batched GPU
forward, with `--selfplay-concurrency` (≈ 2× batch) games in flight to overlap the
two. libtorch is pinned to 1 CPU thread so the cores go to the workers. Bigger
`--selfplay-batch` amortizes the inference-thread overhead (256→~25k rows/s,
512→~43k rows/s), but raises the cohort size (≈ concurrency × game length), so the
single SGD step + cohort I/O grow too. The cohort is written to `/dev/shm` (tmpfs)
when present to keep its round-trip off the SSD.

## Key flags

| flag | default | meaning |
|------|---------|---------|
| `--cohort-rows` | 8192 | floor for rows per cohort (actual ≈ concurrency × game length) |
| `--selfplay-batch` | 512 | max GPU forward width |
| `--selfplay-threads` | 16 | CPU game-worker threads |
| `--selfplay-concurrency` | 0 (=2×batch) | games kept in flight (overlap CPU/GPU) |
| `--lr` | 1e-3 | AdamW learning rate |
| `--c-value` | 1.0 | value-baseline loss weight |
| `--c-entropy` | 0.05 | entropy bonus coefficient |
| `--temperature` / `--temp-end` | 1.0 / 0.35 | opening → annealed sampling temp |
| `--max-plies` | 200 | ply cap; a game over it is scored a draw |
| `--eval-every` | 30m | snapshot + pool-Elo eval cadence |
| `--eval-games` | 60 | games per opponent in eval |
