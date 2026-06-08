# Stockfish-anchored Elo of the live web-app model

**Model:** `public/models/current.onnx` = snapshot **h00100**
(`snap_h00100_20260604T1840Z`, ~100 h cumulative self-play on `asus-nvidia`,
exported from the snapshot `.pt`), played with the exact browser search
(sequential PUCT, C_PUCT=1.5, **400 sims**).
**Opponent:** Stockfish 17.1 (x86-64-bmi2), `UCI_LimitStrength`, 0.1 s/move,
1 thread, 64 MB hash. Alternating colors, 2 random opening plies, draw
adjudication at 200 moves.

## Match results

| SF UCI_Elo | Games | AZ score | W–D–L  | Single-anchor AZ Elo |
|-----------:|------:|---------:|:------:|---------------------:|
| 2000       | 4     | 0.750    | 2–2–0  | ~2191 (few games)    |
| **2200**   | **24**| **0.583**| 10–8–6 | **~2258**            |
| **2400**   | **24**| **0.438**| 5–11–8 | **~2356**            |

## Pooled estimate

Maximum-likelihood Elo fit over the two 24-game near-50% anchors (2200/2400,
48 games; adding the 4-game 2000 anchor moves it <10 Elo):

> **AZ Elo ≈ 2305  (95% CI ≈ 2200 – 2410)**

The 50% crossover sits at ~2310 (linear interpolation of the observed 2200/2400
scores agrees: ~2314). The observed curve is slightly flatter than the logistic
(model 0.651/0.371 vs observed 0.583/0.438 at 2200/2400), the expected
non-transitivity of SF's `UCI_Elo` against a net+MCTS opponent — so treat ~2305
as the central estimate, not a tight point.

## Change vs the previous checkpoint

The prior live model (h00080-era, measured 2026-06-03) sat at **~1936 Elo**, and
scored only 0.438 at SF 2000 and 0.125 at SF 2400. h00100 now scores 0.750 at
SF 2000 and 0.438 at SF 2400 — roughly **+370 Elo**, moving from strong club
player into expert / candidate-master territory.

## Why this differs from the in-repo number

`training/evaluate.py` (now `evaluate_rs`) reports a **self-play** Elo with the
random player pinned at 0 — an internally-consistent but unanchored scale. This
match gives an **absolute** rating against a calibrated external engine,
measured at the full 400-sim search strength.

Caveat: engine-anchored Elo is opponent-dependent and SF's `UCI_Elo`
calibration is not perfectly transitive against a net+MCTS opponent. The 2000
row (4 games, far from 50%) is a weak bound; the number to trust is the pooled
fit from the near-50% 2200/2400 matches.

_Reproduce: see `README.md`; e.g._
_`.venv/bin/python match.py --games 24 --sf-elo 2200 --sf-movetime 0.1`._
