"""Equivalence: the Rust in-process driver MctsEngine.run (option B) vs the
Python-driven micro-loop (the rust_mcts mechanism in selfplay.play_batch_rust).

Same MctsEngine + same seed + same deterministic forward, driven two ways:
  A) Python loop: pending_positions -> evaluate -> apply_evals -> step_moves
  B) Rust loop:   engine.run(forward, sink, stop, refill=False)
Both must emit BIT-IDENTICAL (state, pi, z) sequences — run() executes the same
Rust ops as A, just driven from Rust, so any divergence is a porting bug.

Run (from training/, dev venv):  python test_rust_inproc.py
Gate: zero mismatches; exit 0.
"""

import numpy as np

import chess_rs
from test_mcts_parity import DetEval

N_GAMES = 6
SIMS = 32
SEED = 20260601


def drive_python(seed):
    """The rust_mcts mechanism: Python drives pending/apply/step to completion."""
    ev = DetEval()
    eng = chess_rs.MctsEngine(N_GAMES, SIMS, seed, True)
    while not eng.all_done():
        while (b := eng.pending_positions()) is not None:
            lo, va = ev.evaluate(b)
            eng.apply_evals(lo, va)
        eng.step_moves()
    return eng.take_results()


def drive_rust(seed):
    """Option B: the whole self-play loop runs inside Rust via run()."""
    ev = DetEval()
    eng = chess_rs.MctsEngine(N_GAMES, SIMS, seed, True)
    rows = []
    eng.run(ev.evaluate, rows.extend, lambda: False, False)
    return rows


def assert_rows_equal(a, b):
    assert len(a) == len(b), f"row count differs: py={len(a)} rust={len(b)}"
    for i, ((sa, pa, za), (sb, pb, zb)) in enumerate(zip(a, b)):
        assert np.array_equal(sa, sb), f"row {i}: state tensor differs"
        assert np.array_equal(pa, pb), f"row {i}: pi target differs"
        assert za == zb, f"row {i}: z differs py={za} rust={zb}"


def test_inproc_matches_python_drive():
    a = drive_python(SEED)
    b = drive_rust(SEED)
    assert_rows_equal(a, b)
    return len(a)


def test_refill_runs_and_emits():
    """refill=True keeps generating past the first games' completion; stop() ends
    it. Verify it emits strictly more rows than a single no-refill pass (i.e. it
    refilled and kept going) and terminates."""
    one_pass = len(drive_rust(SEED))
    ev = DetEval()
    eng = chess_rs.MctsEngine(N_GAMES, SIMS, SEED, True)
    rows = []
    eng.run(ev.evaluate, rows.extend, lambda: len(rows) > one_pass, True)
    assert len(rows) > one_pass, f"refill produced only {len(rows)} <= {one_pass}"
    return len(rows)


if __name__ == "__main__":
    n = test_inproc_matches_python_drive()
    print(f"[ok] in-process driver bit-identical to python drive ({n} rows)")
    m = test_refill_runs_and_emits()
    print(f"[ok] refill ran past completion and stopped ({m} rows)")
    print("PASS")
