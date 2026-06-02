"""Thread-count invariance: the multi-threaded AsyncMctsEngine (rust_async) must
produce the SAME self-play games regardless of how many worker threads run.

Each game SLOT has its own rng (seeded purely from master_seed + slot_index +
generation) and the forward output for a position is batch-independent (the net
runs in eval() mode; here a deterministic pseudo-evaluator). So a slot's whole
trajectory depends only on its seed + its positions' evals — never on thread
count, scheduling, or how leaves get batched. Therefore running with n_threads=1
vs n_threads=N must yield BYTE-IDENTICAL games.

The sink delivers a flat list of (state, pi, z) rows (no per-game key — the
production contract), and games finish in thread-dependent ORDER, so we compare
the MULTISET of rows: every (state, pi, z) must appear identically across runs.
A mis-routed batch result or a corrupted tree would change some row -> the
multisets diverge -> this test fails. That's exactly the silent-corruption bug
class we care about.

Run (from training/, dev venv):  python test_async_mcts_parity.py
Gate: zero mismatches; exit 0.
"""

import numpy as np

import chess_rs
from test_mcts_parity import DetEval

N_GAMES = 8        # > n_threads so each worker owns several slots (real sharding)
SIMS = 24
SEED = 20260601


def drive_async(seed, n_threads, max_batch=None, n_games=N_GAMES, sims=SIMS):
    """Run AsyncMctsEngine to completion (refill=False) and collect all rows."""
    ev = DetEval()
    eng = chess_rs.AsyncMctsEngine(
        n_games, sims, seed, True, n_threads, max_batch or n_games)
    rows = []
    eng.run(ev.evaluate, lambda rs, n: rows.extend(rs), lambda: False, False)
    return rows


def _row_key(row):
    state, pi, z = row
    return (state.tobytes(), pi.tobytes(), float(z))


def _multiset(rows):
    return sorted(_row_key(r) for r in rows)


def _assert_same_games(a, b, label):
    assert len(a) == len(b), f"{label}: row count differs {len(a)} != {len(b)}"
    assert len(a) > 0, f"{label}: no rows produced"
    ka, kb = _multiset(a), _multiset(b)
    if ka != kb:
        # locate a representative mismatch
        sa, sb = set(ka), set(kb)
        only_a = len(sa - sb)
        only_b = len(sb - sa)
        raise AssertionError(
            f"{label}: row multisets differ "
            f"({only_a} rows only in first run, {only_b} only in second)")


def test_thread_count_invariant():
    """1 worker vs 4 workers -> identical games."""
    a = drive_async(SEED, 1)
    b = drive_async(SEED, 4)
    _assert_same_games(a, b, "n_threads 1 vs 4")
    return len(a)


def test_max_batch_invariant():
    """Batch composition must not affect results: tiny cap vs large cap, same
    thread count -> identical games (drain-to-cap splits batches differently)."""
    a = drive_async(SEED, 4, max_batch=4)
    b = drive_async(SEED, 4, max_batch=512)
    _assert_same_games(a, b, "max_batch 4 vs 512")
    return len(a)


def test_repeatable():
    """Same config twice -> byte-identical (no hidden nondeterminism)."""
    a = drive_async(SEED, 4)
    b = drive_async(SEED, 4)
    _assert_same_games(a, b, "repeat n_threads=4")
    return len(a)


def test_refill_runs_and_emits():
    """refill=True keeps generating past first completion; stop() ends it."""
    one_pass = len(drive_async(SEED, 4))
    ev = DetEval()
    eng = chess_rs.AsyncMctsEngine(N_GAMES, SIMS, SEED, True, 4, N_GAMES)
    rows = []
    eng.run(ev.evaluate, lambda rs, n: rows.extend(rs),
            lambda: len(rows) > one_pass, True)
    assert len(rows) > one_pass, f"refill produced only {len(rows)} <= {one_pass}"
    return len(rows)


if __name__ == "__main__":
    n = test_thread_count_invariant()
    print(f"[ok] thread-count invariant: 1 vs 4 workers identical ({n} rows)")
    m = test_max_batch_invariant()
    print(f"[ok] max-batch invariant: batch composition irrelevant ({m} rows)")
    r = test_repeatable()
    print(f"[ok] repeatable: same seed -> identical ({r} rows)")
    k = test_refill_runs_and_emits()
    print(f"[ok] refill ran past completion and stopped ({k} rows)")
    print("PASS")
