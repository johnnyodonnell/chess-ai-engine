"""Deterministic parity: the Rust MctsEngine vs the Python MCTS (mcts.py).

With Dirichlet noise OFF and an identical *deterministic* pseudo-evaluator
(a pure function of the position encoding), MCTS is fully deterministic, so the
two implementations must build bit-identical trees. We assert exact parity of
per-child root visit counts, value_sums (f64), and visits_to_pi — for single
searches across many positions, and across a multi-move sequence that exercises
subtree reuse.

Both paths use the same Rust `Board` (identical legal-move order + encoding) and
the same f64 softmax (numpy-f64 exp == libm == Rust f64::exp), so "exact" is
genuinely exact. Run (from training/, dev venv):
    python test_mcts_parity.py
Gate: zero mismatches; exit 0.
"""

import hashlib
from types import SimpleNamespace

import numpy as np

import chess_rs
from mcts import Node, run_simulations, visits_to_pi

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",      # start
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",   # after 1.e4
    "r3k2r/pppqbppp/2np1n2/4p3/4P3/2NP1N2/PPPQBPPP/R3K2R w KQkq - 0 1",  # castling-rich
    "4k3/8/8/8/8/8/Q7/4K3 w - - 0 1",                              # KQ vs K (legal)
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",  # Italian
    "8/P7/8/8/8/8/8/k1K5 w - - 0 1",                               # promotion
    "r2q1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",  # midgame
    "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",                         # K+P endgame
    "rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 6",  # QGD
    "2kr3r/pp1q1ppp/2n1pn2/8/3P4/2NBPN2/PPP2PPP/R2Q1RK1 b - - 0 10",  # opposite castle
]


class DetEval:
    """Deterministic pseudo-net: position bytes -> seeded (logits f32, value f32)."""

    def evaluate(self, positions):
        m = positions.shape[0]
        logits = np.empty((m, 4672), dtype=np.float32)
        values = np.empty(m, dtype=np.float32)
        for i in range(m):
            seed = int.from_bytes(
                hashlib.blake2b(positions[i].tobytes(), digest_size=8).digest(), "little")
            r = np.random.default_rng(seed)
            logits[i] = r.standard_normal(4672, dtype=np.float64).astype(np.float32)
            values[i] = np.float32(np.tanh(r.standard_normal()))
        return logits, values


def drive(engine, ev):
    while (b := engine.pending_positions()) is not None:
        lo, va = ev.evaluate(b)
        engine.apply_evals(lo, va)


def py_stats(holder):
    return {c.move_index: (c.visit_count, c.value_sum)
            for c in holder.root.children.values()}


def rs_stats(engine):
    return {mi: (vc, vs) for (mi, vc, vs) in engine.root_child_stats(0)}


def assert_stats_equal(py, rs, ctx):
    assert set(py) == set(rs), f"{ctx}: move-index set differs"
    for mi in py:
        assert py[mi][0] == rs[mi][0], \
            f"{ctx}: visits mi={mi} py={py[mi][0]} rs={rs[mi][0]}"
        assert py[mi][1] == rs[mi][1], \
            f"{ctx}: value_sum mi={mi} py={py[mi][1]!r} rs={rs[mi][1]!r}"


def test_single_search():
    n = 0
    for fen in FENS:
        for sims in (8, 50, 200):
            holder = SimpleNamespace(board=chess_rs.Board.from_fen(fen),
                                     root=Node(), done=False)
            run_simulations([holder], DetEval(), sims, add_root_noise=False)
            eng = chess_rs.MctsEngine.from_fens([fen], sims, 0, False)
            drive(eng, DetEval())
            ctx = f"single fen={fen[:20]} sims={sims}"
            assert_stats_equal(py_stats(holder), rs_stats(eng), ctx)
            for temp in (1.0, 0.5, 0.0):
                pi_py = visits_to_pi(holder.root, holder.board, temp)
                pi_rs = eng.root_pi(0, temp)
                assert np.array_equal(pi_py, pi_rs), f"{ctx} temp={temp}: pi differs"
            n += 1
    print(f"[single-search] {n} (fen,sims) cases — exact parity OK", flush=True)


def test_multi_move_reuse():
    cases = 0
    for fen in FENS:
        sims = 64
        holder = SimpleNamespace(board=chess_rs.Board.from_fen(fen),
                                 root=Node(), done=False)
        eng = chess_rs.MctsEngine.from_fens([fen], sims, 0, False)
        for move_no in range(8):
            run_simulations([holder], DetEval(), sims, add_root_noise=False)
            drive(eng, DetEval())
            py, rs = py_stats(holder), rs_stats(eng)
            assert_stats_equal(py, rs, f"reuse fen={fen[:20]} move={move_no}")
            if not py:
                break
            # deterministic move: highest visit count, ties -> lowest move_index
            best_mi = min(py, key=lambda mi: (-py[mi][0], mi))
            handle = next(h for h, c in holder.root.children.items()
                          if c.move_index == best_mi)
            holder.board.push(handle)
            holder.root = holder.root.children[handle]
            assert eng.advance_by_move_index(0, best_mi), "engine advance failed"
            eng.reset_cycle()
            cases += 1
    print(f"[multi-move-reuse] {cases} reused-search comparisons — exact parity OK",
          flush=True)


def test_multi_game_batch():
    # Drive several games at once through one engine; compare each game's root
    # to an independent single-game Python search (batching must not change math).
    sims = 40
    eng = chess_rs.MctsEngine.from_fens(FENS, sims, 0, False)
    drive(eng, DetEval())
    for gi, fen in enumerate(FENS):
        holder = SimpleNamespace(board=chess_rs.Board.from_fen(fen),
                                 root=Node(), done=False)
        run_simulations([holder], DetEval(), sims, add_root_noise=False)
        assert_stats_equal(py_stats(holder), rs_stats_at(eng, gi),
                           f"batch gi={gi}")
    print(f"[multi-game-batch] {len(FENS)} games in one engine — exact parity OK",
          flush=True)


def rs_stats_at(engine, gi):
    return {mi: (vc, vs) for (mi, vc, vs) in engine.root_child_stats(gi)}


def random_positions(n, seed=0):
    """n legal positions via random python-chess playouts (all cozy-valid)."""
    import chess
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(int(rng.integers(2, 60))):
            moves = list(b.legal_moves)
            if not moves or b.is_game_over(claim_draw=True):
                break
            b.push(moves[int(rng.integers(len(moves)))])
        if not b.is_game_over(claim_draw=True):
            out.append(b.fen())
    return out


def test_random_positions(n=40, sims=128):
    for fen in random_positions(n):
        holder = SimpleNamespace(board=chess_rs.Board.from_fen(fen),
                                 root=Node(), done=False)
        run_simulations([holder], DetEval(), sims, add_root_noise=False)
        eng = chess_rs.MctsEngine.from_fens([fen], sims, 0, False)
        drive(eng, DetEval())
        assert_stats_equal(py_stats(holder), rs_stats(eng), f"rand fen={fen[:24]}")
    print(f"[random] {n} random positions @ {sims} sims — exact parity OK",
          flush=True)


def main():
    print("=== MCTS deterministic parity (Rust engine vs Python mcts.py) ===",
          flush=True)
    test_single_search()
    test_multi_move_reuse()
    test_multi_game_batch()
    test_random_positions()
    print("\nALL MCTS PARITY CHECKS PASSED.", flush=True)


if __name__ == "__main__":
    main()
