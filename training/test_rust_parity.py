"""Differential parity harness: chess_rs (Rust) vs python-chess (the oracle).

python-chess + encode.py ARE the reference. We assert the Rust board agrees on
encoding, legal-move policy-index sets, scalar state, and game outcome at every
ply of many random games, plus hand-built edge positions and the two
repetition behaviours (full-history threefold; search-clone repetition-blind).

Run (from training/, with the venv python):
    python test_rust_parity.py [n_games] [seed]
Gate: zero mismatches; exit code 0.
"""

import sys

import chess
import numpy as np

import chess_rs
from encode import encode_position, move_to_index

MAX_PLIES = 256


def py_outcome_white(pyb):
    oc = pyb.outcome(claim_draw=True)
    if oc is None:
        return None
    if oc.winner is None:
        return 0
    return 1 if oc.winner == chess.WHITE else -1


def py_terminal_value(pyb):
    oc = pyb.outcome(claim_draw=True)
    if oc is None:
        return None
    return 0.0 if oc.winner is None else -1.0


def parity_check(pyb, rb, ctx=""):
    """Assert the two boards agree on everything. Returns {policy_index: pymove}."""
    # 1. encoding byte-identical
    pe = encode_position(pyb)
    re = rb.encode()
    if not np.array_equal(pe, re):
        diff = np.argwhere(pe != re)
        raise AssertionError(
            f"{ctx} ENCODE mismatch fen={pyb.fen()} first_diff={diff[0].tolist()} "
            f"py={pe[tuple(diff[0])]} rs={re[tuple(diff[0])]}")

    # 2. legal-move policy-index sets identical
    py_pairs = {move_to_index(m, pyb): m for m in pyb.legal_moves}
    assert len(py_pairs) == pyb.legal_moves.count(), \
        f"{ctx} move_to_index not injective at fen={pyb.fen()}"
    h_arr, i_arr = rb.legal_moves()
    rs_idx = [int(x) for x in i_arr.tolist()]
    assert len(rs_idx) == len(set(rs_idx)), \
        f"{ctx} rust indices not unique at fen={pyb.fen()}"
    if set(py_pairs) != set(rs_idx):
        only_py = set(py_pairs) - set(rs_idx)
        only_rs = set(rs_idx) - set(py_pairs)
        raise AssertionError(
            f"{ctx} LEGALMOVE index-set mismatch fen={pyb.fen()} "
            f"only_py={sorted(only_py)[:5]} only_rs={sorted(only_rs)[:5]}")

    # 3. scalar state
    assert pyb.turn == rb.turn, f"{ctx} turn fen={pyb.fen()}"
    assert pyb.halfmove_clock == rb.halfmove_clock, \
        f"{ctx} halfmove {pyb.halfmove_clock} vs {rb.halfmove_clock} fen={pyb.fen()}"

    # 4. outcome (claim_draw) — white POV and side-to-move POV
    ow_py, ow_rs = py_outcome_white(pyb), rb.outcome_white()
    assert ow_py == ow_rs, \
        f"{ctx} OUTCOME_WHITE py={ow_py} rs={ow_rs} fen={pyb.fen()}"
    tv_py, tv_rs = py_terminal_value(pyb), rb.terminal_value()
    assert tv_py == tv_rs, \
        f"{ctx} TERMINAL py={tv_py} rs={tv_rs} fen={pyb.fen()}"

    rs_pairs = {int(idx): int(h) for h, idx in zip(h_arr.tolist(), i_arr.tolist())}
    return py_pairs, rs_pairs


def play_random_game(rng):
    """Play one random game in lockstep, checking parity each ply. Returns
    (positions_checked, became_decisive)."""
    pyb = chess.Board()
    rb = chess_rs.Board()
    n = 0
    for ply in range(MAX_PLIES):
        py_pairs, rs_pairs = parity_check(pyb, rb, ctx=f"ply{ply}")
        n += 1
        if pyb.is_game_over(claim_draw=True):
            return n, py_outcome_white(pyb) not in (None, 0)
        idx = int(rng.choice(list(py_pairs.keys())))
        pyb.push(py_pairs[idx])
        rb.push(rs_pairs[idx])
    return n, False


def run_random(n_games, seed):
    rng = np.random.default_rng(seed)
    total = 0
    decisive = 0
    for g in range(n_games):
        pos, dec = play_random_game(rng)
        total += pos
        decisive += int(dec)
        if (g + 1) % 500 == 0:
            print(f"  ...{g + 1} games, {total} positions checked", flush=True)
    print(f"[random] {n_games} games, {total} positions, "
          f"{decisive} decisive — all parity OK", flush=True)
    return total


EDGE_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",   # ep phantom
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",  # ep real
    "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1",     # partial castling rights
    "8/P7/8/8/8/8/8/k1K5 w - - 0 1",          # near promotion / underpromo
    "8/8/8/4k3/8/8/3NK3/8 w - - 0 1",         # K+N vs K -> insufficient
    "2b5/8/8/3k4/8/8/8/5BK1 w - - 0 1",       # KB vs KB same color -> insufficient
    "8/8/8/4k3/8/8/4N3/3NK3 w - - 0 1",       # K+N+N vs K -> NOT insufficient
    "4k3/8/8/8/8/8/8/4K2R w - - 100 100",     # fifty-move claimable (hmc==100)
    "4k3/8/8/8/8/8/8/4K2R w - - 99 99",       # hmc 99: claim via next non-zeroing move
]


def run_edges():
    for fen in EDGE_FENS:
        pyb = chess.Board(fen)
        rb = chess_rs.Board.from_fen(fen)
        parity_check(pyb, rb, ctx="edge")
    print(f"[edge] {len(EDGE_FENS)} positions — all parity OK", flush=True)


# A knight out-and-back cycle returns to the prior position (same placement,
# side, castling, ep): repeating it forces threefold.
CYCLE = ["g1f3", "g8f6", "f3g1", "f6g8"]


def run_threefold_full_history():
    pyb = chess.Board()
    rb = chess_rs.Board()
    claimed_ply = None
    for ply in range(40):
        parity_check(pyb, rb, ctx=f"3fold-ply{ply}")
        if pyb.is_game_over(claim_draw=True):
            claimed_ply = ply
            break
        uci = CYCLE[ply % 4]
        m = chess.Move.from_uci(uci)
        idx = move_to_index(m, pyb)
        h_arr, i_arr = rb.legal_moves()
        rs_pairs = {int(i): int(h) for h, i in zip(h_arr.tolist(), i_arr.tolist())}
        pyb.push(m)
        rb.push(rs_pairs[idx])
    assert claimed_ply is not None, "threefold never fired on the game board"
    assert pyb.outcome(claim_draw=True).winner is None
    assert rb.outcome_white() == 0
    print(f"[threefold] game board claimed draw at ply {claimed_ply} "
          f"(py & rs agree) — OK", flush=True)


def run_search_clone_blind():
    """A clone_search() board must be repetition-blind: even when the game board
    is a threefold draw, the search clone reports ongoing (None)."""
    pyb = chess.Board()
    rb = chess_rs.Board()
    for ply in range(40):
        if rb.outcome_white() == 0 and pyb.is_game_over(claim_draw=True):
            break
        uci = CYCLE[ply % 4]
        m = chess.Move.from_uci(uci)
        idx = move_to_index(m, pyb)
        h_arr, i_arr = rb.legal_moves()
        rs_pairs = {int(i): int(h) for h, i in zip(h_arr.tolist(), i_arr.tolist())}
        pyb.push(m)
        rb.push(rs_pairs[idx])
    assert rb.outcome_white() == 0, "setup: game board should be a threefold draw"
    clone = rb.clone_search()
    assert clone.outcome_white() is None, \
        "search clone must be repetition-blind (got a draw)"
    assert clone.terminal_value() is None
    print("[search-blind] clone_search ignores threefold (matches "
          "copy(stack=False)) — OK", flush=True)


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(f"=== chess_rs parity harness (n_games={n_games}, seed={seed}) ===",
          flush=True)
    run_edges()
    run_threefold_full_history()
    run_search_clone_blind()
    total = run_random(n_games, seed)
    print(f"\nALL PARITY CHECKS PASSED — {total} random positions + edges + "
          f"repetition behaviours.", flush=True)


if __name__ == "__main__":
    main()
