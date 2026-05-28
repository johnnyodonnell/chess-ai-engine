"""Baseline ELO check: play a uniform-random move policy against Stockfish
at its weakest settings, both via Skill Level and UCI_Elo. The expectation
is 0/N — random is well below Stockfish's floor — but we record the
specifics for the README."""

import argparse
import random
import time

import chess
import chess.engine


def random_move(board, rng):
    moves = list(board.legal_moves)
    return rng.choice(moves) if moves else None


def play_one(engine_us, engine_them, sf_limit, our_color, rng):
    board = chess.Board()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < 256:
        if board.turn == our_color:
            move = random_move(board, rng)
        else:
            res = engine_them.play(board, sf_limit)
            move = res.move
        board.push(move)
        plies += 1
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0, plies
    return (1 if outcome.winner == our_color else -1), plies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--games-per-setting", type=int, default=16)
    ap.add_argument("--sf-movetime-ms", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    settings = [
        ("uci_elo=1320", {"UCI_LimitStrength": True, "UCI_Elo": 1320}),
        ("skill=0",      {"Skill Level": 0}),
        ("skill=5",      {"Skill Level": 5}),
    ]
    limit = chess.engine.Limit(time=args.sf_movetime_ms / 1000.0)

    for name, opts in settings:
        sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        sf.configure(opts)
        wins = draws = losses = 0
        total_plies = 0
        t0 = time.time()
        for i in range(args.games_per_setting):
            our_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            r, plies = play_one(None, sf, limit, our_color, rng)
            total_plies += plies
            if r > 0: wins += 1
            elif r < 0: losses += 1
            else: draws += 1
        sf.quit()
        n = args.games_per_setting
        score = (wins + 0.5 * draws) / n
        print(
            f"opponent={name:18s} n={n:3d} W={wins} D={draws} L={losses} "
            f"score={score:.3f} avg_plies={total_plies/n:.1f} "
            f"elapsed={time.time()-t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
