"""Arena: play our net (with MCTS) against Stockfish at varying skill
levels. Prints a per-skill-level win/draw/loss summary so we can read
out a rough strength estimate at each milestone.

Stockfish skill levels 0..20; we sample a sub-range. Each side plays
both colors. Move limit is the lesser of `--sims` MCTS visits for our
side or `--sf-movetime-ms` clock for Stockfish."""

import argparse
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch

from encode import encode_position
from mcts import Node, run_simulations, sample_move
from net import ChessNet
from selfplay import MAX_PLIES


def _our_move(net, device, board, sims, rng):
    g = type("G", (), {})()
    g.board = board
    g.root = Node()
    g.done = False
    run_simulations([g], net, device, sims, add_root_noise=False)
    return sample_move(g.root, board, temperature=0.0, rng=rng)


def play_one_game(net, device, stockfish, sims, sf_movetime_ms, our_color, rng):
    board = chess.Board()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
        if board.turn == our_color:
            move = _our_move(net, device, board, sims, rng)
        else:
            res = stockfish.play(board, chess.engine.Limit(time=sf_movetime_ms / 1000.0))
            move = res.move
        board.push(move)
        plies += 1
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0, plies  # draw
    return (1 if outcome.winner == our_color else -1), plies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--skill-levels", default="0,2,4,6")
    ap.add_argument("--games-per-level", type=int, default=8)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--sf-movetime-ms", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg = state.get("config", {})
    from net import N_BLOCKS, N_FILTERS
    net = ChessNet(
        n_blocks=cfg.get("n_blocks", N_BLOCKS),
        n_filters=cfg.get("n_filters", N_FILTERS),
    ).to(device)
    net.load_state_dict(state["weights"])
    net.eval()
    rng = np.random.default_rng(args.seed)

    levels = [int(x) for x in args.skill_levels.split(",")]
    summary = []
    for level in levels:
        sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        sf.configure({"Skill Level": level})
        wins = draws = losses = 0
        t0 = time.time()
        for i in range(args.games_per_level):
            our_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            r, plies = play_one_game(
                net, device, sf, args.sims, args.sf_movetime_ms, our_color, rng,
            )
            if r > 0:
                wins += 1
            elif r < 0:
                losses += 1
            else:
                draws += 1
        sf.quit()
        elapsed = time.time() - t0
        n = args.games_per_level
        score = (wins + 0.5 * draws) / n
        summary.append(
            {
                "skill": level,
                "n": n,
                "w": wins, "d": draws, "l": losses,
                "score": round(score, 3),
                "sec": round(elapsed, 1),
            }
        )
        print(summary[-1], flush=True)

    print("---")
    print({"ckpt": args.ckpt, "summary": summary}, flush=True)


if __name__ == "__main__":
    main()
