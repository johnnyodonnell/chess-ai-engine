"""MCTS vs no-MCTS, same network (public/models/current.onnx).

Both sides use the exact same weights. "A" runs the full browser MCTS search
(AZAgent, 400 sims). "B" plays the raw policy head: argmax over the softmax of
legal-move logits, no tree search at all. This isolates how much the search adds
on top of the network's instinct.

Mirrors match_az.py: alternating colors, randomized openings, draw adjudication
at the move cap, A's score and the Elo gap (A - B).
"""
import argparse
import random

import chess

from az_agent import AZAgent, RawPolicyAgent
from match import random_opening, elo_from_score


def play_game(agent_a, agent_b, args, a_is_white, rng):
    board = chess.Board()
    random_opening(board, args.open_plies, rng)
    if board.is_game_over():
        return 0.5
    while not board.is_game_over(claim_draw=True):
        if board.fullmove_number > args.max_moves:
            return 0.5
        if board.turn == (chess.WHITE if a_is_white else chess.BLACK):
            mv = agent_a.best_move(board)
        else:
            mv = agent_b.best_move(board)
        board.push(mv)
    outcome = board.outcome(claim_draw=True)
    if outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == (chess.WHITE if a_is_white else chess.BLACK) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../public/models/current.onnx")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--open-plies", type=int, default=2)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    a = AZAgent(args.model, sims=args.sims, intra_threads=args.threads)        # MCTS
    b = RawPolicyAgent(args.model, intra_threads=args.threads)                 # no MCTS

    print(f"A=MCTS(sims={args.sims})  vs  B=raw-policy  | model={args.model} | "
          f"{args.games} games, open_plies={args.open_plies}", flush=True)

    w = d = l = 0
    score = 0.0
    for g in range(args.games):
        a_white = (g % 2 == 0)
        r = play_game(a, b, args, a_white, rng)
        score += r
        if r == 1.0:
            w += 1; res = "W"
        elif r == 0.0:
            l += 1; res = "L"
        else:
            d += 1; res = "D"
        print(f"  game {g+1:3d}: MCTS {'white' if a_white else 'black'} -> {res}  "
              f"(running {w}-{d}-{l}, score {score/(g+1):.3f})", flush=True)

    n = args.games
    s = score / n
    gap, lo, hi = elo_from_score(s, n)
    print("\n=== RESULT ===")
    print(f"MCTS score vs raw-policy: {score}/{n} = {s:.3f}  (W{w} D{d} L{l})")
    print(f"Elo(MCTS) - Elo(raw) = {gap:+.0f}  (95% CI {lo:+.0f} .. {hi:+.0f})")


if __name__ == "__main__":
    main()
