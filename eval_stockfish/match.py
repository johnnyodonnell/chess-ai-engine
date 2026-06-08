"""Play the web-app AlphaZero model against Stockfish and estimate its Elo.

The model + search is identical to what ships in the browser (az_agent.py).
Stockfish is the absolute anchor: we run it at a fixed strength setting, play a
match with alternating colors and randomized openings, then convert the score
to an Elo gap with a confidence interval.

Usage:
  python match.py --games 60 --sf-elo 1320
  python match.py --games 40 --sf-skill 3 --sf-movetime 0.05
"""

import argparse
import math
import random
import sys

import chess
import chess.engine

from az_agent import AZAgent, RawPolicyAgent

SF_PATH = __import__("os").path.join(__import__("os").path.dirname(__file__), "bin", "stockfish")


def make_stockfish(args):
    eng = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    opts = {"Threads": 1, "Hash": 64}
    if args.sf_elo is not None:
        opts["UCI_LimitStrength"] = True
        opts["UCI_Elo"] = args.sf_elo
    if args.sf_skill is not None:
        opts["Skill Level"] = args.sf_skill
    eng.configure(opts)
    return eng


def sf_limit(args):
    if args.sf_nodes is not None:
        return chess.engine.Limit(nodes=args.sf_nodes)
    return chess.engine.Limit(time=args.sf_movetime)


def random_opening(board, plies, rng):
    """Apply `plies` random legal half-moves to diversify the opening."""
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            break
        board.push(rng.choice(moves))


def play_game(agent, sf, args, az_is_white, rng):
    board = chess.Board()
    random_opening(board, args.open_plies, rng)
    if board.is_game_over():  # rare: random opening ended it
        return 0.5

    limit = sf_limit(args)
    while not board.is_game_over(claim_draw=True):
        if board.fullmove_number > args.max_moves:
            return 0.5  # adjudicate overlong game as a draw
        if board.turn == (chess.WHITE if az_is_white else chess.BLACK):
            mv = agent.best_move(board)
        else:
            mv = sf.play(board, limit).move
        board.push(mv)

    outcome = board.outcome(claim_draw=True)
    if outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == (chess.WHITE if az_is_white else chess.BLACK) else 0.0


def elo_from_score(score, n):
    """Elo gap (AZ - opponent) from match score, with a ~95% CI via the
    normal approximation on the per-game score variance."""
    eps = 1.0 / (2 * n)
    p = min(max(score, eps), 1 - eps)
    gap = -400.0 * math.log10(1.0 / p - 1.0)
    # stderr of mean score; map +/-1.96 sigma through the same transform
    var = max(score * (1 - score), eps) / n
    se = math.sqrt(var)
    lo_p = min(max(score - 1.96 * se, eps), 1 - eps)
    hi_p = min(max(score + 1.96 * se, eps), 1 - eps)
    lo = -400.0 * math.log10(1.0 / lo_p - 1.0)
    hi = -400.0 * math.log10(1.0 / hi_p - 1.0)
    return gap, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--no-mcts", action="store_true",
                    help="play the raw policy head (argmax, no search)")
    ap.add_argument("--model", default="../public/models/current.onnx")
    ap.add_argument("--sf-elo", type=int, default=None)
    ap.add_argument("--sf-skill", type=int, default=None)
    ap.add_argument("--sf-movetime", type=float, default=0.1)
    ap.add_argument("--sf-nodes", type=int, default=None)
    ap.add_argument("--open-plies", type=int, default=2)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.no_mcts:
        agent = RawPolicyAgent(args.model, intra_threads=args.threads)
    else:
        agent = AZAgent(args.model, sims=args.sims, intra_threads=args.threads)
    sf = make_stockfish(args)

    anchor = (f"UCI_Elo={args.sf_elo}" if args.sf_elo is not None
              else f"Skill={args.sf_skill}")
    limdesc = (f"nodes={args.sf_nodes}" if args.sf_nodes is not None
               else f"movetime={args.sf_movetime}s")
    azdesc = "raw-policy" if args.no_mcts else f"sims={args.sims}"
    print(f"AZ({azdesc}) vs Stockfish 17.1 [{anchor}, {limdesc}] | "
          f"{args.games} games, open_plies={args.open_plies}", flush=True)

    w = d = l = 0
    score = 0.0
    for g in range(args.games):
        az_white = (g % 2 == 0)
        r = play_game(agent, sf, args, az_white, rng)
        score += r
        if r == 1.0:
            w += 1; tag = "W"
        elif r == 0.0:
            l += 1; tag = "L"
        else:
            d += 1; tag = "D"
        print(f"  game {g+1:3d}: AZ {'white' if az_white else 'black'} -> {tag}"
              f"  (running {w}-{d}-{l}, score {score/(g+1):.3f})", flush=True)

    sf.quit()
    n = args.games
    s = score / n
    gap, lo, hi = elo_from_score(s, n)
    print("\n=== RESULT ===")
    print(f"AZ score: {score}/{n} = {s:.3f}  (W{w} D{d} L{l})")
    print(f"Elo(AZ) - Elo(SF) = {gap:+.0f}  (95% CI {lo:+.0f} .. {hi:+.0f})")
    if args.sf_elo is not None:
        print(f"=> AZ Elo ~ {args.sf_elo + gap:.0f}  "
              f"(95% CI {args.sf_elo + lo:.0f} .. {args.sf_elo + hi:.0f})")


if __name__ == "__main__":
    main()
