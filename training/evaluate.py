"""Self-play evaluation, Elo rating, and model serving.

Domain-general, purely self-play: no rules engines, no external reference.
Each invocation evaluates one candidate snapshot against the active pool
(3 top performers + a fixed random anchor + 4 frozen ex-snapshot anchors
spread across the strength range), refits a lightweight Elo over all
accumulated match results (random pinned at 0), updates the pool, and
promotes the highest-rated model to `<run-dir>/best.onnx` if it clears the
current served model by a confidence margin.

Run standalone:
    python evaluate.py --run-dir runs/run1 --candidate runs/run1/snapshots/<snap>.pt

`loop.py` launches it as a detached subprocess after each snapshot.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import chess
import numpy as np
import torch

from evaluator import LocalEvaluator
from export import export
from mcts import Node, run_simulations, sample_move
from net import ChessNet
from selfplay import MAX_PLIES

RANDOM = "random"  # reserved name for the fixed floor anchor (Elo pinned at 0)
OPENING_TEMP = 1.0  # temperature for the opening plies, for game diversity


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------
class RandomPlayer:
    name = RANDOM

    def select_moves(self, boards, plies, opening_plies, rng):
        out = []
        for b in boards:
            moves = list(b.legal_moves)
            out.append(moves[int(rng.integers(len(moves)))])
        return out


class NetPlayer:
    def __init__(self, name, weights_path, sims, device):
        self.name = name
        self.sims = sims
        self.device = device
        state = torch.load(weights_path, map_location=device, weights_only=False)
        cfg = state.get("config", {})
        self.net = ChessNet(
            n_blocks=cfg.get("n_blocks"), n_filters=cfg.get("n_filters")
        ).to(device)
        self.net.load_state_dict(state["weights"])
        self.net.eval()
        self.evaluator = LocalEvaluator(self.net, device)

    def select_moves(self, boards, plies, opening_plies, rng):
        # Fresh tree per move (no reuse) — simpler and fine for eval volumes.
        holders = [
            SimpleNamespace(board=b, root=Node(), done=False) for b in boards
        ]
        run_simulations(holders, self.evaluator, self.sims, add_root_noise=False)
        out = []
        for h, ply in zip(holders, plies):
            temp = OPENING_TEMP if ply < opening_plies else 0.0
            out.append(sample_move(h.root, h.board, temperature=temp, rng=rng))
        return out


# ---------------------------------------------------------------------------
# Match play
# ---------------------------------------------------------------------------
def _result_white(board, ply):
    """+1 white win, -1 black win, 0 draw (incl. max-plies cutoff)."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0
    return 1 if outcome.winner == chess.WHITE else -1


def play_match(player_a, player_b, n_games, opening_plies, rng):
    """Play `n_games` between two players, alternating colors. The first
    `opening_plies` are sampled with temperature for diversity (otherwise
    net-vs-net would be one deterministic game repeated). Net evaluations are
    batched across all games where the same player is to move.

    Returns (wins, draws, losses) from player_a's perspective.
    """
    games = [
        SimpleNamespace(board=chess.Board(), a_is_white=(i % 2 == 0),
                        done=False, ply=0, rw=0)
        for i in range(n_games)
    ]

    while any(not g.done for g in games):
        a_turn, b_turn = [], []
        for g in games:
            if g.done:
                continue
            if (g.board.turn == chess.WHITE) == g.a_is_white:
                a_turn.append(g)
            else:
                b_turn.append(g)

        for player, group in ((player_a, a_turn), (player_b, b_turn)):
            if not group:
                continue
            moves = player.select_moves(
                [g.board for g in group], [g.ply for g in group], opening_plies, rng
            )
            for g, mv in zip(group, moves):
                g.board.push(mv)
                g.ply += 1
                if g.board.is_game_over(claim_draw=True) or g.ply >= MAX_PLIES:
                    g.rw = _result_white(g.board, g.ply)
                    g.done = True

    wins = draws = losses = 0
    for g in games:
        a = g.rw if g.a_is_white else -g.rw
        if a > 0:
            wins += 1
        elif a < 0:
            losses += 1
        else:
            draws += 1
    return wins, draws, losses


# ---------------------------------------------------------------------------
# Elo fit (Bradley-Terry, coordinate-Newton; random pinned, L2-regularized)
# ---------------------------------------------------------------------------
def fit_elo(names, games, fixed=None, reg=1e-4, iters=400):
    """`games`: list of (a, b, score_a, n) — in n games between a and b, a
    scored score_a points (win=1, draw=0.5). Returns {name: rating}.

    random (or any name in `fixed`) is held at its fixed rating to pin the
    scale; reg pulls ratings toward 0 so a 100% sweep stays finite.
    """
    fixed = fixed or {RANDOM: 0.0}
    R = {n: fixed.get(n, 0.0) for n in names}
    q = math.log(10) / 400.0

    adj = {n: [] for n in names}
    for a, b, score_a, n in games:
        if n <= 0:
            continue
        adj[a].append((b, score_a, n))
        adj[b].append((a, n - score_a, n))

    for _ in range(iters):
        for p in names:
            if p in fixed:
                continue
            g = h = 0.0
            for opp, score_p, n in adj[p]:
                e = 1.0 / (1.0 + 10 ** ((R[opp] - R[p]) / 400.0))
                g += q * (score_p - n * e)
                h += q * q * n * e * (1.0 - e)
            g -= reg * R[p]
            h += reg
            if h > 1e-12:
                R[p] += g / h
    for n, v in fixed.items():
        R[n] = v
    return R


# ---------------------------------------------------------------------------
# Pool / serving
# ---------------------------------------------------------------------------
def _load_pool(path):
    if path.exists():
        return json.loads(path.read_text())
    return {"served": None, "served_rating": None, "models": {}, "results": []}


def _save_pool(path, pool):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(pool, indent=2))
    tmp.replace(path)


def _select_anchors(ratings, models, n_top, n_anchors):
    """Active opponents = top `n_top` rated + random + `n_anchors` frozen
    snapshots whose ratings most evenly cover (0, rating of the n_top-th)."""
    ranked = sorted(
        (m for m in models if m != RANDOM),
        key=lambda m: ratings.get(m, 0.0),
        reverse=True,
    )
    top = ranked[:n_top]
    active = list(top) + [RANDOM]

    below = [m for m in ranked[n_top:]]
    if below:
        ceiling = ratings.get(top[-1], 0.0) if top else max(
            (ratings.get(m, 0.0) for m in below), default=0.0
        )
        # Even target rungs across (0, ceiling); pick the closest unused model.
        k = min(n_anchors, len(below))
        targets = [ceiling * (i + 1) / (k + 1) for i in range(k)]
        chosen = []
        pool = list(below)
        for t in targets:
            best = min(pool, key=lambda m: abs(ratings.get(m, 0.0) - t))
            chosen.append(best)
            pool.remove(best)
        active += chosen
    return active, top


def evaluate(run_dir, candidate, games_per_pair, sims, opening_plies,
             n_top, n_anchors, serve_margin, seed):
    run_dir = Path(run_dir)
    pool_path = run_dir / "pool.json"
    pool = _load_pool(pool_path)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cand_pt = Path(candidate)
    cand_name = cand_pt.stem
    cand_onnx = cand_pt.with_suffix(".onnx")
    print(f"[eval] candidate={cand_name} device={device}", flush=True)

    # Register candidate (paths stored relative to run-dir for portability).
    pool["models"][cand_name] = {
        "pt": str(cand_pt.relative_to(run_dir)) if cand_pt.is_relative_to(run_dir) else str(cand_pt),
        "onnx": str(cand_onnx.relative_to(run_dir)) if cand_onnx.is_relative_to(run_dir) else str(cand_onnx),
        "rating": None,
    }
    pool["models"].setdefault(RANDOM, {"pt": None, "onnx": None, "rating": 0.0})

    # Pick the active opponent set from prior ratings (top-3 + random + anchors).
    prior = {m: (pool["models"][m].get("rating") or 0.0) for m in pool["models"]}
    active, _ = _select_anchors(prior, list(pool["models"]), n_top, n_anchors)
    opponents = [m for m in active if m != cand_name]
    if not opponents:
        opponents = [RANDOM]
    print(f"[eval] opponents={opponents}", flush=True)

    cand_player = NetPlayer(cand_name, cand_pt, sims, device)

    def make_player(name):
        if name == RANDOM:
            return RandomPlayer()
        pt = run_dir / pool["models"][name]["pt"]
        return NetPlayer(name, pt, sims, device)

    cand_vs_random = None
    for opp in opponents:
        w, d, l = play_match(
            cand_player, make_player(opp), games_per_pair, opening_plies, rng
        )
        score_a = w + 0.5 * d
        pool["results"].append(
            {"a": cand_name, "b": opp, "score_a": score_a, "n": w + d + l}
        )
        if opp == RANDOM:
            cand_vs_random = score_a / max(1, w + d + l)
        print(f"[eval] {cand_name} vs {opp}: W{w} D{d} L{l}", flush=True)

    # Refit Elo over all accumulated results.
    names = list(pool["models"])
    games = [(r["a"], r["b"], r["score_a"], r["n"]) for r in pool["results"]]
    ratings = fit_elo(names, games)
    for m in names:
        pool["models"][m]["rating"] = round(ratings.get(m, 0.0), 1)

    # Recompute active set for the next round and decide serving.
    _, top = _select_anchors(ratings, names, n_top, n_anchors)
    leader = top[0] if top else None
    pool["top"] = top
    print(f"[eval] ratings: " + ", ".join(
        f"{m}={pool['models'][m]['rating']}" for m in
        sorted(names, key=lambda m: ratings.get(m, 0.0), reverse=True)
    ), flush=True)

    served = pool.get("served")
    # Compare against the served model's CURRENT refit rating, not the frozen
    # served_rating: every refit drifts the whole scale, so a stale threshold
    # becomes unbeatable and best.onnx would stick forever. served_rating is
    # kept as an informational stamp of the rating at promotion time.
    served_now = ratings.get(served) if served else None
    sane = (cand_vs_random is None) or (cand_vs_random >= 0.6)
    if leader and sane:
        lead_r = ratings[leader]
        if served is None or lead_r > (served_now or -1e9) + serve_margin:
            _serve(run_dir, pool, leader)
            pool["served"], pool["served_rating"] = leader, round(lead_r, 1)
            print(f"[eval] SERVED {leader} (rating={lead_r:.1f})", flush=True)
        else:
            print(f"[eval] kept {served} (rating={served_now:.1f}); "
                  f"leader {leader}={lead_r:.1f} within margin", flush=True)
    elif not sane:
        print(f"[eval] candidate failed random sanity gate "
              f"(score vs random={cand_vs_random:.2f}); not serving", flush=True)

    _save_pool(pool_path, pool)
    print("[eval] done", flush=True)


def _serve(run_dir, pool, name):
    """Copy the model's onnx (+pt) to best.onnx/best.pt, exporting if needed."""
    info = pool["models"][name]
    pt = run_dir / info["pt"]
    onnx = run_dir / info["onnx"]
    if not onnx.exists():
        export(str(pt), str(onnx))
    shutil.copyfile(onnx, run_dir / "best.onnx")
    shutil.copyfile(pt, run_dir / "best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--candidate", required=True, help="path to the snapshot .pt")
    ap.add_argument("--games-per-pair", type=int, default=100)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--opening-plies", type=int, default=10)
    ap.add_argument("--n-top", type=int, default=3)
    ap.add_argument("--n-anchors", type=int, default=4)
    ap.add_argument("--serve-margin", type=float, default=35.0,
                    help="Elo the leader must beat the served model by to be "
                         "promoted (≈ rating standard error at 100 games).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    evaluate(
        args.run_dir, args.candidate, args.games_per_pair, args.sims,
        args.opening_plies, args.n_top, args.n_anchors, args.serve_margin,
        args.seed,
    )


if __name__ == "__main__":
    main()
