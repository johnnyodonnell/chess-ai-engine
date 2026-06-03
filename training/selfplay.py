"""Batched self-play: run N concurrent games sharing a single evaluator for
batched leaf evaluations. Emits training tuples (state, policy_pi, z) into a
replay buffer.

`play_batch` is the pure-CPU/numpy reference loop (used by loop.py via a
LocalEvaluator). Production self-play now runs in the standalone Rust worker
(`selfplay_rs`); the orchestrator spawns it as a subprocess and reads finished
games over a frame stream — there is no in-process Python self-play anymore."""

import numpy as np

from chess_backend import make_board
from mcts import Node, run_simulations, sample_move, visits_to_pi


# Temperature schedule: full exploration in the opening, then ANNEAL toward a
# nonzero floor — never fully deterministic. A hard cut to 0 made both sides
# play argmax of a near-uniform policy and shuffle into threefold-repetition
# draws every game (z=0 everywhere → no learning signal: the draw collapse).
# Keeping temperature > 0 throughout keeps games decisive and learnable.
TEMPERATURE_OPENING = 1.0
TEMPERATURE_MOVES = 20         # plies of full-temperature (1.0) opening play
TEMPERATURE_FLOOR = 0.35       # self-play never goes below this (never argmax)
TEMPERATURE_ANNEAL_MOVES = 40  # plies to anneal from opening temp down to floor
MAX_PLIES = 256


class GameRunner:
    def __init__(self):
        self.board = make_board()
        self.root = Node()
        self.history = []  # list of (state_tensor, pi_target, side_to_move)
        self.done = False
        self.result = None  # +1 white wins, -1 black wins, 0 draw

    def temperature(self):
        ply = len(self.history)
        if ply < TEMPERATURE_MOVES:
            return TEMPERATURE_OPENING
        frac = min(1.0, (ply - TEMPERATURE_MOVES) / TEMPERATURE_ANNEAL_MOVES)
        return TEMPERATURE_OPENING + frac * (TEMPERATURE_FLOOR - TEMPERATURE_OPENING)


def _finalize(game):
    """Stamp z onto every history position from that side-to-move's POV."""
    out = []
    z_white = game.result
    for state, pi, side_to_move in game.history:
        # side_to_move is the board's `turn` flag (True == white).
        z = z_white if side_to_move else -z_white
        out.append((state, pi, z))
    return out


def play_batch(evaluator, n_games, n_sims, rng=None):
    """Run `n_games` self-play games to completion. Returns a flat list of
    (state, pi, z) tuples for the replay buffer, plus per-game stats.
    Leaf evaluations are batched across all active games through `evaluator`."""
    rng = rng or np.random
    games = [GameRunner() for _ in range(n_games)]
    results = []
    n_completed = 0
    total_plies = 0

    while any(not g.done for g in games):
        active = [g for g in games if not g.done]
        run_simulations(active, evaluator, n_sims, add_root_noise=True)

        for g in active:
            pi = visits_to_pi(g.root, g.board, temperature=g.temperature())
            state = g.board.encode()
            g.history.append((state, pi, g.board.turn))
            move = sample_move(g.root, g.board, temperature=g.temperature(), rng=rng)
            g.board.push(move)

            # Advance tree: keep the subtree under the chosen move, discard rest.
            if move in g.root.children:
                g.root = g.root.children[move]
            else:
                g.root = Node()

            res = g.board.outcome_white()
            if res is not None or len(g.history) >= MAX_PLIES:
                g.result = res if res is not None else 0  # max-plies cutoff = draw
                g.done = True
                results.extend(_finalize(g))
                n_completed += 1
                total_plies += len(g.history)

    return results, {"games": n_completed, "avg_plies": total_plies / max(1, n_completed)}


