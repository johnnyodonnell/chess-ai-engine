"""Batched self-play: run N concurrent games sharing a single net for
batched leaf evaluations. Emits training tuples (state, policy_pi, z)
into a replay buffer."""

import chess
import numpy as np
import torch

from encode import POLICY_SIZE, encode_position
from mcts import Node, run_simulations, sample_move, visits_to_pi


# Temperature schedule: high exploration for the opening, deterministic after.
TEMPERATURE_OPENING = 1.0
TEMPERATURE_MOVES = 20  # plies of high-temp play
TEMPERATURE_LATE = 0.0
MAX_PLIES = 256


class GameRunner:
    def __init__(self):
        self.board = chess.Board()
        self.root = Node()
        self.history = []  # list of (state_tensor, pi_target, side_to_move)
        self.done = False
        self.result = None  # +1 white wins, -1 black wins, 0 draw

    def temperature(self):
        return (
            TEMPERATURE_OPENING
            if len(self.history) < TEMPERATURE_MOVES
            else TEMPERATURE_LATE
        )


def _finalize(game):
    """Stamp z onto every history position from that side-to-move's POV."""
    out = []
    z_white = game.result
    for state, pi, side_to_move in game.history:
        z = z_white if side_to_move == chess.WHITE else -z_white
        out.append((state, pi, z))
    return out


def play_batch(net, device, n_games, n_sims, rng=None):
    """Run `n_games` self-play games to completion. Returns a flat list of
    (state, pi, z) tuples for the replay buffer, plus per-game stats."""
    rng = rng or np.random
    games = [GameRunner() for _ in range(n_games)]
    results = []
    n_completed = 0
    total_plies = 0

    while any(not g.done for g in games):
        active = [g for g in games if not g.done]
        run_simulations(active, net, device, n_sims, add_root_noise=True)

        for g in active:
            pi = visits_to_pi(g.root, g.board, temperature=g.temperature())
            state = encode_position(g.board)
            g.history.append((state, pi, g.board.turn))
            move = sample_move(g.root, g.board, temperature=g.temperature(), rng=rng)
            g.board.push(move)

            # Advance tree: keep the subtree under the chosen move, discard rest.
            if move in g.root.children:
                g.root = g.root.children[move]
            else:
                g.root = Node()

            if g.board.is_game_over(claim_draw=True) or len(g.history) >= MAX_PLIES:
                outcome = g.board.outcome(claim_draw=True)
                if outcome is None:
                    g.result = 0  # max-plies cutoff = draw
                elif outcome.winner is None:
                    g.result = 0
                elif outcome.winner == chess.WHITE:
                    g.result = 1
                else:
                    g.result = -1
                g.done = True
                results.extend(_finalize(g))
                n_completed += 1
                total_plies += len(g.history)

    return results, {"games": n_completed, "avg_plies": total_plies / max(1, n_completed)}
