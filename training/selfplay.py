"""Batched self-play: run N concurrent games sharing a single evaluator for
batched leaf evaluations. Emits training tuples (state, policy_pi, z)
into a replay buffer. Pure CPU/numpy (no torch import) so it can run in
torch-free self-play workers driving a RemoteEvaluator."""

import chess
import numpy as np

from encode import POLICY_SIZE, encode_position
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
        self.board = chess.Board()
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
        z = z_white if side_to_move == chess.WHITE else -z_white
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


def run_worker(channel, out_queue, stop_event, seed, games_per_worker, sims):
    """Self-play worker process entrypoint (torch-free, CPU-only).

    Generates games via a RemoteEvaluator that ships leaf positions to the
    central InferenceServer over `channel`, and pushes each finished batch —
    (results, n_games, avg_plies) — onto `out_queue` for the trainer to drain.
    The bounded queue applies backpressure: if the trainer falls behind, put()
    blocks here rather than growing memory.
    """
    import os
    import queue as _queue

    from evaluator import RemoteEvaluator

    rng = np.random.default_rng(seed)
    evaluator = RemoteEvaluator(channel)
    # Exit if orphaned (parent hard-killed without setting stop_event).
    while not stop_event.is_set() and os.getppid() != 1:
        results, stats = play_batch(evaluator, games_per_worker, sims, rng=rng)
        item = (results, stats["games"], stats["avg_plies"])
        while not stop_event.is_set():
            try:
                out_queue.put(item, timeout=0.5)
                break
            except _queue.Full:
                pass
