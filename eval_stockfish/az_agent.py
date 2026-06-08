"""AlphaZero MCTS agent — a faithful Python port of the web app's engine.

This mirrors src/engine/alphazero/mcts.js exactly (sequential PUCT, C_PUCT=1.5,
softmax over legal moves, leaf value from the tanh value head, pick the
most-visited child with ties broken by q). It reuses training/encode.py for the
position/move encoding so the tensor fed to the net is identical to what the
browser builds.

The goal is to measure the strength of the *exact* model + search that ships in
public/models/current.onnx, so we can anchor its Elo against Stockfish.
"""

import math
import os
import sys

import numpy as np
import onnxruntime as ort

# Reuse the canonical encoder (source of truth that encode.js mirrors).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from encode import encode_position, move_to_index, POLICY_SIZE  # noqa: E402

C_PUCT = 1.5


class _Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "expanded")

    def __init__(self, prior=0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = []  # list of (move, _Node)
        self.expanded = False

    @property
    def q(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class AZAgent:
    def __init__(self, onnx_path, sims=400, time_budget_ms=None, intra_threads=1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.in_name = self.sess.get_inputs()[0].name
        self.sims = sims
        self.time_budget_ms = time_budget_ms
        self.evals = 0  # net forward passes, for benchmarking

    def _evaluate(self, board):
        x = encode_position(board)[None].astype(np.float32)
        policy, value = self.sess.run(None, {self.in_name: x})
        self.evals += 1
        return policy[0], float(np.asarray(value).ravel()[0])

    def _expand(self, node, board):
        logits, value = self._evaluate(board)
        moves = list(board.legal_moves)
        idxs = np.fromiter((move_to_index(m, board) for m in moves), dtype=np.int64,
                           count=len(moves))
        leg = logits[idxs].astype(np.float64)
        if leg.size:
            leg -= leg.max()
            exp = np.exp(leg)
            total = exp.sum()
            priors = exp / total if total > 0 else exp
        else:
            priors = leg
        node.children = [(m, _Node(float(p))) for m, p in zip(moves, priors)]
        node.expanded = True
        return value

    @staticmethod
    def _select_child(node):
        sqrt_parent = math.sqrt(max(node.visit_count, 1))
        best, best_score = None, -math.inf
        for mv, child in node.children:
            score = -child.q + C_PUCT * child.prior * sqrt_parent / (1 + child.visit_count)
            if score > best_score:
                best_score, best = score, (mv, child)
        return best

    @staticmethod
    def _terminal_value(board):
        if not board.is_game_over(claim_draw=False):
            return None
        if board.is_checkmate():
            return -1.0  # side to move was just mated
        return 0.0  # stalemate / insufficient material / etc.

    def best_move(self, board):
        import time
        if board.is_game_over(claim_draw=False):
            return None
        root = _Node()
        self._expand(root, board)
        if len(root.children) == 1:
            return root.children[0][0]

        start = time.monotonic()
        for _ in range(self.sims):
            if self.time_budget_ms is not None and \
                    (time.monotonic() - start) * 1000 > self.time_budget_ms:
                break
            path = [root]
            node = root
            pushed = 0
            while node.expanded and node.children:
                mv, child = self._select_child(node)
                board.push(mv)
                pushed += 1
                path.append(child)
                node = child
                if board.is_game_over(claim_draw=False):
                    break
            tval = self._terminal_value(board)
            value = tval if tval is not None else self._expand(node, board)
            # backprop with sign flip per ply
            v = value
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += v
                v = -v
            for _ in range(pushed):
                board.pop()

        best, best_visits, best_q = None, -1, -math.inf
        for mv, child in root.children:
            if child.visit_count > best_visits or \
                    (child.visit_count == best_visits and child.q > best_q):
                best_visits, best_q, best = child.visit_count, child.q, mv
        return best
