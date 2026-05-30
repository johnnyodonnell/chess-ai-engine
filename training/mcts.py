"""PUCT MCTS over batched parallel games for self-play data generation.

Each call to `run_simulations` advances all N games' trees by `n_sims`
simulations. Leaves from all games are gathered into a single batch per
simulation step and scored through an `Evaluator` (see evaluator.py), which
batches them onto the GPU — the only way to get reasonable throughput out of
one GB10 for tiny nets. This module is pure CPU/numpy (no torch import) so
self-play workers stay lightweight.
"""

import math
import numpy as np

from encode import (
    POLICY_SIZE,
    encode_position,
    legal_move_mask,
    move_to_index,
)

C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25  # mix weight at the root during self-play


class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "expanded", "noised")

    def __init__(self, prior=0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.expanded = False
        self.noised = False

    @property
    def q(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


def _select_child(node):
    """PUCT child selection. Q is stored from the child's side-to-move POV,
    so we negate it: from the parent's POV, a high child-Q is bad."""
    sqrt_parent = math.sqrt(max(node.visit_count, 1))
    best_move, best_child, best_score = None, None, -float("inf")
    for move, child in node.children.items():
        score = -child.q + C_PUCT * child.prior * sqrt_parent / (1 + child.visit_count)
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
    return best_move, best_child


def _terminal_value_for_side_to_move(board):
    """Return +1/0/-1 from the side-to-move's POV if game is over, else None."""
    if not board.is_game_over(claim_draw=True):
        return None
    outcome = board.outcome(claim_draw=True)
    if outcome.winner is None:
        return 0.0
    # If game is over with a winner, the current side-to-move has just been
    # mated (no legal moves), so value is -1 for them.
    return -1.0


def _expand_node(node, board, policy_logits):
    """Mask illegal moves, softmax over legal ones, attach child priors."""
    mask = legal_move_mask(board)
    masked = np.where(mask, policy_logits, -1e9)
    masked -= masked.max()
    exp = np.exp(masked)
    exp = exp * mask
    total = exp.sum()
    priors = exp / total if total > 0 else exp
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        node.children[move] = Node(prior=float(priors[idx]))
    node.expanded = True


def _add_dirichlet_noise(node):
    """Idempotent: noise is applied at most once per node, even if called
    repeatedly (which happens when a subtree is reused across moves)."""
    if node.noised or not node.children:
        return
    moves = list(node.children.keys())
    noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
    for move, n in zip(moves, noise):
        c = node.children[move]
        c.prior = (1 - DIRICHLET_EPS) * c.prior + DIRICHLET_EPS * float(n)
    node.noised = True


def _walk_to_leaf(root, board):
    """Walk PUCT-greedily until we hit an unexpanded node or a terminal."""
    path = [root]
    node = root
    while node.expanded:
        if not node.children:
            break  # terminal (no legal moves) — treated below
        move, child = _select_child(node)
        board.push(move)
        path.append(child)
        node = child
    return path


def _backprop(path, value):
    """Backprop, negating value at each step (zero-sum two-player game)."""
    v = value
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += v
        v = -v


def run_simulations(games, evaluator, n_sims, add_root_noise=False):
    """Advance each game's MCTS tree by `n_sims` simulations, batching leaf
    evaluations across all games through `evaluator` (see evaluator.py)."""
    # Ensure root is expanded for each game (single pass, batched).
    needs_expand = [g for g in games if g.root is not None and not g.root.expanded]
    if needs_expand:
        positions = np.stack([encode_position(g.board) for g in needs_expand])
        logits, _ = evaluator.evaluate(positions)
        for g, logit in zip(needs_expand, logits):
            _expand_node(g.root, g.board, logit)

    # Dirichlet noise at the root for exploration during self-play. The
    # _add_dirichlet_noise helper is idempotent (one noise injection per
    # root node), so it's safe to call here even with subtree reuse.
    if add_root_noise:
        for g in games:
            if g.root is not None and g.root.expanded:
                _add_dirichlet_noise(g.root)

    for _ in range(n_sims):
        leaves = []  # (game, path, leaf_board, terminal_value_or_None)
        for g in games:
            if g.done:
                continue
            board = g.board.copy(stack=False)
            path = _walk_to_leaf(g.root, board)
            tval = _terminal_value_for_side_to_move(board)
            leaves.append((g, path, board, tval))

        # Batch-evaluate non-terminal leaves.
        eval_slots = [(i, lf) for i, lf in enumerate(leaves) if lf[3] is None]
        if eval_slots:
            positions = np.stack([encode_position(lf[2]) for _, lf in eval_slots])
            logits, values = evaluator.evaluate(positions)
        else:
            logits, values = None, None

        eval_idx = 0
        for slot, (g, path, board, tval) in enumerate(leaves):
            if tval is not None:
                v = tval
            else:
                _expand_node(path[-1], board, logits[eval_idx])
                v = float(values[eval_idx])
                eval_idx += 1
            _backprop(path, v)


def visits_to_pi(root, board, temperature=1.0):
    """Convert visit counts at the root to a policy-target distribution."""
    pi = np.zeros(POLICY_SIZE, dtype=np.float32)
    if not root.children:
        return pi
    counts = []
    indices = []
    for move, child in root.children.items():
        counts.append(child.visit_count)
        indices.append(move_to_index(move, board))
    counts = np.array(counts, dtype=np.float64)
    if temperature == 0:
        # Deterministic argmax
        i = int(counts.argmax())
        pi[indices[i]] = 1.0
        return pi
    counts = counts ** (1.0 / temperature)
    total = counts.sum()
    if total <= 0:
        return pi
    probs = counts / total
    for idx, p in zip(indices, probs):
        pi[idx] = float(p)
    return pi


def sample_move(root, board, temperature=1.0, rng=None):
    """Sample a move from the visit-count distribution at the root."""
    moves = list(root.children.keys())
    counts = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)
    if temperature == 0:
        return moves[int(counts.argmax())]
    counts = counts ** (1.0 / temperature)
    counts /= counts.sum()
    rng = rng or np.random
    return moves[int(rng.choice(len(moves), p=counts))]
