"""Evaluator abstraction: decouples MCTS from *where* the net runs.

`mcts.run_simulations` calls `evaluator.evaluate(positions) -> (logits, values)`
rather than a torch net directly, so the same MCTS code can run with the net
in-process (single-process self-play via selfplay.play_batch, eval.py, loop.py).

torch is imported lazily inside LocalEvaluator so callers that never evaluate
(e.g. board-only paths) don't load torch/CUDA.
"""

import numpy as np


class Evaluator:
    """positions: float32 (B, INPUT_CHANNELS, 8, 8).
    evaluate() returns (logits float32 (B, POLICY_SIZE), values float32 (B,))."""

    def evaluate(self, positions):
        raise NotImplementedError


class LocalEvaluator(Evaluator):
    """Run the net directly in this process (single-process self-play, eval)."""

    def __init__(self, net, device):
        self.net = net
        self.device = device

    def evaluate(self, positions):
        import torch  # lazy: keep non-evaluating callers torch-free

        with torch.no_grad():
            pos_t = torch.from_numpy(np.ascontiguousarray(positions)).to(self.device)
            logits, values = self.net(pos_t)
            logits = logits.float().cpu().numpy()
            values = values.float().cpu().numpy()
        return logits, values
