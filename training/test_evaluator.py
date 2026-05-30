"""Correctness test for the evaluator seam + shared-memory transport.

`RemoteEvaluator` (positions shipped to an `InferenceServer` over shared memory)
must return the same logits/values as `LocalEvaluator` (net.forward in-process)
for identical inputs. The server runs on CPU here so the comparison isolates the
transport from any cuda/cpu numerical drift.
"""

import multiprocessing as mp
import os
import tempfile
import unittest

import numpy as np
import torch

from encode import INPUT_CHANNELS
from evaluator import (
    LocalEvaluator,
    RemoteEvaluator,
    make_channels,
    unlink_blocks,
)
from infer_server import run_server
from net import ChessNet, N_BLOCKS, N_FILTERS


class TestEvaluatorParity(unittest.TestCase):
    def test_remote_matches_local(self):
        ctx = mp.get_context("spawn")
        torch.manual_seed(0)
        net = ChessNet().eval()

        # Publish weights to a temp file for the server to load.
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tmp.close()
        torch.save(
            {"weights": net.state_dict(),
             "config": {"n_blocks": N_BLOCKS, "n_filters": N_FILTERS}},
            tmp.name,
        )

        local = LocalEvaluator(net, torch.device("cpu"))

        max_batch = 16
        channels, blocks = make_channels(ctx, 1, max_batch)
        stop = ctx.Event()
        ready = ctx.Event()
        proc = ctx.Process(
            target=run_server,
            args=(channels, tmp.name, stop, "cpu"),
            kwargs={"ready_event": ready},
        )
        proc.start()
        try:
            self.assertTrue(ready.wait(30), "server failed to start")
            remote = RemoteEvaluator(channels[0])
            rng = np.random.default_rng(0)
            for b in (1, 5, max_batch):
                pos = rng.standard_normal(
                    (b, INPUT_CHANNELS, 8, 8)).astype(np.float32)
                ll, lv = local.evaluate(pos)
                rl, rv = remote.evaluate(pos)
                np.testing.assert_allclose(rl, ll, rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(rv, lv, rtol=1e-5, atol=1e-5)
        finally:
            stop.set()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
            unlink_blocks(blocks)
            os.unlink(tmp.name)


if __name__ == "__main__":
    unittest.main()
