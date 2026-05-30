"""Evaluator abstraction: decouples MCTS from *where* the net runs.

`mcts.run_simulations` no longer calls a torch net directly; it calls
`evaluator.evaluate(positions) -> (logits, values)`. This lets the same MCTS
code run three ways:

  - LocalEvaluator: net.forward in-process (single-process self-play, eval.py,
    legacy loop.py modes).
  - RemoteEvaluator: ship positions to a central InferenceServer that batches
    leaf evaluations across many CPU-only self-play workers onto one GPU
    (see infer_server.py). Added alongside the server.

torch is imported lazily inside LocalEvaluator so that self-play workers — which
only ever use a RemoteEvaluator — never load torch/CUDA.
"""

import os
import time
from multiprocessing import shared_memory

import numpy as np

from encode import INPUT_CHANNELS, POLICY_SIZE

_POS_SHAPE = (INPUT_CHANNELS, 8, 8)


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
        import torch  # lazy: keep workers torch-free

        with torch.no_grad():
            pos_t = torch.from_numpy(np.ascontiguousarray(positions)).to(self.device)
            logits, values = self.net(pos_t)
            logits = logits.float().cpu().numpy()
            values = values.float().cpu().numpy()
        return logits, values


# ---------------------------------------------------------------------------
# Shared-memory transport: self-play workers <-> central InferenceServer
# ---------------------------------------------------------------------------
class Channel:
    """A shared-memory request/response slot for one self-play worker.

    Holds only picklable meta (shm block names + mp sync primitives) so it can
    be passed to spawned worker/server processes. The numpy views over the
    shared blocks are created lazily by `attach()`, which MUST run in whichever
    process actually uses the channel (shm can't be attached pre-spawn).

    Protocol (one outstanding request per worker — a strict ping-pong):
      worker: write pos+count -> clear resp_ready -> set req_ready -> wait resp_ready
      server: see req_ready -> read pos+count -> clear req_ready -> ... ->
              write logits+vals -> set resp_ready
    """

    _RUNTIME_ATTRS = ("_pos_shm", "_logit_shm", "_val_shm", "pos", "logits", "vals")

    def __init__(self, max_batch, pos_name, logit_name, val_name,
                 count, req_ready, resp_ready):
        self.max_batch = max_batch
        self.pos_name = pos_name
        self.logit_name = logit_name
        self.val_name = val_name
        self.count = count            # ctx.Value("i")
        self.req_ready = req_ready    # ctx.Event
        self.resp_ready = resp_ready  # ctx.Event
        self._attached = False

    def attach(self):
        if self._attached:
            return
        self._pos_shm = shared_memory.SharedMemory(name=self.pos_name)
        self._logit_shm = shared_memory.SharedMemory(name=self.logit_name)
        self._val_shm = shared_memory.SharedMemory(name=self.val_name)
        self.pos = np.ndarray((self.max_batch, *_POS_SHAPE),
                              dtype=np.float32, buffer=self._pos_shm.buf)
        self.logits = np.ndarray((self.max_batch, POLICY_SIZE),
                                 dtype=np.float32, buffer=self._logit_shm.buf)
        self.vals = np.ndarray((self.max_batch,),
                               dtype=np.float32, buffer=self._val_shm.buf)
        self._attached = True

    def close(self):
        if self._attached:
            self._pos_shm.close()
            self._logit_shm.close()
            self._val_shm.close()
            self._attached = False

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items()
                if k not in self._RUNTIME_ATTRS}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._attached = False


def make_channels(ctx, n_workers, max_batch):
    """Create `n_workers` shared-memory channels in the current (parent)
    process. Returns (channels, shm_blocks). The caller MUST keep shm_blocks
    referenced for the lifetime of the run and unlink() them at shutdown."""
    pos_bytes = max_batch * INPUT_CHANNELS * 8 * 8 * 4
    logit_bytes = max_batch * POLICY_SIZE * 4
    val_bytes = max_batch * 4
    channels, blocks = [], []
    for _ in range(n_workers):
        pos = shared_memory.SharedMemory(create=True, size=pos_bytes)
        logit = shared_memory.SharedMemory(create=True, size=logit_bytes)
        val = shared_memory.SharedMemory(create=True, size=val_bytes)
        blocks += [pos, logit, val]
        channels.append(Channel(
            max_batch, pos.name, logit.name, val.name,
            ctx.Value("i", 0), ctx.Event(), ctx.Event(),
        ))
    return channels, blocks


def unlink_blocks(blocks):
    """Release shared-memory blocks created by make_channels (call at shutdown)."""
    for b in blocks:
        try:
            b.close()
            b.unlink()
        except FileNotFoundError:
            pass


class RemoteEvaluator(Evaluator):
    """Ship positions to the InferenceServer over a shared-memory Channel and
    block for the reply. Used by torch-free, CPU-only self-play workers."""

    def __init__(self, channel, timeout=120.0):
        self.ch = channel
        self.timeout = timeout
        self.ch.attach()

    def evaluate(self, positions):
        n = positions.shape[0]
        if n > self.ch.max_batch:
            raise ValueError(
                f"batch {n} exceeds channel max_batch {self.ch.max_batch}")
        self.ch.pos[:n] = positions
        self.ch.count.value = n
        self.ch.resp_ready.clear()
        self.ch.req_ready.set()
        # Wait in short slices so an orphaned worker (dead parent/server) exits
        # promptly instead of blocking for the full timeout.
        deadline = time.time() + self.timeout
        while not self.ch.resp_ready.wait(0.5):
            if os.getppid() == 1:
                raise SystemExit("orphaned self-play worker: parent gone")
            if time.time() > deadline:
                raise TimeoutError("inference server did not respond within "
                                   f"{self.timeout}s")
        return self.ch.logits[:n].copy(), self.ch.vals[:n].copy()
