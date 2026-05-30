"""Central GPU inference server for parallel self-play.

This is the ONLY process that holds the net on the GPU and scores leaves.
Self-play workers (torch-free, CPU-only) ship leaf positions over shared-memory
Channels (see evaluator.py); the server gathers every pending request across all
workers into a single batched forward and scatters the results back. Batching
across workers is what keeps the GPU efficient and avoids the per-process
contention that caps independent-worker GPU inference.

The trainer publishes fresh weights to a file; the server reloads them when the
file's mtime changes (workers never need weights — they delegate all scoring).
"""

import os
import time

import numpy as np
import torch

from net import ChessNet


def load_net(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    net = ChessNet(
        n_blocks=cfg.get("n_blocks"), n_filters=cfg.get("n_filters")
    ).to(device)
    net.load_state_dict(ckpt["weights"])
    net.eval()
    return net


def run_server(channels, weights_path, stop_event, device_str="cuda",
               reload_every=5.0, idle_sleep=0.0002, ready_event=None):
    """Serve batched leaf evaluations until `stop_event` is set.

    channels:    list of evaluator.Channel (one per worker)
    weights_path: file the trainer publishes weights to (reloaded on mtime change)
    ready_event: optional mp.Event set once the net is loaded and serving begins
    """
    use_cuda = device_str == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    for ch in channels:
        ch.attach()

    net = load_net(weights_path, device)
    last_mtime = os.path.getmtime(weights_path)
    last_reload_check = time.time()
    if ready_event is not None:
        ready_event.set()

    n = len(channels)
    # Also exit if orphaned (parent hard-killed without setting stop_event).
    while not stop_event.is_set() and os.getppid() != 1:
        # Reload published weights if they changed (throttled, cheap getmtime).
        now = time.time()
        if now - last_reload_check >= reload_every:
            last_reload_check = now
            try:
                m = os.path.getmtime(weights_path)
                if m > last_mtime:
                    net = load_net(weights_path, device)
                    last_mtime = m
            except (OSError, KeyError):
                pass  # mid-write / transient — try again next tick

        # Gather all currently-ready requests across workers.
        ready = []
        for idx in range(n):
            ch = channels[idx]
            if ch.req_ready.is_set():
                cnt = ch.count.value
                ch.req_ready.clear()
                ready.append((idx, cnt))
        if not ready:
            time.sleep(idle_sleep)
            continue

        # Concatenate into one batch, remembering each worker's span.
        total = sum(c for _, c in ready)
        batch = np.empty((total, *channels[0].pos.shape[1:]), dtype=np.float32)
        spans = []
        off = 0
        for idx, cnt in ready:
            batch[off:off + cnt] = channels[idx].pos[:cnt]
            spans.append((idx, off, cnt))
            off += cnt

        with torch.no_grad():
            x = torch.from_numpy(batch).to(device)
            logits, values = net(x)
            logits = logits.float().cpu().numpy()
            values = values.float().cpu().numpy()

        # Scatter results back and signal each waiting worker.
        for idx, off, cnt in spans:
            ch = channels[idx]
            ch.logits[:cnt] = logits[off:off + cnt]
            ch.vals[:cnt] = values[off:off + cnt]
            ch.resp_ready.set()

    for ch in channels:
        ch.close()
