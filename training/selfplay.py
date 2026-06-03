"""Batched self-play: run N concurrent games sharing a single evaluator for
batched leaf evaluations. Emits training tuples (state, policy_pi, z) into a
replay buffer.

`play_batch` is the pure-CPU/numpy reference loop (used by loop.py via a
LocalEvaluator). `run_selfplay_async` / `run_selfplay_pipeline` are the native
in-process self-play processes the orchestrator spawns (Rust owns the games and
calls back into Python only for the GPU forward)."""

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


def run_selfplay_async(out_queue, stop_event, seed, n_games, sims, weights_path,
                       n_threads, max_batch, device_str="cuda", reload_every=5.0):
    """In-process MULTI-THREADED self-play process (CHESS_BACKEND=rust_async).

    Rust owns the games and calls back into Python ONLY for the in-process GPU
    forward; the native engine runs `n_threads` MCTS worker threads (GIL
    released) over `n_games` game slots, feeding ONE consumer that batches all
    workers' leaves into big forwards (up to `max_batch` rows). The consumer is
    the only thread calling `forward`, so the net + mtime-reload state stay
    single-threaded and race-free.
    """
    import os
    import queue as _queue
    import time

    import torch

    import chess_rs
    from net import ChessNet

    use_cuda = device_str == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    def load_net(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        net = ChessNet(n_blocks=cfg.get("n_blocks"),
                       n_filters=cfg.get("n_filters")).to(device)
        net.load_state_dict(ckpt["weights"])
        net.eval()
        return net

    state = {"net": load_net(weights_path),
             "mtime": os.path.getmtime(weights_path),
             "last_check": time.time()}

    @torch.no_grad()
    def forward(batch):
        now = time.time()
        if now - state["last_check"] >= reload_every:
            state["last_check"] = now
            try:
                m = os.path.getmtime(weights_path)
                if m > state["mtime"]:
                    state["net"] = load_net(weights_path)
                    state["mtime"] = m
            except (OSError, KeyError):
                pass  # mid-write / transient — retry next check
        x = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
        logits, values = state["net"](x)
        return logits.float().cpu().numpy(), values.float().cpu().numpy()

    def sink(rows, n_finished):
        item = (rows, n_finished, len(rows) / max(1, n_finished))
        while not should_stop():
            try:
                out_queue.put(item, timeout=0.5)
                return
            except _queue.Full:
                pass

    def should_stop():
        return stop_event.is_set() or os.getppid() == 1

    engine = chess_rs.AsyncMctsEngine(n_games, sims, seed, True, n_threads, max_batch)
    engine.run(forward, sink, should_stop, True)  # refill=True: run until stop


def run_selfplay_pipeline(out_queue, stop_event, seed, sims, weights_path,
                          n_threads, bucket_size, device_str="cuda", reload_every=5.0):
    """In-process DECOUPLED-PIPELINE self-play process (CHESS_BACKEND=rust_pipeline).

    Like run_selfplay_async, but the native engine never blocks workers on the
    GPU: `n_threads` workers push leaves into two inference buckets and advance
    OTHER games while the single consumer runs batched in-process forwards. The
    number of in-flight games is dynamic (no n_games) and self-regulates to
    ~2*bucket_size; `bucket_size` is the GPU batch width and the one tuning knob.
    Rust calls back into Python for the GPU forward (consumer) and for
    `submit_game` on each finished game (worker). Same trainer contract as
    run_selfplay_async: (rows, n_games, avg_plies) onto the bounded out_queue.
    """
    import os
    import queue as _queue
    import time

    import torch

    import chess_rs
    import zerocopy
    from net import ChessNet

    use_cuda = device_str == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # This self-play subprocess only does inference (the trainer is a separate
    # process), so disable autograd ONCE here instead of entering/exiting a
    # per-forward @torch.no_grad() context on the consumer's hot path.
    torch.set_grad_enabled(False)

    def load_net(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        net = ChessNet(n_blocks=cfg.get("n_blocks"),
                       n_filters=cfg.get("n_filters")).to(device)
        net.load_state_dict(ckpt["weights"])
        net.eval()
        if BF16:
            # Pure bf16 weights (NOT autocast): on this overhead-bound net,
            # autocast's inserted cast ops cost extra launches. Benchmarked ~2.3x
            # on the forward vs fp32 on the compiled path; channels_last adds
            # nothing once compiled (Inductor picks its own layout), so we skip it.
            net = net.bfloat16()
        return net

    # CHESS_COMPILE=1 wraps the net in torch.compile. The net is tiny (4x96), so
    # eager mode is dominated by per-op dispatch overhead; compile fuses the ~17
    # ops into a few kernels. Default mode (max-autotune is unsupported on GB10;
    # CUDA-graphs/reduce-overhead deferred — it constrains input addresses).
    COMPILE = os.environ.get("CHESS_COMPILE", "0") == "1"
    # CHESS_BF16=1 runs the net in bfloat16 (see load_net). bf16 is numerically
    # safe for inference (fp32 exponent range), but breaks bit-reproducibility and
    # can flip argmax on near-tied positions — keep evaluate.py in fp32.
    BF16 = os.environ.get("CHESS_BF16", "0") == "1"

    eager_net = load_net(weights_path)  # persistent module; reloaded IN PLACE
    state = {"net": torch.compile(eager_net) if COMPILE else eager_net,
             "eager": eager_net,
             "mtime": os.path.getmtime(weights_path),
             "last_check": time.time(),
             "warned_shape": False}
    # Zero-copy host I/O ring (GB10/ATS): Rust writes bf16 inputs + reads bf16
    # outputs from pinned host slots the GPU accesses in place, eliminating the
    # per-forward H2D and the blocking D2H. Needs CUDA + bf16; otherwise we keep
    # the legacy numpy path (also used by the routing test). See zerocopy.py.
    N_SLOTS = int(os.environ.get("CHESS_RING_SLOTS", "6"))
    ZEROCOPY = os.environ.get("CHESS_ZEROCOPY", "1") == "1"
    ring = (zerocopy.HostRing(N_SLOTS, bucket_size)
            if (use_cuda and BF16 and ZEROCOPY) else None)
    print(f"rust_pipeline: torch.compile={'ON' if COMPILE else 'OFF'} "
          f"bf16={'ON' if BF16 else 'OFF'} bucket={bucket_size} "
          f"zerocopy_ring={'ON('+str(N_SLOTS)+')' if ring else 'OFF'}", flush=True)

    def reload_weights(path):
        # In-place state_dict swap into the persistent (possibly compiled) module
        # so torch.compile does NOT retrace on every weight publish (~15s). The
        # compiled graph captured these param tensors; load_state_dict copies into
        # them, so fresh weights are picked up without recompiling.
        ckpt = torch.load(path, map_location=device, weights_only=False)
        try:
            state["eager"].load_state_dict(ckpt["weights"])
            state["eager"].eval()
        except RuntimeError:
            # Architecture changed (shouldn't happen within a run): rebuild +
            # recompile rather than crash.
            eager = load_net(path)
            state["eager"] = eager
            state["net"] = torch.compile(eager) if COMPILE else eager

    def maybe_reload():
        now = time.time()
        if now - state["last_check"] >= reload_every:
            state["last_check"] = now
            try:
                m = os.path.getmtime(weights_path)
                if m > state["mtime"]:
                    reload_weights(weights_path)
                    state["mtime"] = m
            except (OSError, KeyError):
                pass  # mid-write / transient — retry next check

    if ring is not None:
        # Zero-copy ring path: Rust passes a slot index + row count; the net runs
        # in place on the slot's pinned bf16 input and writes bf16 outputs into
        # the slot's pinned buffers. The scatter thread calls sync_slot before
        # reading them (replaces the implicit .cpu() sync).
        def forward(slot_idx, m):
            maybe_reload()
            return ring.run(state["net"], slot_idx, m)

        def sync_slot(slot_idx):
            ring.sync(slot_idx)
    else:
        # Legacy numpy path (CPU / bf16-off / routing test): one f32 batch in,
        # (logits, values) f32 numpy out. Unchanged contract.
        def forward(batch):
            maybe_reload()
            if batch.shape[0] != bucket_size and not state["warned_shape"]:
                state["warned_shape"] = True
                print(f"rust_pipeline: forward batch={batch.shape[0]} != "
                      f"bucket={bucket_size}; torch.compile may recompile",
                      flush=True)
            x = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
            if BF16:
                x = x.bfloat16()
            logits, values = state["net"](x)
            return logits.float().cpu().numpy(), values.float().cpu().numpy()

        sync_slot = None

    def submit_game(rows):
        # `rows` is one finished game's [(state, pi, z), ...]. Same trainer
        # contract as the async sink: (rows, n_games, avg_plies).
        item = (rows, 1, len(rows))
        while not should_stop():
            try:
                out_queue.put(item, timeout=0.5)
                return
            except _queue.Full:
                pass

    def should_stop():
        return stop_event.is_set() or os.getppid() == 1

    engine = chess_rs.PipelineEngine(sims, seed, True, n_threads, bucket_size)
    if ring is not None:
        engine.run(forward, submit_game, should_stop, sync_slot,
                   ring.input_ptrs, ring.logit_ptrs, ring.value_ptrs,
                   N_SLOTS, ring.in_stride, ring.logit_stride)
    else:
        engine.run(forward, submit_game, should_stop)
