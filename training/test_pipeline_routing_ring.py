"""Routing-integrity test for the ZERO-COPY RING path of PipelineEngine.

Same guarantee and trick as test_pipeline_routing.py (deterministic eval + cold
move-1 root => every game's first pi is identical unless a result is mis-routed),
but driven through the ring contract: Rust writes each leaf as bf16 directly into
a slot's host input buffer, the `forward(slot_idx, m)` callback reads it and writes
bf16 outputs into the slot's output buffers, and the scatter thread reads them by
pointer. This exercises the new, riskiest code: the per-slot free-list lifecycle
(incl. the deadlock-safe blocking acquire under slot scarcity), the raw-pointer
row math, and the f32<->bf16 round-trip — the silent-corruption surface.

CPU-only + deterministic (no GPU/event needed; sync_slot is a no-op because the
forward writes the outputs synchronously before the slot is handed to scatter).
N_SLOTS is intentionally small to force slot recycling + the blocking acquire.

Run (from training/, dev venv):  python test_pipeline_routing_ring.py
Gate: zero violations; exit 0.
"""
import numpy as np
import torch

import chess_rs

POLICY_SIZE = 4672
ENC = 18 * 8 * 8
SIMS = 32
SEED = 20260601
BUCKET = 48          # > 1 so each forward mixes many distinct games
N_THREADS = 8
TARGET_GAMES = 32
N_SLOTS = 4          # small -> forces recycling + the blocking free-slot acquire


def run_collect(target_games):
    # Host I/O ring (bf16), exactly the buffers Rust writes inputs into / reads
    # outputs from. Plain (non-pinned) host tensors suffice for this CPU test.
    inp = [torch.zeros(BUCKET, 18, 8, 8, dtype=torch.bfloat16) for _ in range(N_SLOTS)]
    log = [torch.zeros(BUCKET, POLICY_SIZE, dtype=torch.bfloat16) for _ in range(N_SLOTS)]
    val = [torch.zeros(BUCKET, dtype=torch.bfloat16) for _ in range(N_SLOTS)]
    input_ptrs = [t.data_ptr() for t in inp]
    logit_ptrs = [t.data_ptr() for t in log]
    value_ptrs = [t.data_ptr() for t in val]

    proj = np.random.default_rng(7).standard_normal(ENC).astype(np.float32)

    def forward(slot_idx, m):
        # Read the bf16 input Rust wrote into this slot, compute a deterministic
        # position-dependent value (uniform logits), write bf16 outputs back.
        b = inp[slot_idx][:m].float().numpy().reshape(m, ENC)
        values = np.tanh(b @ proj).astype(np.float32)
        log[slot_idx][:m].zero_()
        val[slot_idx][:m].copy_(torch.from_numpy(values))
        return slot_idx

    def sync_slot(slot_idx):
        pass  # forward already completed synchronously before scatter reads

    state = {"canonical_first_pi": None, "n_games": 0,
             "first_pi_violations": [], "struct_violations": []}

    def submit_game(rows):
        for state_arr, pi, z in rows:
            p = np.asarray(pi, dtype=np.float32)
            if not (np.isfinite(np.asarray(state_arr)).all() and np.isfinite(p).all()
                    and np.isfinite(z)):
                state["struct_violations"].append(("nonfinite", float(z)))
            psum = float(p.sum())
            if not (0.99 <= psum <= 1.01):
                state["struct_violations"].append(("pi_sum", psum))
            if float(z) not in (-1.0, 0.0, 1.0):
                state["struct_violations"].append(("z", float(z)))
        first_pi = np.asarray(rows[0][1], dtype=np.float32)
        if state["canonical_first_pi"] is None:
            state["canonical_first_pi"] = first_pi
        elif not np.array_equal(first_pi, state["canonical_first_pi"]):
            md = float(np.abs(first_pi - state["canonical_first_pi"]).max())
            state["first_pi_violations"].append(md)
        state["n_games"] += 1

    def stop():
        return state["n_games"] >= target_games

    eng = chess_rs.PipelineEngine(SIMS, SEED, False, N_THREADS, BUCKET)
    eng.run(forward, submit_game, stop, sync_slot,
            input_ptrs, logit_ptrs, value_ptrs, N_SLOTS, ENC, POLICY_SIZE)
    return state


def main():
    s = run_collect(TARGET_GAMES)
    assert s["n_games"] >= TARGET_GAMES, f"only {s['n_games']} games completed"
    assert s["canonical_first_pi"] is not None, "no games produced rows"
    assert not s["first_pi_violations"], (
        f"ROUTING CORRUPTION (ring): {len(s['first_pi_violations'])} games' move-1 "
        f"pi diverged (max abs diffs e.g. {s['first_pi_violations'][:3]})")
    assert not s["struct_violations"], (
        f"structural violations: {s['struct_violations'][:5]}")
    nnz = int((s["canonical_first_pi"] > 0).sum())
    print(f"[ok] ring routing-integrity: {s['n_games']} games, identical cold "
          f"move-1 pi ({nnz} legal moves) across all; structural invariants hold")
    print("PASS")


if __name__ == "__main__":
    main()
