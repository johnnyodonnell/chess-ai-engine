"""Routing-integrity test for the rust_pipeline PipelineEngine.

The pipeline routes every NN leaf  game -> inference bucket -> GPU forward -> back
to the SAME game  across an async boundary. A mis-routed result (one game getting
another game's logits/value) is SILENT: training would slowly degrade on corrupted
targets. This guards that bug class WITHOUT relying on deterministic scheduling
(which we dropped together with the old async parity test).

Trick: drive the engine with a DETERMINISTIC eval (uniform logits + a value that
is a fixed function of the position) and DISABLE root noise. Then the move-1
search — a FRESH (cold) root on the standard start position, run for `SIMS`
simulations — is a pure function of that position, identical in every game. Its
policy target `pi` is therefore the SAME in every game... unless a result was
mis-routed into some game's move-1 simulations, corrupting its visit counts and
diverging its first pi. The start position is searched cold in every game, so the
warm-started-subtree path-dependence that makes mid-game positions legitimately
vary does NOT apply to the first move.

Asserts: (1) every game's first policy target is byte-identical (routing intact);
(2) structural invariants on all rows (pi is a distribution, z in {-1,0,1},
finite). Uses BUCKET>1 and several threads so batches genuinely mix many games.

Run (from training/, dev venv):  python test_pipeline_routing.py
Gate: zero violations; exit 0.
"""

import numpy as np

import chess_rs

POLICY_SIZE = 4672
ENC = 18 * 8 * 8
SIMS = 32
SEED = 20260601
BUCKET = 48          # > 1 so each forward mixes many distinct games
N_THREADS = 8
TARGET_GAMES = 32    # enough cold start-position searches to compare


def make_forward():
    """Uniform logits + a deterministic, position-dependent value in [-1, 1].

    The value is a fixed random projection of the encoded position, so it depends
    ONLY on the position (not on batch composition: it's a row-wise gemv), giving a
    deterministic search under a cold root.
    """
    proj = np.random.default_rng(7).standard_normal(ENC).astype(np.float32)

    def forward(batch):
        b = np.ascontiguousarray(batch, dtype=np.float32)
        m = b.shape[0]
        values = np.tanh(b.reshape(m, ENC) @ proj).astype(np.float32)
        logits = np.zeros((m, POLICY_SIZE), dtype=np.float32)
        return logits, values

    return forward


def run_collect(target_games):
    forward = make_forward()
    state = {"canonical_first_pi": None, "n_games": 0, "first_pi_violations": [],
             "struct_violations": []}

    def submit_game(rows):
        # Structural invariants on every recorded position.
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

        # Routing check: the first row is the cold move-1 root on the start
        # position; its pi must be identical across all games.
        first_pi = np.asarray(rows[0][1], dtype=np.float32)
        if state["canonical_first_pi"] is None:
            state["canonical_first_pi"] = first_pi
        elif not np.array_equal(first_pi, state["canonical_first_pi"]):
            max_diff = float(np.abs(first_pi - state["canonical_first_pi"]).max())
            state["first_pi_violations"].append(max_diff)

        state["n_games"] += 1

    def stop():
        return state["n_games"] >= target_games

    eng = chess_rs.PipelineEngine(SIMS, SEED, False, N_THREADS, BUCKET)
    eng.run(forward, submit_game, stop)
    return state


def main():
    s = run_collect(TARGET_GAMES)
    assert s["n_games"] >= TARGET_GAMES, f"only {s['n_games']} games completed"
    assert s["canonical_first_pi"] is not None, "no games produced rows"
    assert not s["first_pi_violations"], (
        f"ROUTING CORRUPTION: {len(s['first_pi_violations'])} games' move-1 pi "
        f"diverged from canonical (max abs diffs e.g. {s['first_pi_violations'][:3]})")
    assert not s["struct_violations"], (
        f"structural violations: {s['struct_violations'][:5]}")
    nnz = int((s["canonical_first_pi"] > 0).sum())
    print(f"[ok] routing-integrity: {s['n_games']} games, identical cold move-1 pi "
          f"({nnz} legal moves) across all; structural invariants hold")
    print("PASS")


if __name__ == "__main__":
    main()
