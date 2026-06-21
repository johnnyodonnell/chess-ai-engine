"""Read a self-play cohort file written by the Rust REINFORCE worker.

Row layout (ROW_FLOATS = INPUT + POLICY + 2):
  [ state(1152) | legal_mask(4672) | action_index | z ]
File: u32 num_rows (LE), u32 row_floats (LE), then num_rows*row_floats f32 LE.

Run as a script to sanity-check a cohort:
  python cohort.py /path/to/cohort.bin
"""

import struct
import sys
from pathlib import Path

import numpy as np

# Pulled in from the shared AlphaZero encoder (../training/encode.py). Append
# (not insert at 0) so local same-named modules keep priority. INPUT = 18*8*8,
# POLICY = 64*73.
sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
from encode import INPUT_CHANNELS, POLICY_SIZE  # noqa: E402

INPUT = INPUT_CHANNELS * 8 * 8  # 1152
ROW_FLOATS = INPUT + POLICY_SIZE + 2  # 1152 + 4672 + 2 = 5826


def read_cohort(path: str) -> dict:
    with open(path, "rb") as f:
        n_rows = struct.unpack("<I", f.read(4))[0]
        row_floats = struct.unpack("<I", f.read(4))[0]
        if row_floats != ROW_FLOATS:
            raise ValueError(f"row_floats {row_floats} != {ROW_FLOATS}")
        data = np.frombuffer(f.read(), dtype="<f4")
    if data.size != n_rows * row_floats:
        raise ValueError(f"truncated cohort: {data.size} != {n_rows * row_floats}")
    data = data.reshape(n_rows, row_floats)
    return {
        "states": np.ascontiguousarray(data[:, :INPUT]),
        "masks": np.ascontiguousarray(data[:, INPUT:INPUT + POLICY_SIZE]),
        "actions": np.ascontiguousarray(data[:, INPUT + POLICY_SIZE]).astype(np.int64),
        "z": np.ascontiguousarray(data[:, -1]),
        "n": n_rows,
    }


def main() -> int:
    c = read_cohort(sys.argv[1])
    a, m, z = c["actions"], c["masks"], c["z"]
    chosen_mask = m[np.arange(c["n"]), a]
    legal_ok = bool(np.all(chosen_mask == 1.0))
    # Vanilla REINFORCE reward: terminal game outcome from the mover's POV.
    z_ok = bool(np.all(np.isin(z, [-1.0, 0.0, 1.0])))
    mask_vals_ok = bool(np.all(np.isin(m, [0.0, 1.0])))
    legal_counts = m.sum(axis=1)
    print(f"rows={c['n']}")
    print(f"  chosen-action-legal: {legal_ok}")
    print(f"  z in {{-1,0,1}}: {z_ok}   z.mean={float(z.mean()):+.3f}")
    print(f"  mask is 0/1: {mask_vals_ok}")
    print(f"  legal-move count range: [{int(legal_counts.min())},{int(legal_counts.max())}]")
    print(f"  action idx range: [{int(a.min())},{int(a.max())}]")
    ok = legal_ok and z_ok and mask_vals_ok and int(legal_counts.min()) >= 1
    print("COHORT OK" if ok else "COHORT BAD")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
