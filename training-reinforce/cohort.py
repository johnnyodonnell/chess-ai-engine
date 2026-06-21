"""Read a self-play cohort file written by the Rust REINFORCE worker.

Row layout, in 4-byte words (ROW_WORDS = INPUT + MASK_WORDS + 2):
  [ state(1152 f32) | legal_mask(146 u32 bitset) | action(u32) | z(f32) ]
The legal mask is packed 1 bit per move (move i -> bit i%32 of word i/32), 32x
smaller than a 4672-f32 mask. File: u32 num_rows (LE), u32 row_words (LE), then
num_rows*row_words 4-byte words LE.

read_cohort returns the mask still packed (`mask_bits`, int32 [n, MASK_WORDS]);
the trainer unpacks it to a float mask on the GPU per micro-batch so the savings
reach RAM and the host->device transfer, not just disk.

Run as a script to sanity-check a cohort:
  python cohort.py /path/to/cohort.bin
"""

import struct
import sys
from pathlib import Path

import numpy as np

# Shared AlphaZero encoder (../training/encode.py). INPUT = 18*8*8, POLICY = 64*73.
sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
from encode import INPUT_CHANNELS, POLICY_SIZE  # noqa: E402

INPUT = INPUT_CHANNELS * 8 * 8  # 1152
MASK_WORDS = (POLICY_SIZE + 31) // 32  # 146
ROW_WORDS = INPUT + MASK_WORDS + 2  # 1300


def read_cohort(path: str) -> dict:
    with open(path, "rb") as f:
        n_rows = struct.unpack("<I", f.read(4))[0]
        row_words = struct.unpack("<I", f.read(4))[0]
        if row_words != ROW_WORDS:
            raise ValueError(f"row_words {row_words} != {ROW_WORDS}")
        data = np.frombuffer(f.read(), dtype="<u4")
    if data.size != n_rows * row_words:
        raise ValueError(f"truncated cohort: {data.size} != {n_rows * row_words}")
    data = data.reshape(n_rows, row_words)
    # Each field is one 4-byte word; reinterpret f32 fields by bit pattern.
    states = np.ascontiguousarray(data[:, :INPUT]).view(np.float32)
    mask_bits = np.ascontiguousarray(data[:, INPUT:INPUT + MASK_WORDS]).view(np.int32)
    actions = data[:, INPUT + MASK_WORDS].astype(np.int64)  # move-index value
    z = np.ascontiguousarray(data[:, INPUT + MASK_WORDS + 1]).view(np.float32)
    return {
        "states": states,
        "mask_bits": mask_bits,
        "actions": actions,
        "z": z,
        "n": n_rows,
    }


def main() -> int:
    c = read_cohort(sys.argv[1])
    a, mb, z, n = c["actions"], c["mask_bits"], c["z"], c["n"]
    # Chosen action must have its bit set in the packed mask.
    chosen_bit = (mb[np.arange(n), a // 32] >> (a % 32)) & 1
    legal_ok = bool(np.all(chosen_bit == 1))
    z_ok = bool(np.all(np.isin(z, [-1.0, 0.0, 1.0])))
    # Legal-move count per row = popcount of the bitset.
    counts = np.unpackbits(mb.view(np.uint8).reshape(n, -1), axis=1).sum(axis=1)
    print(f"rows={n}")
    print(f"  chosen-action-legal: {legal_ok}")
    print(f"  z in {{-1,0,1}}: {z_ok}   z.mean={float(z.mean()):+.3f}")
    print(f"  legal-move count range: [{int(counts.min())},{int(counts.max())}]")
    print(f"  action idx range: [{int(a.min())},{int(a.max())}]")
    ok = legal_ok and z_ok and int(counts.min()) >= 1
    print("COHORT OK" if ok else "COHORT BAD")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
