"""Generate the forward-parity fixture for the Rust `forward-check` gate.

Writes, into the given dir:
  fwd_weights.safetensors   a random ChessNet's fp32 weights (state_dict FQNs)
  fwd_fixture.safetensors   input [n,18,8,8] + PyTorch ref policy [n,4672] / value [n]

The Rust worker loads the weights, runs its tch forward on `input`, and checks it
matches ref_logits / ref_value (CPU fp32 max|Δ| < 1e-4).
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "training"))  # shared encode.py / net.py

from net import ChessNet  # noqa: E402

from weights_io import save_weights_st  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "parity"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    net = ChessNet().cpu().eval()
    # Randomize BN running stats too (default is mean 0 / var 1, which would make
    # the BN layers near-identity and hide layout/eps bugs).
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.normal_(0, 0.5)
            m.running_var.uniform_(0.5, 1.5)

    save_weights_st(net, str(out / "fwd_weights.safetensors"))

    x = torch.randn(args.n, 18, 8, 8, dtype=torch.float32)
    with torch.no_grad():
        logits, value = net(x)
    save_file(
        {
            "input": x.contiguous(),
            "ref_logits": logits.contiguous(),
            "ref_value": value.contiguous(),
        },
        str(out / "fwd_fixture.safetensors"),
    )
    print(f"wrote fixture to {out} (n={args.n})")


if __name__ == "__main__":
    main()
