"""Forward-parity fixture generator + driver (Stage 4 correctness gate).

Generates real encoded positions (python-chess random playouts + encode.py),
saves them with PyTorch fp32 AND bf16 reference outputs + per-position legal
masks, then runs the Rust `selfplay_rs forward-check`, which asserts:
  - fp32 Rust(folded) vs PyTorch fp32: tight numeric match;
  - bf16 Rust(folded) vs PyTorch bf16: argmax-over-legal agreement, with any
    disagreement required to be a genuine near-tie.

Run from training/selfplay_rs/ with the training venv:
    .venv/bin/python test_forward_parity.py [checkpoint.pt] [n_positions]
(then the script builds + runs the binary itself if --run is passed).
"""
import os
import random
import subprocess
import sys

import chess
import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, "..")  # encode.py / net.py
from encode import encode_position, legal_move_mask
from net import ChessNet

CKPT = sys.argv[1] if len(sys.argv) > 1 else "../../runs/run1/serving_weights.pt"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
DEV = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck.get("config", {}) or {}
net = ChessNet(n_blocks=cfg.get("n_blocks") or 4, n_filters=cfg.get("n_filters") or 96)
net.load_state_dict(ck["weights"])
net.eval()
save_file({k: v.contiguous() for k, v in net.state_dict().items()}, "weights.safetensors")

# Real positions: random playouts to varied depths, skipping terminal positions.
rng = random.Random(0)
positions, masks = [], []
while len(positions) < N:
    b = chess.Board()
    for _ in range(rng.randint(0, 50)):
        if b.is_game_over(claim_draw=True):
            break
        b.push(rng.choice(list(b.legal_moves)))
    if b.is_game_over(claim_draw=True):
        continue
    positions.append(encode_position(b))
    masks.append(legal_move_mask(b).astype(np.float32))

x = torch.from_numpy(np.stack(positions)).float()
mask = torch.from_numpy(np.stack(masks)).float()

with torch.no_grad():
    net_f32 = net.to(DEV)
    lf, vf = net_f32(x.to(DEV))
    net_bf16 = net.to(DEV).bfloat16()
    xb = x.to(DEV).bfloat16()
    lb, vb = net_bf16(xb)
    # The PROD forward is torch.compile (Inductor-fused, folds BN). This is the
    # true equivalence target for the AOTI (also Inductor-fused) Rust worker —
    # eager-unfolded bf16 is NOT what prod runs.
    cnet = torch.compile(net_bf16)
    lc, vc = cnet(xb)

save_file(
    {
        "input": x.contiguous(),
        "legal_mask": mask.contiguous(),
        "ref_logits_f32": lf.float().cpu().contiguous(),
        "ref_values_f32": vf.float().cpu().contiguous(),
        "ref_logits_bf16": lb.float().cpu().contiguous(),
        "ref_values_bf16": vb.float().cpu().contiguous(),
        "ref_logits_compile": lc.float().cpu().contiguous(),
        "ref_values_compile": vc.float().cpu().contiguous(),
    },
    "fixture.safetensors",
)
print(f"wrote weights.safetensors + fixture.safetensors ({N} real positions)")

if "--run" in sys.argv:
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    env = {**os.environ, "LD_LIBRARY_PATH": torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
    rc = subprocess.run(["./target/release/selfplay_rs", "forward-check", "."], env=env).returncode
    sys.exit(rc)
