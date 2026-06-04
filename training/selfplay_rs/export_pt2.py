"""Export the net to an AOTInductor .pt2 package (fused, Python-free forward).

The trainer will do this on each weight publish (~5.4s). Here it's a standalone
helper for the integration spike. Run with the training venv:
    .venv/bin/python export_pt2.py [checkpoint.pt] [out.pt2] [batch]
"""
import sys

import torch

sys.path.insert(0, "..")
from net import ChessNet

CKPT = sys.argv[1] if len(sys.argv) > 1 else "../../runs/run1/serving_weights.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "serving_model.pt2"
B = int(sys.argv[3]) if len(sys.argv) > 3 else 512

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck.get("config", {}) or {}
net = ChessNet(n_blocks=cfg.get("n_blocks") or 4, n_filters=cfg.get("n_filters") or 96)
net.load_state_dict(ck["weights"])
net.eval().to("cuda").bfloat16()

x = torch.zeros(B, 18, 8, 8, device="cuda", dtype=torch.bfloat16)
# Match the trainer: keep folded constants runtime-updatable so the worker can
# push raw weights via update_constant_buffer + run_const_fold (no recompile).
import torch._inductor  # noqa: E402  (ensures torch._inductor.config is loaded)
torch._inductor.config.aot_inductor.use_runtime_constant_folding = True
with torch.no_grad():
    ep = torch.export.export(net, (x,))
    path = torch._inductor.aoti_compile_and_package(ep, package_path=OUT)
print(f"wrote {path}")
