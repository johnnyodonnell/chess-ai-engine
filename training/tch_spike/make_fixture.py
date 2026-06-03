"""Spike fixture: export the live net's weights to safetensors and a fixed-input
PyTorch fp32 reference forward, for the Rust spike to load and check against.

Run from training/tch_spike/ with the training venv:
    .venv/bin/python make_fixture.py [checkpoint.pt]
"""
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, "..")  # import net.py / encode.py from training/
from net import ChessNet

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "../../runs/run1/serving_weights.pt"
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = ck.get("config", {}) or {}
net = ChessNet(n_blocks=cfg.get("n_blocks") or 4, n_filters=cfg.get("n_filters") or 96)
net.load_state_dict(ck["weights"])
net.eval()

save_file({k: v.contiguous() for k, v in net.state_dict().items()}, "weights.safetensors")

torch.manual_seed(0)
x = torch.rand(8, 18, 8, 8)
with torch.no_grad():
    logits, values = net(x)

save_file(
    {
        "input": x.contiguous(),
        "ref_logits": logits.contiguous(),
        "ref_values": values.contiguous(),
    },
    "fixture.safetensors",
)
print(
    f"saved weights.safetensors ({len(net.state_dict())} tensors) + "
    f"fixture.safetensors (input={tuple(x.shape)} logits={tuple(logits.shape)} "
    f"values={tuple(values.shape)})"
)
