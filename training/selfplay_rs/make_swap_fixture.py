"""Build the fixture for `selfplay_rs aoti-swap-parity`.

Compiles two AOTInductor packages with runtime constant folding — one for the
checkpoint weights (W1) and one for W1+noise (W2) — and writes W2's raw weights as
safetensors. The Rust gate pushes W2 into the W1 package via the real
update_constant_buffer -> run_const_fold -> swap flow and checks the forward
matches the natively-compiled W2 package (same fusion both sides).

Run on the box with the training venv:
    .venv/bin/python selfplay_rs/make_swap_fixture.py [checkpoint.pt] [outdir] [batch] [noise]

noise (default 0.02) is a MULTIPLICATIVE perturbation: W2 = W1 * (1 + noise*randn),
which keeps BN running_var positive (additive noise can drive the 1-channel
value_bn var negative -> ill-conditioned fold). noise=0 gives an identity swap
(W2==W1), which must yield an exact match and isolates the swap mechanism.
"""
import sys

import torch
import torch._inductor
from safetensors.torch import save_file

sys.path.insert(0, "..")
from net import ChessNet  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else "../../runs/run1/serving_weights.pt"
OUT = sys.argv[2].rstrip("/") if len(sys.argv) > 2 else "."
B = int(sys.argv[3]) if len(sys.argv) > 3 else 512
NOISE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02

# Match the trainer: folded constants stay runtime-updatable.
torch._inductor.config.aot_inductor.use_runtime_constant_folding = True

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck.get("config", {}) or {}
sd1 = ck["weights"] if "weights" in ck else ck


def compile_from(sd, path):
    net = ChessNet(n_blocks=cfg.get("n_blocks") or 4, n_filters=cfg.get("n_filters") or 96)
    net.load_state_dict(sd)
    net.eval().to("cuda").bfloat16()
    x = torch.zeros(B, 18, 8, 8, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ep = torch.export.export(net, (x,))
        torch._inductor.aoti_compile_and_package(ep, package_path=path)
    print(f"  compiled {path}", flush=True)


# W2 = W1 * (1 + noise*randn) on float tensors (well-conditioned: keeps BN var > 0).
torch.manual_seed(7)
sd2 = {k: (v * (1.0 + NOISE * torch.randn_like(v)) if v.is_floating_point() else v.clone())
       for k, v in sd1.items()}

compile_from(sd1, f"{OUT}/swap_w1.pt2")
compile_from(sd2, f"{OUT}/swap_w2.pt2")

# W2's raw primaries (bf16, dotted FQN, minus num_batches_tracked) — the Rust gate
# mangles '.'->'_' and pushes these.
st = {k: v.detach().to("cpu", torch.bfloat16).contiguous()
      for k, v in sd2.items() if "num_batches_tracked" not in k}
save_file(st, f"{OUT}/swap_w2.safetensors")
print(f"wrote {OUT}/swap_w1.pt2, {OUT}/swap_w2.pt2, {OUT}/swap_w2.safetensors")
