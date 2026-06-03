"""Forward-speed spike: compare B=512 bf16 CUDA forward latency across
eager / torch.compile / TorchScript-frozen / AOTInductor for the 4x96 net.

Goal: find out whether a Python-free fused path (AOTInductor .pt2, or TorchScript
freeze) recovers the torch.compile speed (~1400us) that the hand-written tch
forward (~2474us) lacks. Run with the training venv, exclusive GPU:
    .venv/bin/python aoti_spike.py [checkpoint.pt]
"""
import sys
import time

import torch

sys.path.insert(0, "..")
from net import ChessNet

CKPT = sys.argv[1] if len(sys.argv) > 1 else "../../runs/run1/serving_weights.pt"
B = 512
DEV = "cuda"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck.get("config", {}) or {}
net = ChessNet(n_blocks=cfg.get("n_blocks") or 4, n_filters=cfg.get("n_filters") or 96)
net.load_state_dict(ck["weights"])
net.eval().to(DEV).bfloat16()

x = torch.zeros(B, 18, 8, 8, device=DEV, dtype=torch.bfloat16)
torch.set_grad_enabled(False)


def timeit(fn, label, iters=500, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    us = (time.time() - t) / iters * 1e6
    print(f"{label:28s} {us:7.0f} us/fwd", flush=True)
    return us


# 1) eager
timeit(lambda: net(x), "eager bf16")

# 2) torch.compile (the prod baseline path)
try:
    cnet = torch.compile(net)
    timeit(lambda: cnet(x), "torch.compile")
except Exception as e:
    print(f"torch.compile FAILED: {e}", flush=True)

# 3) TorchScript freeze / optimize_for_inference
try:
    traced = torch.jit.trace(net, (x,))
    frozen = torch.jit.optimize_for_inference(traced)
    timeit(lambda: frozen(x), "torchscript optimize_for_inf")
except Exception as e:
    print(f"torchscript FAILED: {e}", flush=True)

# 4) AOTInductor — the Python-free fused path we care about
try:
    ep = torch.export.export(net, (x,))
    t0 = time.time()
    pkg = torch._inductor.aoti_compile_and_package(ep)
    print(f"AOTI compile took {time.time()-t0:.1f}s -> {pkg}", flush=True)
    runner = torch._inductor.aoti_load_package(pkg)
    timeit(lambda: runner(x), "aotinductor")
except Exception as e:
    import traceback
    print(f"AOTInductor FAILED: {e}", flush=True)
    traceback.print_exc()
