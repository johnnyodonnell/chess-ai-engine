"""CUDA-graph correctness + weight-reload safety test for the zero-copy ring.

The ring captures a CUDA graph per slot of the compiled forward. The danger: the
trainer republishes weights every 300s via an IN-PLACE state_dict load; if a
captured graph served STALE weights, self-play would silently generate targets
from an old net (no pre-train validation catches it). This proves:

  1. graph replay == a direct compiled forward (capture is correct);
  2. an in-place state_dict reload (same net object) IS reflected by replay
     WITHOUT recapture (no stale weights) -- the production reload path;
  3. swapping the net object (the rare recompile path) triggers a recapture that
     reflects the new net.

Run on asus-nvidia with chess-train PAUSED (needs the GPU).
"""
import os

import torch

import zerocopy
from net import ChessNet

B = 512
WEIGHTS = os.path.expanduser("~/Workspace/chess-ai-engine/runs/run1/latest.pt")


def load():
    ckpt = torch.load(WEIGHTS, map_location="cuda", weights_only=False)
    cfg = ckpt.get("config", {})
    eager = ChessNet(n_blocks=cfg.get("n_blocks"),
                     n_filters=cfg.get("n_filters")).to("cuda")
    eager.load_state_dict(ckpt["weights"])
    eager.eval()
    eager = eager.bfloat16()
    return eager, torch.compile(eager), ckpt["weights"]


def slot0_out(ring):
    ring.slots[0].event.synchronize()
    return ring.slots[0].logit_host.float().clone()


def direct(net, ring):
    l, _ = net(ring.slots[0].input_cuda)
    torch.cuda.synchronize()
    return l.float().cpu().clone()


def main():
    torch.set_grad_enabled(False)
    eager, net, w0 = load()
    ring = zerocopy.HostRing(2, B)
    assert ring.use_graph, "CHESS_CUDAGRAPH must be on for this test"

    # known input into slot 0 (we play Rust's role)
    ring.slots[0].input_host.copy_((torch.rand(B, 18, 8, 8) > 0.85).bfloat16())

    # (1) capture + replay == direct compiled forward
    ring.run(net, 0, B)
    out1 = slot0_out(ring)
    d1 = direct(net, ring)
    e1 = (out1 - d1).abs().max().item()
    print(f"(1) graph replay vs direct compiled : max|diff|={e1:.3e}  "
          f"{'PASS' if e1 < 2e-2 else 'FAIL'}")

    # (2) in-place state_dict reload (SAME net object) -> replay must reflect it
    w1 = {k: (v.float() * 1.1 if v.is_floating_point() else v) for k, v in w0.items()}
    eager.load_state_dict(w1)   # in-place into the params the graph reads
    eager.eval()
    ring.run(net, 0, B)         # same `net` identity -> graph reused, NOT recaptured
    out2 = slot0_out(ring)
    d2 = direct(net, ring)
    changed = (out1 - out2).abs().max().item()
    e2 = (out2 - d2).abs().max().item()
    print(f"(2) output changed after reload     : max|delta|={changed:.3e}  "
          f"{'PASS' if changed > 1e-1 else 'FAIL (stale weights!)'}")
    print(f"(2) replay vs direct (new weights)  : max|diff|={e2:.3e}  "
          f"{'PASS' if e2 < 2e-2 else 'FAIL (stale weights!)'}")

    # (3) net object swap (recompile path) -> recapture reflects the new net
    eager3, net3, _ = load()     # fresh net object, original weights
    ring.run(net3, 0, B)         # different identity -> invalidate + recapture
    out3 = slot0_out(ring)
    d3 = direct(net3, ring)
    e3 = (out3 - d3).abs().max().item()
    print(f"(3) recapture after net swap        : max|diff|={e3:.3e}  "
          f"{'PASS' if e3 < 2e-2 else 'FAIL'}")

    ok = e1 < 2e-2 and changed > 1e-1 and e2 < 2e-2 and e3 < 2e-2
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
