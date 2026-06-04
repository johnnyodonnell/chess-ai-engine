"""Export a trained net to ONNX for browser inference via onnxruntime-web.

Output shape contract:
  input  "position"   float32 [1, 18, 8, 8]
  output "policy"     float32 [1, 4672]    (raw logits, pre-softmax)
  output "value"      float32 [1]          (tanh-bounded, side-to-move POV)
"""

import argparse
import torch

from encode import INPUT_CHANNELS, POLICY_SIZE
from net import ChessNet


def export_module(net, onnx_path):
    """Export an in-memory ChessNet to ONNX. The trainer uses this directly so
    snapshot ONNX no longer round-trips through a .pt on disk (snapshots now
    persist weights as fp32 safetensors instead of pickled .pt)."""
    was_training = net.training
    net.eval()
    dummy = torch.zeros(1, INPUT_CHANNELS, 8, 8, dtype=torch.float32)
    # dynamo=False forces the legacy tracing exporter, which inlines weights
    # in the ONNX file. The new dynamo exporter externalizes them, which
    # would break browser inference (we want a single self-contained .onnx).
    torch.onnx.export(
        net,
        (dummy,),
        onnx_path,
        input_names=["position"],
        output_names=["policy", "value"],
        opset_version=17,
        dynamic_axes=None,
        dynamo=False,
    )
    if was_training:
        net.train()
    print(f"exported {onnx_path} (policy={POLICY_SIZE})")


def export(ckpt_path, onnx_path, n_blocks=None, n_filters=None):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = state.get("config", {})
    nb = n_blocks or cfg.get("n_blocks")
    nf = n_filters or cfg.get("n_filters")
    if nb is None or nf is None:
        from net import N_BLOCKS, N_FILTERS
        nb, nf = N_BLOCKS, N_FILTERS

    net = ChessNet(n_blocks=nb, n_filters=nf)
    net.load_state_dict(state["weights"])
    export_module(net, onnx_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    export(args.ckpt, args.out)
