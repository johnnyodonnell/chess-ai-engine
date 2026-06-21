"""fp32 safetensors weight I/O keyed by state_dict FQN.

This is the format the Rust self-play worker (selfplay_reinforce) loads. The
shared AlphaZero export.py only does ONNX, so we keep the safetensors helpers
here (ported from fox-lite export.py).
"""


def save_weights_st(net, path: str):
    """Save params as fp32 safetensors keyed by state_dict FQN (Rust reads these)."""
    import torch
    from safetensors.torch import save_file

    sd = {k: v.detach().to("cpu", torch.float32).contiguous()
          for k, v in net.state_dict().items()}
    save_file(sd, path)


def load_weights_st(net, path: str, device="cpu"):
    from safetensors.torch import load_file

    sd = load_file(path, device=str(device))
    net.load_state_dict(sd)
    return net
