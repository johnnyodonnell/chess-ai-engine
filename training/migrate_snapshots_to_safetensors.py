"""One-off migration for an existing run dir created before the Rust evaluator:
convert each snapshot from pickled .pt -> fp32 safetensors and migrate pool.json
(model "pt" field -> "st"), so evaluate_rs (which loads safetensors) can use the
historical snapshots/anchors. Operates in place. Run once per run dir at cutover.

    python migrate_snapshots_to_safetensors.py --run-dir runs/run1
"""
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_pt(pt_path: Path) -> Path:
    st_path = pt_path.with_suffix(".safetensors")
    state = torch.load(pt_path, map_location="cpu", weights_only=False)
    weights = state["weights"]
    sd = {k: v.detach().to("cpu", torch.float32).contiguous()
          for k, v in weights.items() if "num_batches_tracked" not in k}
    save_file(sd, str(st_path))
    return st_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    snaps = sorted((run_dir / "snapshots").glob("*.pt"))
    print(f"converting {len(snaps)} snapshots -> safetensors")
    for pt in snaps:
        st = convert_pt(pt)
        print(f"  {pt.name} -> {st.name}")

    pool_path = run_dir / "pool.json"
    pool = json.loads(pool_path.read_text())
    migrated = 0
    for name, entry in pool["models"].items():
        pt = entry.pop("pt", None)
        if pt:
            entry["st"] = pt[:-3] + ".safetensors" if pt.endswith(".pt") else pt
            migrated += 1
        elif "st" not in entry:
            entry["st"] = None
    pool_path.write_text(json.dumps(pool, indent=2))
    print(f"migrated {migrated} model entries (pt -> st) in {pool_path}")


if __name__ == "__main__":
    main()
