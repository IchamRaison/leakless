"""Premier forward/backward réel : signal TimeF de train, sans optimiser les poids."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

from pipe.tslm.model import AcousticQwenSP
from pipe.tslm.prepare import load_manifest
from pipe.tslm.preprocessing import model_input, target_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("manifests"))
    args = parser.parse_args()
    torch.manual_seed(20260912)
    rows = {r["clip_id"]: r for r in load_manifest(args.manifest)}
    with np.load(args.prepared / "train.npz", allow_pickle=False) as data:
        clip_id, series = str(data["ids"][0]), data["series"][0]
    if rows[clip_id]["fold"] != "train":
        raise ValueError("Premier batch hors train interdit")
    item = model_input(series)
    item["answer"] = target_text(rows[clip_id]["label"], series)
    batch = extend_time_series_to_match_patch_size_and_aggregate([item], normalize=False)
    start = time.monotonic()
    model = AcousticQwenSP(args.base)
    model.train()
    loss = model.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    grads = {}
    for name, module in (("encoder", model.encoder), ("projector", model.projector)):
        params = list(module.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        norm = torch.sqrt(sum(p.grad.float().square().sum() for p in params)).item()
        assert norm > 0
        grads[name] = norm
    assert all(not p.requires_grad and p.grad is None for p in model.llm.parameters())
    torch.cuda.synchronize()
    report = {"clip_id": clip_id, "fold": "train", "loss": loss.item(),
              "gradient_norms": grads, "llm_frozen": True,
              "loading_info": model.loading_info,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "elapsed_seconds": time.monotonic() - start,
              "peak_memory_bytes": torch.cuda.max_memory_allocated()}
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
