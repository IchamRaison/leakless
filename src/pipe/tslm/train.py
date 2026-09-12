"""Petit surapprentissage de diagnostic sur train uniquement, checkpoint autonome."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

from pipe.tslm.model import AcousticQwenSP
from pipe.tslm.predict import predict_audio, sha256_file
from pipe.tslm.prepare import MANIFEST_HASHES, PROTOCOL, load_manifest
from pipe.tslm.preprocessing import VERSION, model_input, target_text


def tensor_state_hash(module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def debug_rows(rows: list[dict], count: int) -> list[dict]:
    if count < 2 or count % 2:
        raise ValueError("Sous-ensemble debug pair, au moins deux clips")
    selected = []
    for label in ("leak", "no_leak"):
        seen = set()
        for row in sorted(rows, key=lambda row: row["clip_id"]):
            if row["fold"] == "train" and row["label"] == label and row["group_id"] not in seen:
                seen.add(row["group_id"])
                selected.append(row)
                if len(seen) == count // 2:
                    break
    if len(selected) != count:
        raise ValueError("Nombre de groupes train insuffisant")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tslm/v0.json"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["protocol"] != PROTOCOL or config["preprocessing_version"] != VERSION:
        raise ValueError("Configuration incompatible avec les données")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    rows = load_manifest(args.manifest)
    chosen = debug_rows(rows, config["debug_train_samples"])
    with np.load(args.prepared / "train.npz", allow_pickle=False) as cache:
        if str(cache["preprocessing_version"]) != VERSION:
            raise ValueError("Cache de mauvaise version")
        cached = dict(zip(cache["ids"].tolist(), cache["series"]))
    samples = []
    for row in chosen:
        example = model_input(cached[row["clip_id"]])
        example["answer"] = target_text(row["label"], cached[row["clip_id"]])
        samples.append(example)
    validation = min((r for r in rows if r["fold"] == "val"), key=lambda row: row["clip_id"])
    validation_audio = (args.data_root / validation["path"]).read_bytes()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model = AcousticQwenSP(args.base)
    metadata = {**config, "model_version": f"pipe-qwen3.5-4b-v0-{args.code_revision[:8]}",
                "code_revision": args.code_revision, "manifest_sha256": MANIFEST_HASHES["split_v2.csv"],
                "training_clip_ids": [r["clip_id"] for r in chosen],
                "training_groups": [r["group_id"] for r in chosen],
                "validation_example_id": validation["clip_id"], "purpose": "debug_overfit_not_evaluation"}
    model.eval()
    before_prediction = predict_audio(model, validation_audio, metadata).model_dump(mode="json")
    frozen_before = tensor_state_hash(model.llm)
    initial = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
        {"params": model.projector.parameters(), "lr": config["projector_lr"]},
    ], weight_decay=config["weight_decay"])
    model.train()
    model.llm.eval()
    order = list(range(len(samples)))
    random.shuffle(order)
    losses, first_gradients = [], {}
    with (args.output / "training.jsonl").open("x") as log:
        for step in range(config["max_steps"]):
            indices = [order[(step * config["batch_size"] + i) % len(order)] for i in range(config["batch_size"])]
            batch = extend_time_series_to_match_patch_size_and_aggregate(
                [{**samples[i]} for i in indices], normalize=False)
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss non finie, run arrêté")
            loss.backward()
            gradients = {}
            for name, module in (("encoder", model.encoder), ("projector", model.projector)):
                parameters = list(module.parameters())
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise RuntimeError(f"Gradients invalides : {name}")
                norm = torch.sqrt(sum(p.grad.float().square().sum() for p in parameters)).item()
                if norm <= 0:
                    raise RuntimeError(f"Gradient nul : {name}")
                gradients[name] = norm
            if step == 0:
                first_gradients = gradients
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["gradient_clip"])
            optimizer.step()
            losses.append(loss.item())
            record = {"step": step + 1, "loss": loss.item(), "gradient_norms": gradients,
                      "elapsed_seconds": time.monotonic() - started}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    changed = {}
    for component in ("encoder", "projector"):
        delta = sum((p.detach() - initial[name]).float().square().sum()
                    for name, p in model.named_parameters() if name.startswith(component + "."))
        changed[component] = torch.sqrt(delta).item()
        if changed[component] <= 0:
            raise RuntimeError(f"Poids inchangés : {component}")
    frozen_after = tensor_state_hash(model.llm)
    if frozen_before != frozen_after or any(p.grad is not None or p.requires_grad for p in model.llm.parameters()):
        raise RuntimeError("Le décodeur gelé a changé")
    model.eval()
    after_prediction = predict_audio(model, validation_audio, metadata).model_dump(mode="json")
    # Bundle autonome : décodeur réellement utilisé + tokenizer + paramètres temporels.
    model.llm.save_pretrained(args.output / "base", safe_serialization=True, max_shard_size="3GB")
    model.tokenizer.save_pretrained(args.output / "base")
    torch.save({"encoder_state": model.encoder.state_dict(), "projector_state": model.projector.state_dict()},
               args.output / "temporal.pt")
    torch.save(optimizer.state_dict(), args.output / "optimizer.pt")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = {"purpose": "debug_overfit_not_evaluation", "steps": len(losses),
              "first_loss": losses[0], "last_loss": losses[-1], "all_losses_finite": True,
              "first_gradient_norms": first_gradients, "weight_delta_norms": changed,
              "frozen_llm_sha256_before": frozen_before, "frozen_llm_sha256_after": frozen_after,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "before_validation_prediction": before_prediction,
              "after_validation_prediction": after_prediction,
              "elapsed_seconds": time.monotonic() - started,
              "peak_memory_bytes": torch.cuda.max_memory_allocated()}
    (args.output / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "reload-example.wav").write_bytes(validation_audio)
    checksums = {str(path.relative_to(args.output)): sha256_file(path)
                 for path in sorted(args.output.rglob("*")) if path.is_file()}
    (args.output / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    print(json.dumps({"completed": report, "checksums_sha256": sha256_file(args.output / "checksums.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
