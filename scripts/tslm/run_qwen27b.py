"""Rush27B préinscrit : contrôle technique, fits train bornés, reload/évaluation séparés."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/tslm")]
import run_v2_campaign as campaign
from pipe.tslm.campaign import epoch_batches, train_epoch, write_json
from training_observations import capture_training_diagnostics


def inputs(root):
    """Même cache audité et mêmes598 WAV ; aucune lecture val/test/externe."""
    old = campaign.read_json(ROOT / "docs/evidence/tslm-v2/campaign-c13fd47/preregistration.json")
    paths = {key: root / Path(old["paths"][key]).relative_to("/home/hicham/pipe-v0")
             for key in ("prepared", "data_root")}
    paths["manifests"] = ROOT / "manifests"
    if campaign.sha256_file(paths["prepared"] / "train.npz") != old["identity"]["cache_sha256"]["train"]:
        raise ValueError("Cache train modifié")
    split = campaign.split_loader.load_split(paths["manifests"])
    rows, series, amplitudes, _ = campaign.load_fold(paths, split, "train")
    path = ROOT / "docs/evidence/tslm-v2/train-diagnostic-001/folds.json"
    if campaign.sha256_file(path) != campaign.FOLDS_SHA256:
        raise ValueError("Folds internes modifiés")
    folds = campaign.read_json(path)
    campaign.validate_folds(rows, folds)
    return rows, series, amplitudes, folds["folds"]


def runtime():
    import torch
    return {"python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("torch", "transformers", "peft", "numpy", "scikit-learn", "timenet", "opentslm")},
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "lock_sha256": campaign.sha256_file(ROOT / "requirements-ml.lock"),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "deterministic": torch.are_deterministic_algorithms_enabled()}


def frozen_hash(model):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.llm.state_dict().items()):
        if "lora_" not in name:
            digest.update(name.encode())
            digest.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    if any(p.requires_grad or p.grad is not None for name, p in model.llm.named_parameters()
           if "lora_" not in name):
        raise ValueError("Poids de base non gelés")
    return digest.hexdigest()


def initialize(base, config):
    import torch
    from pipe.tslm.model import AcousticQwenSP
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model = AcousticQwenSP(base, single_clip_acoustic_encoding=True)
    if model.llm.config.hidden_size != config["hidden_size"]:
        raise ValueError("Dimension cachée différente du27B")
    model.configure_lora(config["lora"])
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
        {"params": model.projector.parameters(), "lr": config["projector_lr"]},
        {"params": model.get_lora_parameters(), "lr": config["lora_lr"]}], weight_decay=config["weight_decay"])
    if optimizer.state or model.scoring_spec()["version"] != config["scoring_version"]:
        raise ValueError("Optimiseur non neuf ou scoring incompatible")
    return model, optimizer


def save_bundle(model, base, directory, metadata):
    """Base immuable en liens physiques locaux ; jamais de fusion des LoRA."""
    import torch
    from pipe.tslm.train import complete_bundle
    bundle = directory / "bundle"
    (bundle / "base").mkdir(parents=True, exist_ok=False)
    receipt = campaign.read_json(base / "download-receipt.json")
    for name in receipt["files"]:
        if Path(name).name != name:
            raise ValueError("Nom de base non local")
        os.link(base / name, bundle / "base" / name)
    model.llm.save_pretrained(bundle / "adapter", safe_serialization=True)
    torch.save({"encoder_state": model.encoder.state_dict(), "projector_state": model.projector.state_dict()},
               bundle / "temporal.pt")
    write_json(bundle / "metadata.json", metadata)
    write_json(bundle / "scoring-spec.json", model.scoring_spec())
    checksum = complete_bundle(bundle, base, ROOT / "requirements-ml.lock", provenance_overrides={
        "qwen": "https://huggingface.co/Qwen/Qwen3.8-27B ; révision dans metadata.json",
        "audio": {"reload_example": "Aucun WAV embarqué ; contrôles sur train uniquement."}})
    return checksum


def fit(args, config, rows, series, amplitudes, folds):
    import torch
    from pipe.tslm.train import debug_rows, tensor_state_hash
    probe = args.mode == "probe"
    by_id = {r["clip_id"]: r for r in rows}
    training = [by_id[cid] for cid in folds[args.fold]["train_ids"]]
    if probe:
        training = debug_rows(training, 8)
    metadata = {**config, "single_clip_acoustic_encoding": True, "amplitude_evidence": False,
                "max_new_tokens": 48, "fold_id": args.fold, "n_configs_compared": 1,
                "model_version": f"pipe-qwen3.8-27b-lora-{args.code_revision[:8]}-fold{args.fold}",
                "purpose": "technical_probe_discard" if probe else "internal_train_prototype_not_validated"}
    directory = args.output
    directory.mkdir(parents=True, exist_ok=False)
    pre = {"config": metadata, "code_revision": args.code_revision, "mode": args.mode,
           "epochs": 2 if probe else args.epochs, "microbatch_size": args.microbatch,
           "training_ids": [r["clip_id"] for r in training],
           "heldout_ids": [] if probe else folds[args.fold]["heldout_ids"],
           "folds_sha256": campaign.FOLDS_SHA256, "runtime": runtime(),
           "config_sha256": campaign.sha256_file(args.config),
           "base_receipt_sha256": campaign.sha256_file(args.base / "download-receipt.json"),
           "created_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
           "hostname": platform.node(), "official_validation_test_external_read": False}
    if not probe:
        gate = campaign.read_json(args.gate)
        if (gate["passed"] is not True or gate["microbatch_size"] != args.microbatch
                or gate["epochs"] != args.epochs or gate["config_sha256"] != pre["config_sha256"]
                or gate["runtime"] != pre["runtime"] or gate["code_revision"] != args.code_revision):
            raise ValueError("Gate technique/budget différents du fit")
        pre["gate_sha256"] = campaign.sha256_file(args.gate)
    write_json(directory / "preregistration.json", pre)
    model, optimizer = initialize(args.base, config)
    write_json(directory / "loading.json", {"loading_info": model.loading_info,
        "scoring_spec": model.scoring_spec(), "trainable": {
            name: {"shape": list(p.shape), "dtype": str(p.dtype)}
            for name, p in model.named_parameters() if p.requires_grad}})
    before = {name: tensor_state_hash(getattr(model, name)) for name in ("encoder", "projector")}
    before["base"] = frozen_hash(model)
    samples = campaign.examples(training, series, amplitudes, metadata, training=True)
    recipe = {**config, "microbatch_size": args.microbatch}
    deadline = datetime.fromisoformat(config["training_stop_utc"]).timestamp()
    steps, presentations, completed_epochs, durations = 0, 0, 0, []
    torch.cuda.reset_peak_memory_stats()
    with (directory / "training.jsonl").open("x") as log, capture_training_diagnostics(
            model, optimizer, allow_lora=True, allow_microbatches=True) as audit:
        for epoch in range(1, pre["epochs"] + 1):
            iterator = iter(train_epoch(model, optimizer, samples, recipe, epoch))
            completed = True
            for indices in epoch_batches(len(samples), 8, config["seed"], epoch):
                if time.time() >= deadline:
                    completed = False
                    break
                torch.cuda.synchronize()
                started = time.monotonic()
                record = next(iterator)
                report = audit.pop_step([training[i]["clip_id"] for i in indices], training_record=record)
                torch.cuda.synchronize()
                durations.append(time.monotonic() - started)
                steps += 1
                presentations += len(indices)
                record = {"step": steps, "seconds": durations[-1], **report}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps({"step": steps, "epoch": epoch, "loss": record["training_record"]["loss"],
                    "class_nll": record["terms"]["class"], "seconds": durations[-1]}), flush=True)
            iterator.close()
            write_json(directory / f"epoch-{epoch}.json", audit.epoch_summary(reset=True))
            completed_epochs += int(completed)
            if not completed:
                break
    if not steps:
        raise RuntimeError("Budget expiré avant toute mise à jour ; aucun prototype revendiqué")
    after = {name: tensor_state_hash(getattr(model, name)) for name in ("encoder", "projector")}
    after["base"] = frozen_hash(model)
    if before["base"] != after["base"] or any(before[n] == after[n] for n in ("encoder", "projector")):
        raise RuntimeError("Gel de la base ou apprentissage acoustique invalides")
    peak = torch.cuda.max_memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    result = {"optimizer_steps": steps, "presentations": presentations, "completed_epochs": completed_epochs,
              "planned_epochs": pre["epochs"], "fit_complete": completed_epochs == pre["epochs"],
              "state_before": before, "state_after": after, "runtime": pre["runtime"],
              "microbatch_size": args.microbatch, "seconds_per_step": float(np.mean(durations)),
              "peak_reserved_bytes": peak, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "gpu_total_bytes": total, "memory_gate_passed": peak <= 0.9 * total,
              "official_validation_test_external_read": False}
    if probe and not result["memory_gate_passed"]:
        write_json(directory / "memory-failed.json", result)
        raise RuntimeError("Moins de10% de marge VRAM : essayer le microbatch suivant, pas cette recette en fit")
    model.eval()
    checks = training[:4]
    result["reload_train_ids"] = [r["clip_id"] for r in checks]
    result["reload_probabilities"] = campaign.score_rows(model, checks, series, amplitudes, metadata)
    result["bundle_sha256"] = save_bundle(model, args.base, directory, metadata)
    write_json(directory / "training-result.json", result)
    print(json.dumps(result), flush=True)


def reload_or_evaluate(args, rows, series, amplitudes, folds):
    from pipe.tslm.predict import Predictor
    import torch
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate as collate
    fit_result = campaign.read_json(args.checkpoint / "training-result.json")
    if campaign.sha256_file(args.checkpoint / "bundle/checksums.json") != fit_result["bundle_sha256"]:
        raise ValueError("Empreinte bundle modifiée")
    predictor = Predictor(args.checkpoint / "bundle")
    by_id = {r["clip_id"]: r for r in rows}
    checks = [by_id[cid] for cid in fit_result["reload_train_ids"]]
    scores = campaign.score_rows(predictor.model, checks, series, amplitudes, predictor.metadata)
    delta = max(abs(p - fit_result["reload_probabilities"][cid]) for cid, p in scores.items())
    if delta > 1e-6:
        raise ValueError(f"Reload divergent : {delta}")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"passed": True, "reload_max_absolute_difference": delta,
              "probabilities": scores, "runtime": runtime(), "pid": os.getpid(), "hostname": platform.node(),
              "bundle_sha256": fit_result["bundle_sha256"], "optimizer_steps": 0,
              "official_validation_test_external_read": False}
    if args.mode == "evaluate":
        fold_id = predictor.metadata["fold_id"]
        heldout = [by_id[cid] for cid in folds[fold_id]["heldout_ids"]]
        probabilities, nlls = {}, []
        with (args.output / "observations.jsonl").open("x") as stream:
            for row in heldout:
                example = campaign.examples([row], series, amplitudes, predictor.metadata)
                logps = predictor.model.score_class_logprobs(collate(example, normalize=False))
                normalized = torch.log_softmax(logps, dim=-1)[0]
                p = float(normalized[0].exp())
                nll = -float(normalized[0 if row["label"] == "leak" else 1])
                probabilities[row["clip_id"]] = p
                nlls.append(nll)
                stream.write(json.dumps({"clip_id": row["clip_id"], "class_logprobs": logps[0].tolist(),
                                         "probability_leak": p, "binary_nll": nll}, allow_nan=False) + "\n")
                stream.flush()
        campaign.write_predictions(args.output / "predictions.csv", [r["clip_id"] for r in heldout], probabilities)
        result.update(fold_id=fold_id, heldout_clips=len(heldout), fit_complete=fit_result["fit_complete"],
                      metrics=campaign.metric_report(heldout, probabilities), binary_nll=float(np.mean(nlls)))
    write_json(args.output / "report.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "train", "reload", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tslm/qwen27b_lora.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--epochs", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--microbatch", type=int, choices=(4, 2, 1), default=4)
    args = parser.parse_args()
    config = campaign.read_json(args.config)
    rows, series, amplitudes, folds = inputs(args.root)
    if args.mode in ("probe", "train"):
        receipt = campaign.read_json(args.base / "download-receipt.json")
        if receipt["repo_id"] != config["base_model"] or receipt["revision"] != config["base_revision"]:
            raise ValueError("Base différente de la recette")
        fit(args, config, rows, series, amplitudes, folds)
    else:
        reload_or_evaluate(args, rows, series, amplitudes, folds)


if __name__ == "__main__":
    main()
