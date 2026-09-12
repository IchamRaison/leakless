"""Campagne V1 préenregistrée : train complet, trois candidats, validation seule.

Le dossier bundle/ est l'unique checkpoint autonome retenu. Aucun cache test
n'est ouvert et aucune métrique finale n'est calculée par cette commande.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np


def development_rows(rows):
    """Les labels sont réservés au train/validation ; pas d'entrée modèle ici."""
    ids, group_folds, group_labels = set(), defaultdict(set), defaultdict(set)
    for row in rows:
        if row["clip_id"] in ids or row["fold"] not in ("train", "val", "test"):
            raise ValueError("Identifiant répété ou fold inconnu")
        ids.add(row["clip_id"])
        group_folds[row["group_id"]].add(row["fold"])
        if row["fold"] in ("train", "val"):
            group_labels[row["group_id"]].add(row["label"])
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("Groupe partagé entre folds")
    if any(len(labels) != 1 or not labels <= {"leak", "no_leak"} for labels in group_labels.values()):
        raise ValueError("Labels discordants ou inconnus dans un groupe de développement")
    return {fold: sorted((row for row in rows if row["fold"] == fold),
                         key=lambda row: row["clip_id"]) for fold in ("train", "val")}


def load_cache(prepared, fold, rows, version):
    if fold not in ("train", "val") or any(row["fold"] != fold for row in rows):
        raise ValueError("Seuls les caches train et validation sont autorisés")
    with np.load(Path(prepared) / f"{fold}.npz", allow_pickle=False) as cache:
        if str(cache["preprocessing_version"]) != version:
            raise ValueError("Version de preprocessing du cache incompatible")
        ids, series = cache["ids"].tolist(), cache["series"]
        if (len(ids) != len(set(ids)) or set(ids) != {r["clip_id"] for r in rows}
                or series.shape != (len(rows), 4, 64) or not np.isfinite(series).all()
                or (series < 0).any() or np.any(series[:, :, 61:] != 0)):
            raise ValueError("IDs, couverture ou signaux du cache incompatibles avec le fold")
        return dict(zip(ids, series))


def epoch_batches(count, batch_size, seed, epoch):
    """Chaque clip exactement une fois par époque ; aucun remplissage du dernier lot."""
    if min(count, batch_size, epoch) < 1:
        raise ValueError("Effectif, taille de lot et époque strictement positifs requis")
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return [order[start:start + batch_size] for start in range(0, count, batch_size)]


def train_epoch(model, optimizer, samples, config, epoch):
    """Une époque, mêmes lots/gradients que V1 ; sélection et logs restent à l'appelant."""
    import torch
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate as collate

    model.train()
    model.llm.eval()
    for indices in epoch_batches(len(samples), config["batch_size"], config["seed"], epoch):
        batch = collate([{**samples[i]} for i in indices], normalize=False)
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss non finie ; campagne arrêtée")
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
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                      config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        yield {"epoch": epoch, "batch_samples": len(indices), "loss": loss.item(),
               "gradient_norms": gradients}


def diagnostic_rows(rows, count=8):
    if count < 2 or count % 2 or any(row["fold"] != "val" for row in rows):
        raise ValueError("Diagnostic pair, uniquement sur validation")
    selected = []
    for label in ("leak", "no_leak"):
        seen = set()
        for row in sorted(rows, key=lambda row: row["clip_id"]):
            if row["label"] == label and row["group_id"] not in seen:
                selected.append(row)
                seen.add(row["group_id"])
                if len(seen) == count // 2:
                    break
    if len(selected) != count:
        raise ValueError("Pas assez de groupes validation pour les diagnostics fixés")
    return sorted(selected, key=lambda row: row["clip_id"])


def validation_groups(rows, probabilities):
    """Médiane par groupe ; interdit toute ligne hors validation, même sans label test."""
    if (not rows or any(row["fold"] != "val" for row in rows)
            or len({row["clip_id"] for row in rows}) != len(rows)
            or set(probabilities) != {row["clip_id"] for row in rows}):
        raise ValueError("Couverture validation uniquement requise")
    groups = defaultdict(list)
    for row in rows:
        p = probabilities[row["clip_id"]]
        if row["label"] not in ("leak", "no_leak") or not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Classe ou probabilité validation invalide")
        groups[row["group_id"]].append((row["label"] == "leak", p))
    labels, scores = [], []
    for members in groups.values():
        if len({label for label, _ in members}) != 1:
            raise ValueError("Classes discordantes dans un groupe validation")
        labels.append(int(members[0][0]))
        scores.append(float(np.median([p for _, p in members])))
    if set(labels) != {0, 1}:
        raise ValueError("Deux classes validation requises")
    return labels, scores


def select_candidate(reports):
    if not reports or any(not np.isfinite(r["validation_group_roc_auc"]) for r in reports):
        raise ValueError("Candidats absents ou critère non fini")
    return min(reports, key=lambda r: (-r["validation_group_roc_auc"], r["epoch"]))


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main():
    # Imports lourds uniquement à l'exécution : les invariants du plan sont testables CPU.
    import torch
    from sklearn.metrics import roc_auc_score
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate as collate
    from pipe.tslm.model import AcousticQwenSP
    from pipe.tslm.predict import predict_audio, sha256_file
    from pipe.tslm.prepare import load_manifest, MANIFEST_HASHES, PROTOCOL, safe_member_path
    from pipe.tslm.preprocessing import VERSION, decode_wav, model_input, preprocess_audio, target_text
    from pipe.tslm.train import complete_bundle, tensor_state_hash

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tslm/v1.json"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--environment-lock", type=Path, default=Path("requirements-ml.lock"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (config["protocol"] != PROTOCOL or config["preprocessing_version"] != VERSION
            or config["candidate_epochs"] != [2, 4, 8] or config["n_configs_planned"] != 3
            or config["initialization"] != "fresh_temporal_encoder_projector_frozen_qwen_base"
            or config["selection_rule"] != "max_validation_roc_auc_of_group_median_probability; exact_tie_earliest_epoch"
            or config["diagnostic_validation_samples"] != 8 or config["validation_batch_size"] != 1):
        raise ValueError("Configuration différente de la campagne préenregistrée")
    for path in (args.environment_lock, args.base / "LICENSE", args.base / "README.md"):
        if not path.is_file():
            raise ValueError(f"Prérequis de livraison manquant : {path}")
    rows = development_rows(load_manifest(args.manifest))
    if {fold: len(items) for fold, items in rows.items()} != {"train": 598, "val": 208}:
        raise ValueError("Effectifs de développement différents du v2 gelé")
    if {fold: len({r["group_id"] for r in items}) for fold, items in rows.items()} != {"train": 102, "val": 42}:
        raise ValueError("Groupes de développement différents du v2 gelé")
    caches = {fold: load_cache(args.prepared, fold, items, VERSION) for fold, items in rows.items()}
    diagnostics = diagnostic_rows(rows["val"], config["diagnostic_validation_samples"])
    diagnostic_audio = {row["clip_id"]: safe_member_path(row["path"], args.data_root).read_bytes()
                        for row in diagnostics}
    for cid, audio in diagnostic_audio.items():
        np.testing.assert_allclose(preprocess_audio(decode_wav(audio), 8000), caches["val"][cid],
                                   rtol=1e-6, atol=1e-6)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    preregistration = {"purpose": "model_selection_on_validation_not_final_evaluation",
                      "timestamp": datetime.now(timezone.utc).isoformat(), "config": config,
                      "config_hash": config_hash, "code_revision": args.code_revision,
                      "manifest_sha256": MANIFEST_HASHES["split_v2.csv"],
                      "cache_sha256": {fold: sha256_file(args.prepared / f"{fold}.npz") for fold in rows},
                      "environment_lock_sha256": sha256_file(args.environment_lock),
                      "training_clip_ids": [r["clip_id"] for r in rows["train"]],
                      "validation_clip_ids": [r["clip_id"] for r in rows["val"]],
                      "diagnostic_validation_ids": [r["clip_id"] for r in diagnostics],
                      "diagnostics_used_for_selection": False, "test_audio_or_cache_opened": False,
                      "frozen_manifest_contains_test_metadata": True,
                      "test_labels_not_used_for_tuning": True}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "preregistration.json", preregistration)
    print(json.dumps({"preregistered": str(args.output), "config_hash": config_hash}), flush=True)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    started = time.monotonic()
    model = AcousticQwenSP(args.base)
    model.eval()
    scoring_spec = model.scoring_spec()
    if scoring_spec["version"] != config["scoring_version"]:
        raise ValueError("Méthode de score différente du plan")
    write_json(args.output / "scoring-spec.json", scoring_spec)
    frozen_before = tensor_state_hash(model.llm)
    initial = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
        {"params": model.projector.parameters(), "lr": config["projector_lr"]},
    ], weight_decay=config["weight_decay"])
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for row in rows["train"]:
        series = caches["train"][row["clip_id"]]
        samples.append({**model_input(series), "answer": target_text(row["label"], series)})
    reports, losses, first_gradients, step = [], [], {}, 0
    with (args.output / "training.jsonl").open("x") as log:
        for epoch in range(1, max(config["candidate_epochs"]) + 1):
            for result in train_epoch(model, optimizer, samples, config, epoch):
                if step == 0:
                    first_gradients = result["gradient_norms"]
                step += 1
                losses.append(result["loss"])
                record = {"step": step, **result,
                          "elapsed_seconds": time.monotonic() - started}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
            if epoch not in config["candidate_epochs"]:
                continue
            model.eval()
            candidate = args.output / "candidates" / f"epoch-{epoch:03d}"
            candidate.mkdir(parents=True, exist_ok=False)
            torch.save({"encoder_state": model.encoder.state_dict(), "projector_state": model.projector.state_dict()},
                       candidate / "temporal.pt")
            torch.save(optimizer.state_dict(), candidate / "optimizer.pt")
            probabilities = {}
            score_started = time.monotonic()
            for offset in range(0, len(rows["val"]), config["validation_batch_size"]):
                chunk = rows["val"][offset:offset + config["validation_batch_size"]]
                batch = collate([model_input(caches["val"][r["clip_id"]]) for r in chunk], normalize=False)
                scores = model.score_probability_leak(batch)
                if len(scores) != len(chunk):
                    raise ValueError("Nombre de scores différent du lot validation")
                probabilities.update({r["clip_id"]: p for r, p in zip(chunk, scores)})
            labels, group_scores = validation_groups(rows["val"], probabilities)
            scoring_seconds = time.monotonic() - score_started
            diagnostic_report = []
            version = f"pipe-qwen3.5-4b-v1-{args.code_revision[:8]}-e{epoch}"
            for row in diagnostics:
                result = predict_audio(model, diagnostic_audio[row["clip_id"]],
                                       {**config, "model_version": version}).model_dump(mode="json")
                expected = target_text(row["label"], caches["val"][row["clip_id"]]).split("; ", 1)[1]
                diagnostic_report.append({"clip_id": row["clip_id"], "prediction": result,
                                          "format_valid": not result["abstained"],
                                          "description_matches_dsp": result["description"] == expected,
                                          "expected_measured_description": expected})
            report = {"purpose": "development_only_not_final_evaluation", "epoch": epoch, "step": step,
                      "model_version": version, "validation_clips": len(probabilities),
                      "validation_groups": len(labels), "validation_groups_by_label": dict(Counter(labels)),
                      "validation_group_roc_auc": float(roc_auc_score(labels, group_scores)),
                      "selection_rule": config["selection_rule"], "probabilities": probabilities,
                      "scoring_seconds": scoring_seconds, "text_diagnostics_not_for_selection": diagnostic_report,
                      "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                      "temporal_sha256": sha256_file(candidate / "temporal.pt"),
                      "optimizer_sha256": sha256_file(candidate / "optimizer.pt")}
            write_json(candidate / "development.json", report)
            reports.append(report)
            print(json.dumps({"candidate_epoch": epoch, "validation_group_roc_auc": report["validation_group_roc_auc"],
                              "development_only": True, "scoring_seconds": scoring_seconds}), flush=True)
    frozen_after = tensor_state_hash(model.llm)
    if frozen_before != frozen_after or any(p.grad is not None or p.requires_grad for p in model.llm.parameters()):
        raise RuntimeError("Le décodeur Qwen gelé a changé")
    selected = select_candidate(reports)
    if [r["epoch"] for r in reports] != config["candidate_epochs"]:
        raise RuntimeError("Campagne incomplète ; aucune livraison partielle sélectionnée")
    source = args.output / "candidates" / f"epoch-{selected['epoch']:03d}"
    if any(sha256_file(source / f"{name}.pt") != selected[f"{name}_sha256"]
           for name in ("temporal", "optimizer")):
        raise ValueError("Le candidat sélectionné a changé depuis sa validation")
    temporal = torch.load(source / "temporal.pt", map_location=model.device, weights_only=True)
    model.encoder.load_state_dict(temporal["encoder_state"], strict=True)
    model.projector.load_state_dict(temporal["projector_state"], strict=True)
    model.eval()
    changed = {}
    for component in ("encoder", "projector"):
        delta = sum((p.detach() - initial[name]).float().square().sum()
                    for name, p in model.named_parameters() if name.startswith(component + "."))
        changed[component] = torch.sqrt(delta).item()
        if not np.isfinite(changed[component]) or changed[component] <= 0:
            raise RuntimeError(f"Poids temporels inchangés ou invalides : {component}")
    selection = {"selected_epoch": selected["epoch"], "n_configs_compared": len(reports),
                 "selection_rule": config["selection_rule"],
                 "candidates": [{key: r[key] for key in ("epoch", "step", "validation_group_roc_auc", "temporal_sha256")}
                                for r in reports], "test_labels_not_used_for_tuning": True}
    write_json(args.output / "selection.json", selection)
    bundle = args.output / "bundle"
    bundle.mkdir(exist_ok=False)
    metadata = {**config, **selection, "model_version": selected["model_version"],
                "code_revision": args.code_revision, "training_commit": args.code_revision,
                "config_hash": config_hash, "manifest_sha256": MANIFEST_HASHES["split_v2.csv"],
                "scoring_spec": scoring_spec, "validation_example_id": diagnostics[0]["clip_id"],
                "training_clip_ids": preregistration["training_clip_ids"],
                "training_groups": sorted({r["group_id"] for r in rows["train"]}),
                "purpose": "v1_selected_on_validation_final_evaluation_delegated_to_nevil"}
    example_id = diagnostics[0]["clip_id"]
    audio = diagnostic_audio[example_id]
    example_series = preprocess_audio(decode_wav(audio), 8000)
    reload_probability = model.score_probability_leak(collate([model_input(example_series)], normalize=False))[0]
    reload_prediction = predict_audio(model, audio, metadata).model_dump(mode="json")
    write_json(bundle / "reload-expected.json", {"clip_id": example_id,
                                               "input_sha256": hashlib.sha256(audio).hexdigest(),
                                               "probability_leak": reload_probability,
                                               "prediction": reload_prediction,
                                               "scoring_spec": scoring_spec})
    (bundle / "reload-example.wav").write_bytes(audio)
    model.llm.save_pretrained(bundle / "base", safe_serialization=True, max_shard_size="3GB")
    model.tokenizer.save_pretrained(bundle / "base")
    for name in ("temporal.pt", "optimizer.pt"):
        shutil.copyfile(source / name, bundle / name)
    write_json(bundle / "metadata.json", metadata)
    write_json(bundle / "scoring_spec.json", scoring_spec)
    write_json(bundle / "development.json", selected)
    write_json(bundle / "preregistration.json", preregistration)
    shutil.copyfile(args.output / "training.jsonl", bundle / "training.jsonl")
    report = {"purpose": metadata["purpose"], "steps": step, "epochs": max(config["candidate_epochs"]),
              "selected_epoch": selected["epoch"], "n_configs_compared": len(reports),
              "first_loss": losses[0], "last_loss": losses[-1], "all_losses_finite": True,
              "first_gradient_norms": first_gradients, "selected_weight_delta_norms": changed,
              "frozen_llm_sha256_before": frozen_before, "frozen_llm_sha256_after": frozen_after,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "elapsed_seconds": time.monotonic() - started,
              "peak_memory_bytes": torch.cuda.max_memory_allocated(), "test_audio_or_cache_opened": False,
              "frozen_manifest_contains_test_metadata": True, "test_labels_not_used_for_tuning": True}
    write_json(bundle / "training-report.json", report)
    digest = complete_bundle(bundle, args.base, args.environment_lock)
    print(json.dumps({"completed": report, "bundle": str(bundle), "checksums_sha256": digest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
