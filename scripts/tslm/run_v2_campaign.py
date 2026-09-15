"""Campagne A/C bornée : préinscrire, six folds train, sélection, refit, puis seuil val.

Chaque commande est une étape explicite. Les étapes terminées sont vérifiées et
réutilisées ; un dossier incomplet bloque, sans réentraînement ou écrasement caché.
Aucun WAV/cache test ni holdout externe n'est ouvert par ce runner.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as packages
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/eval"), str(ROOT / "scripts/timenet")]
import diagnose_train as diagnostic
from diagnose_parity import selected_ids
import v2_c1
from export_run import sha256_file, write_predictions
from harness import features, metrics, split_loader
from pipe.tslm.campaign import load_cache, train_epoch, write_json

SCHEMA = "pipe-v2-campaign-v1"
FOLDS_SHA256 = "5c2bea733f8f2a2dc525b9738a5aa40ae3ce220cf6d76d712cab984afb4d5efb"
VARIANTS = ("A", "C")
RECIPE = {"seed": 20260912, "epochs": 4, "batch_size": 8, "microbatch_size": 1,
          "encoder_lr": 0.0002, "projector_lr": 0.0001, "weight_decay": 0.01,
          "gradient_clip": 1.0, "max_new_tokens": 48,
          "single_clip_acoustic_encoding": True,
          "loss": "mean_all_supervised_answer_and_eos_tokens",
          "initialization": "fresh_temporal_encoder_projector_frozen_qwen_base"}
SELECTION_RULE = "max_mean_of_three_inner_fold_group_roc_auc;exact_tie_A_before_C"
THRESHOLD_RULE = "argmax_validation_cluster_macro_f1;median_cluster_score;smallest_threshold_on_exact_tie"
COUNTS = {"tslm_configurations": 2, "tslm_comparison_fits": 6, "tslm_final_fits": 1,
          "c1_configurations": 4, "c1_comparison_fits": 12, "c1_final_fits": 1}
PRIOR_DIAGNOSTIC_FITS = {"timenet_linear_probe": 3, "c1_fixed_C_1": 3,
                        "total_logistic_fits": 6, "tslm_training_fits": 0,
                        "included_in_comparison_counts": False}
DIAGNOSTIC_NAMES = ("docs/evidence/tslm-v2/train-diagnostic-001/probes.json",
                    "docs/evidence/tslm-v2/train-diagnostic-001/supervision.json")
SOURCE_NAMES = (
    "scripts/tslm/run_v2_campaign.py", "scripts/tslm/diagnose_train.py", "scripts/tslm/v2_c1.py",
    "scripts/tslm/diagnose_parity.py", "scripts/tslm/export_run.py", "scripts/tslm/audit_text.py",
    "src/pipe/tslm/campaign.py", "src/pipe/tslm/train.py", "src/pipe/tslm/model.py",
    "src/pipe/tslm/predict.py", "src/pipe/tslm/preprocessing.py", "src/pipe/tslm/prepare.py",
    "src/pipe/tslm/coherent.py", "scripts/timenet/leakless_acoustic/connector.py",
    "scripts/eval/harness/features.py", "scripts/eval/harness/split_loader.py",
    "scripts/eval/harness/metrics.py", "scripts/eval/run_controls.py", "configs/tslm/v1.json")
PATH_ARGUMENTS = ("prepared", "previous_prepared", "folds", "gate_a", "gate_c", "data_root",
                  "manifests", "base", "environment_lock")


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {name: sha256_file(ROOT / name) for name in SOURCE_NAMES}


def finished(directory, *, preregistration_sha256):
    """Un résultat complet n'est réutilisé qu'avec tous ses fichiers intacts."""
    directory = Path(directory)
    if not directory.exists():
        return None
    receipt = directory / "complete.json"
    if not receipt.is_file():
        raise ValueError(f"Étape incomplète conservée pour inspection : {directory}")
    result = read_json(receipt)
    if digest({k: v for k, v in result.items() if k != "receipt_sha256"}) != result.get("receipt_sha256"):
        raise ValueError(f"Reçu de fin modifié : {directory}")
    if result.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError(f"Étape terminée d'une autre préinscription : {directory}")
    actual = {str(p.relative_to(directory)): sha256_file(p) for p in sorted(directory.rglob("*"))
              if p.is_file() and p != receipt}
    if actual != result["artifacts_sha256"]:
        raise ValueError(f"Artefacts modifiés depuis la fin de l'étape : {directory}")
    return result


def finish(directory, result):
    result = {**result, "artifacts_sha256": {
        str(p.relative_to(directory)): sha256_file(p) for p in sorted(directory.rglob("*")) if p.is_file()}}
    result["receipt_sha256"] = digest(result)
    write_json(directory / "complete.json", result)
    return result


def variant_metadata(variant):
    from pipe.tslm.preprocessing import AMPLITUDE_EVIDENCE_VERSION, CANONICAL_VERSION
    if variant not in VARIANTS:
        raise ValueError("Seules les variantes préinscrites A et C sont autorisées")
    return {**RECIPE, "preprocessing_version": CANONICAL_VERSION,
            "amplitude_evidence": variant == "C",
            "amplitude_evidence_version": AMPLITUDE_EVIDENCE_VERSION if variant == "C" else None}


def provenance_overrides(variant):
    if variant not in VARIANTS:
        raise ValueError("Provenance réservée aux variantes A/C")
    transformations = (
        "Normalisation RMS par clip puis conversion float32 canonique TimeF ; fenêtres Hann "
        "de 256 échantillons, pas de 128 ; quatre bandes d'énergie log1p, cibles textuelles déterministes.")
    if variant == "C":
        transformations += (
            " Neuf descripteurs statiques C1 mesurés sur le même signal normalisé float32 "
            "sont ajoutés au prompt sous forme de texte numérique à six chiffres significatifs ; "
            "ni score de baseline, ni métadonnée du dataset, ni série temporelle supplémentaire.")
    return {"audio": {"reload_example": (
        "Aucun WAV de validation embarqué. Les reçus de rechargement et de parité finale sont "
        "externes au bundle ; ce fichier n'atteste pas leur réussite.")}, "transformations": transformations}


def validate_folds(rows, document):
    if (document.get("source_fold") != "train" or document.get("seed") != RECIPE["seed"]
            or document.get("n_folds") != 3 or document.get("n_clips") != 598
            or document.get("split_sha256") != split_loader.FROZEN_SPLIT_SHA256):
        raise ValueError("Protocole des folds train incompatible")
    # Réutiliser la validation complète de couverture/classes/groupes avant tout fit.
    v2_c1._partition_indices(rows, np.zeros((len(rows), 9)), document["folds"])


def validate_gate(gate, variant, expected_ids):
    from pipe.tslm.model import AMPLITUDE_SCORING_VERSION, CANONICAL_SCORING_VERSION
    wanted = AMPLITUDE_SCORING_VERSION if variant == "C" else CANONICAL_SCORING_VERSION
    if (gate.get("clip_ids") != expected_ids or gate.get("amplitude_evidence") is not (variant == "C")
            or gate.get("scoring_spec", {}).get("version") != wanted
            or gate.get("scoring_spec", {}).get("acoustic_batching") != "one_clip_four_channels"
            or gate.get("test_audio_or_cache_opened") is not False
            or gate.get("quality_metrics_calculated") is not False):
        raise ValueError(f"Gate {variant} incomplet ou d'une autre représentation")
    if variant == "C" and set(gate.get("amplitude_tokens", {})) != set(expected_ids):
        raise ValueError("Gate C sans couverture complète des tokens d'amplitude")


def load_fold(paths, split, fold):
    """Relire les seuls WAV du fold demandé et vérifier les deux représentations."""
    from pipe.tslm.preprocessing import (AMPLITUDE_EVIDENCE_VERSION, CANONICAL_VERSION,
        amplitude_features, amplitude_text, decode_wav, preprocess_audio)
    if fold not in ("train", "val"):
        raise ValueError("Aucun accès test/externe autorisé")
    rows = (diagnostic.train_rows(split) if fold == "train" else sorted([
        {"clip_id": c.clip_id, "group_id": c.group_id, "fold": "val",
         "label": "leak" if c.label else "no_leak"} for c in split.fold("val")], key=lambda r: r["clip_id"]))
    if fold == "val" and len(rows) != 208:
        raise ValueError("208 validations officielles requises")
    series = load_cache(paths["prepared"], fold, rows, CANONICAL_VERSION)
    with np.load(Path(paths["prepared"]) / f"{fold}.npz", allow_pickle=False) as cache:
        values, texts = cache["amplitude_features"], cache["amplitude_text"]
        if (str(cache["amplitude_evidence_version"]) != AMPLITUDE_EVIDENCE_VERSION
                or values.dtype != np.float64 or values.shape != (len(rows), 9)
                or texts.shape != (len(rows),) or not np.isfinite(values).all()):
            raise ValueError("Cache d'amplitude incompatible")
        amplitudes = dict(zip(cache["ids"].tolist(), values))
        serialized_text = dict(zip(cache["ids"].tolist(), texts.tolist()))
    md5 = diagnostic.load_expected_audio_md5(paths["manifests"])
    root, matrix = Path(paths["data_root"]).resolve(), []
    for row in rows:
        cid = row["clip_id"]
        path = (root / split.path_of(cid)).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Chemin WAV hors data-root")
        raw = path.read_bytes()
        diagnostic.verify_audio_bytes(raw, cid, md5)
        waveform = decode_wav(raw)
        canonical = preprocess_audio(waveform, 8000, version=CANONICAL_VERSION)
        amplitude = amplitude_features(waveform)
        if (canonical.dtype != series[cid].dtype or not np.array_equal(canonical, series[cid])
                or not np.array_equal(amplitude, amplitudes[cid])
                or amplitude_text(amplitude) != serialized_text[cid]):
            raise ValueError(f"Représentation canonique/cache divergente : {cid}")
        # C1 reste la définition officielle sur le WAV ; C utilise le round-trip TimeF déclaré.
        matrix.append(features.c1_envelope(waveform))
    return rows, series, amplitudes, np.stack(matrix)


def prerequisites(paths):
    from pipe.tslm.preprocessing import CANONICAL_VERSION
    split = split_loader.load_split(paths["manifests"])
    rows = diagnostic.train_rows(split)
    if sha256_file(paths["folds"]) != FOLDS_SHA256:
        raise ValueError("Les folds diagnostiques gelés ont changé")
    folds = read_json(paths["folds"])
    validate_folds(rows, folds)
    expected_ids = selected_ids(split, True)
    gates = {v: diagnostic.verify_parity(paths[f"gate_{v.lower()}"], paths["prepared"], CANONICAL_VERSION)
             for v in VARIANTS}
    for variant, gate in gates.items():
        validate_gate(gate, variant, expected_ids)
    for field in ("cache_sha256", "timef_manifest_sha256", "source_sha256", "state_sha256", "runtime"):
        if gates["A"][field] != gates["C"][field]:
            raise ValueError(f"Gates A/C sur des chaînes différentes : {field}")
    previous = Path(paths["previous_prepared"]) / "train.npz"
    if sha256_file(previous) != folds["cache_train_sha256"]:
        raise ValueError("Ancien cache différent de celui lié aux folds diagnostiques")
    current_series = load_cache(paths["prepared"], "train", rows, CANONICAL_VERSION)
    old_series = load_cache(paths["previous_prepared"], "train", rows, CANONICAL_VERSION)
    if any(current_series[cid].dtype != old_series[cid].dtype
           or not np.array_equal(current_series[cid], old_series[cid]) for cid in current_series):
        raise ValueError("Les quatre bandes ont changé depuis le diagnostic train")
    base = Path(paths["base"])
    if not all((base / name).is_file() for name in ("config.json", "tokenizer_config.json", "LICENSE", "README.md")):
        raise ValueError("Base Qwen et licence incomplètes")
    identity = {"folds_sha256": FOLDS_SHA256,
        "gate_sha256": {v: sha256_file(paths[f"gate_{v.lower()}"]) for v in VARIANTS},
        "cache_sha256": gates["A"]["cache_sha256"],
        "preparation_sha256": sha256_file(Path(paths["prepared"]) / "preparation.json"),
        "base_sha256": {str(p.relative_to(base)): sha256_file(p) for p in sorted(base.rglob("*")) if p.is_file()},
        "environment_lock_sha256": sha256_file(paths["environment_lock"]), "source_sha256": source_hashes()}
    identity["prior_diagnostic_sha256"] = {name: sha256_file(ROOT / name) for name in DIAGNOSTIC_NAMES}
    return split, folds, gates, identity


def preregister(args):
    if not re.fullmatch(r"[0-9a-f]{40}", args.code_revision or ""):
        raise ValueError("Révision Git SHA40 réelle requise")
    if any(getattr(args, name) is None for name in PATH_ARGUMENTS):
        raise ValueError("Tous les chemins de préinscription sont requis")
    paths = {name: str(getattr(args, name).resolve()) for name in PATH_ARGUMENTS}
    split, folds, gates, identity = prerequisites(paths)
    load_fold(paths, split, "train")  # Vérification des 598 WAV avant préinscription, aucun fit.
    reference = read_json(ROOT / "configs/tslm/v1.json")
    registration = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": args.code_revision, "paths": paths, "identity": identity,
        "recipe": RECIPE, "variants": list(VARIANTS), "selection_rule": SELECTION_RULE,
        "threshold_rule": THRESHOLD_RULE, "diagnostic_threshold": 0.5,
        "omitted_B_reason": "train_autopsy_does_not_support_description_dominating_loss_or_gradients",
        "counts": COUNTS,
        "prior_diagnostic_fits": PRIOR_DIAGNOSTIC_FITS,
        "base_identity": {k: reference[k] for k in ("base_model", "base_revision", "architecture",
                          "opentslm_revision", "timenet_revision", "protocol")},
        "folds": folds["folds"], "gate_provenance": gates,
        "official_validation_for_selection": False, "test_or_external_data_read": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "preregistration.json", registration)
    return {"preregistered": True, "preregistration_sha256": sha256_file(args.output / "preregistration.json")}


def context(output):
    registration = read_json(output / "preregistration.json")
    if (registration.get("schema") != SCHEMA or registration.get("recipe") != RECIPE
            or registration.get("variants") != list(VARIANTS)
            or registration.get("selection_rule") != SELECTION_RULE
            or registration.get("threshold_rule") != THRESHOLD_RULE
            or registration.get("counts") != COUNTS
            or registration.get("prior_diagnostic_fits") != PRIOR_DIAGNOSTIC_FITS
            or registration.get("diagnostic_threshold") != 0.5
            or registration.get("official_validation_for_selection") is not False
            or registration.get("test_or_external_data_read") is not False):
        raise ValueError("Préinscription absente ou différente de la campagne autorisée")
    split, folds, gates, identity = prerequisites(registration["paths"])
    if identity != registration["identity"] or folds["folds"] != registration["folds"]:
        raise ValueError("Code, cache, gates, base ou folds modifiés depuis la préinscription")
    return {"output": output, "registration": registration, "split": split,
            "gates": gates, "preregistration_sha256": sha256_file(output / "preregistration.json")}


def initialize_training(base, variant):
    import torch
    from pipe.tslm.model import AcousticQwenSP
    random.seed(RECIPE["seed"])
    np.random.seed(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RECIPE["seed"])
    # Nouvelle instance ET nouvel optimiseur pour chaque fold et pour le refit.
    model = AcousticQwenSP(base, device="cuda", single_clip_acoustic_encoding=True,
                          amplitude_evidence=variant == "C")
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": RECIPE["encoder_lr"]},
        {"params": model.projector.parameters(), "lr": RECIPE["projector_lr"]},
    ], weight_decay=RECIPE["weight_decay"])
    if optimizer.state or any(p.requires_grad for p in model.llm.parameters()):
        raise ValueError("Optimiseur non neuf ou Qwen non gelé")
    return model, optimizer


def verify_runtime(expected):
    """Même pile numérique que les deux gates, sans changer ses options."""
    import torch
    actual = {"python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": "cuda",
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_device": torch.cuda.get_device_name()}
    for package in ("transformers", "scikit-learn", "timenet", "opentslm"):
        try:
            actual[package] = packages.version(package)
        except packages.PackageNotFoundError:
            actual[package] = "not-installed-as-distribution"
    if actual != expected:
        raise ValueError("Runtime différent de celui des gates : " + str(
            {k: (expected.get(k), actual.get(k)) for k in set(expected) | set(actual)
             if expected.get(k) != actual.get(k)}))


def examples(rows, series, amplitudes, metadata, *, training=False):
    from pipe.tslm.predict import model_input_for_model
    from pipe.tslm.preprocessing import target_text
    result = []
    for row in rows:
        cid = row["clip_id"]
        item = model_input_for_model(series[cid], metadata,
                                    amplitude_features=amplitudes[cid] if metadata["amplitude_evidence"] else None)
        if training:
            item["answer"] = target_text(row["label"], series[cid])
        result.append(item)
    return result


def score_rows(model, rows, series, amplitudes, metadata):
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate as collate
    model.eval()
    probabilities = {}
    for row in rows:
        batch = collate(examples([row], series, amplitudes, metadata), normalize=False)
        values = model.score_probability_leak(batch)
        if len(values) != 1 or not np.isfinite(values[0]) or not 0 <= values[0] <= 1:
            raise ValueError("Score absent, non fini ou hors bornes")
        probabilities[row["clip_id"]] = float(values[0])
    return probabilities


def metric_report(rows, probabilities, threshold=0.5):
    if set(probabilities) != {r["clip_id"] for r in rows} or len(rows) != len(probabilities):
        raise ValueError("Couverture des scores incomplète ou répétée")
    y = np.array([int(r["label"] == "leak") for r in rows])
    scores = np.array([probabilities[r["clip_id"]] for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Probabilités invalides")
    result = metrics.evaluate(y, scores, groups, threshold)
    _, gy, gs = metrics.aggregate_clusters(y, scores, groups)
    result.update(group_roc_auc_full=metrics.roc_auc(gy, gs), clip_roc_auc_full=metrics.roc_auc(y, scores))
    return result


def fit_tslm(ctx, variant, fold_id=None):
    import torch
    from pipe.tslm.train import complete_bundle, tensor_state_hash
    registration, output = ctx["registration"], ctx["output"]
    directory = output / (f"cv/{variant}/fold-{fold_id}" if fold_id is not None else "final/tslm")
    previous = finished(directory, preregistration_sha256=ctx["preregistration_sha256"])
    if previous is not None:
        return previous
    metadata = {**registration["base_identity"], **variant_metadata(variant)}
    rows, series, amplitudes, _ = load_fold(registration["paths"], ctx["split"], "train")
    if fold_id is None:
        selected = finished(output / "selection", preregistration_sha256=ctx["preregistration_sha256"])
        if selected is None or selected["selected_variant"] != variant:
            raise ValueError("Refit interdit avant sélection complète du candidat exact")
        training, heldout = rows, []
    else:
        if type(fold_id) is not int or fold_id not in (0, 1, 2):
            raise ValueError("Fold interne 0, 1 ou 2 requis")
        fold = registration["folds"][fold_id]
        by_id = {r["clip_id"]: r for r in rows}
        training = [by_id[cid] for cid in fold["train_ids"]]
        heldout = [by_id[cid] for cid in fold["heldout_ids"]]
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(directory / "started.json", {"variant": variant, "fold_id": fold_id,
        "preregistration_sha256": ctx["preregistration_sha256"], "training_ids": [r["clip_id"] for r in training],
        "heldout_ids": [r["clip_id"] for r in heldout], "pid": os.getpid(), "hostname": platform.node()})
    model, optimizer = initialize_training(registration["paths"]["base"], variant)
    before = {name: tensor_state_hash(getattr(model, name)) for name in ("encoder", "projector", "llm")}
    gate = ctx["gates"][variant]
    verify_runtime(gate["runtime"])
    if (before["llm"] != gate["state_sha256"]["llm"]
            or any(before[name] == gate["state_sha256"][name] for name in ("encoder", "projector"))
            or model.scoring_spec() != gate["scoring_spec"]):
        raise ValueError("Initialisation non fraîche, base Qwen ou méthode de score différente du gate")
    samples = examples(training, series, amplitudes, metadata, training=True)
    steps, first_loss, last_loss = 0, None, None
    torch.cuda.reset_peak_memory_stats()
    with (directory / "training.jsonl").open("x") as log:
        for epoch in range(1, RECIPE["epochs"] + 1):
            for record in train_epoch(model, optimizer, samples, RECIPE, epoch):
                steps += 1
                first_loss = record["loss"] if first_loss is None else first_loss
                last_loss = record["loss"]
                record = {"step": steps, **record, "elapsed_seconds": time.monotonic() - started}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
    after = {name: tensor_state_hash(getattr(model, name)) for name in before}
    if (before["llm"] != after["llm"] or any(p.grad is not None for p in model.llm.parameters())
            or any(before[name] == after[name] for name in ("encoder", "projector"))
            or steps != RECIPE["epochs"] * ((len(training) + RECIPE["batch_size"] - 1) // RECIPE["batch_size"])):
        raise ValueError("Steps, adaptation acoustique ou gel Qwen invalides")
    temporal = directory / "temporal.pt"
    torch.save({"encoder_state": model.encoder.state_dict(), "projector_state": model.projector.state_dict()}, temporal)
    probabilities = score_rows(model, heldout, series, amplitudes, metadata) if heldout else {}
    result = {"variant": variant, "fold_id": fold_id, "preregistration_sha256": ctx["preregistration_sha256"],
        "source_fold": "train", "training_ids": [r["clip_id"] for r in training],
        "heldout_ids": [r["clip_id"] for r in heldout], "steps": steps, "epochs": RECIPE["epochs"],
        "first_loss": first_loss, "last_loss": last_loss, "state_sha256_before": before,
        "state_sha256_after": after, "scoring_spec": model.scoring_spec(),
        "temporal_sha256": sha256_file(temporal), "pid": os.getpid(), "hostname": platform.node(),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(), "validation_or_test_used": False}
    if heldout:
        write_predictions(directory / "predictions.csv", [r["clip_id"] for r in heldout], probabilities)
        result["metrics"] = metric_report(heldout, probabilities)
    else:
        bundle = directory / "bundle"
        bundle.mkdir()
        configuration = {"recipe": RECIPE, "variant": variant, "base": registration["base_identity"]}
        metadata.update(model_version=f"pipe-qwen3.5-4b-v2-{registration['code_revision'][:8]}-{variant.lower()}-e4",
            training_commit=registration["code_revision"], code_revision=registration["code_revision"],
            config_hash=digest(configuration),
            scoring_spec=model.scoring_spec(), manifest_sha256=ctx["split"].sha256,
            training_clip_ids=result["training_ids"], training_groups=sorted({r["group_id"] for r in training}),
            n_configs_compared=2, n_comparison_fits=6, n_final_fits=1, selected_variant=variant,
            test_labels_not_used_for_tuning=True, calibration="none", purpose="v2_selected_by_train_internal_cv")
        model.llm.save_pretrained(bundle / "base", safe_serialization=True, max_shard_size="3GB")
        model.tokenizer.save_pretrained(bundle / "base")
        import shutil
        shutil.copyfile(temporal, bundle / "temporal.pt")
        write_json(bundle / "metadata.json", metadata)
        write_json(bundle / "config.json", configuration)
        write_json(bundle / "scoring_spec.json", model.scoring_spec())
        write_json(bundle / "training-report.json", result)
        result["checkpoint_checksums_sha256"] = complete_bundle(bundle, Path(registration["paths"]["base"]),
            Path(registration["paths"]["environment_lock"]), provenance_overrides=provenance_overrides(variant))
    if source_hashes() != registration["identity"]["source_sha256"]:
        raise ValueError("Les sources ont changé pendant l'entraînement")
    return finish(directory, result)


def compare_c1(ctx):
    directory = ctx["output"] / "cv/C1"
    if (previous := finished(directory, preregistration_sha256=ctx["preregistration_sha256"])) is not None:
        return previous
    rows, _, _, X = load_fold(ctx["registration"]["paths"], ctx["split"], "train")
    directory.mkdir(parents=True, exist_ok=False)
    result = v2_c1.compare(rows, X, ctx["registration"]["folds"])
    write_json(directory / "comparison.json", result)
    return finish(directory, {**result, "preregistration_sha256": ctx["preregistration_sha256"]})


def choose_variant(candidates):
    if set(candidates) != set(VARIANTS):
        raise ValueError("Les deux candidats A/C complets sont requis")
    summaries = []
    for variant in VARIANTS:
        reports = candidates[variant]
        if len(reports) != 3 or sorted(r["fold_id"] for r in reports) != [0, 1, 2]:
            raise ValueError("Trois folds distincts par candidat requis")
        group = [r["metrics"]["group_roc_auc_full"] for r in reports]
        clip = [r["metrics"]["clip_roc_auc_full"] for r in reports]
        if not np.isfinite([*group, *clip]).all():
            raise ValueError("AUC non définie : aucune sélection de remplacement")
        summaries.append({"variant": variant, "mean_group_roc_auc": float(np.mean(group)),
                          "mean_clip_roc_auc": float(np.mean(clip)), "fold_metrics": [r["metrics"] for r in reports]})
    return min(summaries, key=lambda r: (-r["mean_group_roc_auc"], VARIANTS.index(r["variant"])))["variant"], summaries


def select(ctx):
    directory = ctx["output"] / "selection"
    if (previous := finished(directory, preregistration_sha256=ctx["preregistration_sha256"])) is not None:
        return previous
    candidates = {v: [finished(ctx["output"] / f"cv/{v}/fold-{f}",
                               preregistration_sha256=ctx["preregistration_sha256"])
                       for f in range(3)] for v in VARIANTS}
    c1 = finished(ctx["output"] / "cv/C1", preregistration_sha256=ctx["preregistration_sha256"])
    if c1 is None or any(r is None for reports in candidates.values() for r in reports):
        raise ValueError("Sélection interdite avant les six fits TSLM et les douze fits C1")
    if c1.get("n_configs_compared") != 4 or c1.get("n_fits") != 12:
        raise ValueError("Comparaison C1 incomplète")
    for variant, reports in candidates.items():
        for report, fold in zip(reports, ctx["registration"]["folds"]):
            if (report["preregistration_sha256"] != ctx["preregistration_sha256"]
                    or report["training_ids"] != fold["train_ids"] or report["heldout_ids"] != fold["heldout_ids"]
                    or report["variant"] != variant or report["validation_or_test_used"] is not False):
                raise ValueError("Rapport CV d'un autre candidat/fold/protocole")
    variant, summaries = choose_variant(candidates)
    directory.mkdir(parents=True, exist_ok=False)
    result = {"selected_variant": variant, "selected_C1_C": c1["selected_C"], "candidates": summaries,
        "selection_rule": SELECTION_RULE, "c1_candidates": c1["candidates"],
        "preregistration_sha256": ctx["preregistration_sha256"], "counts": ctx["registration"]["counts"],
        "independent_final_evaluation": False, "pooled_out_of_fold_auc_calculated": False,
        "decision_note": "Examiner aussi rappel/FP/clip avant le refit ; un classement n'est pas une preuve de fiabilité."}
    write_json(directory / "selection.json", result)
    return finish(directory, result)


def fit_final(ctx):
    selection = finished(ctx["output"] / "selection", preregistration_sha256=ctx["preregistration_sha256"])
    if selection is None:
        raise ValueError("Sélection complète requise avant refit")
    directory = ctx["output"] / "final/c1"
    if finished(directory, preregistration_sha256=ctx["preregistration_sha256"]) is None:
        rows, _, _, X = load_fold(ctx["registration"]["paths"], ctx["split"], "train")
        directory.mkdir(parents=True, exist_ok=False)
        scaler, classifier = v2_c1.fit_final(X, np.array([int(r["label"] == "leak") for r in rows]), selection["selected_C1_C"])
        checksum = v2_c1.save_checkpoint(directory / "checkpoint.pkl", scaler, classifier)
        finish(directory, {"checkpoint_sha256": checksum, "selected_C": selection["selected_C1_C"],
            "training_ids": [r["clip_id"] for r in rows], "preregistration_sha256": ctx["preregistration_sha256"]})
    return fit_tslm(ctx, selection["selected_variant"])


def validate(ctx):
    from pipe.tslm.coherent import checkpoint_identity, decision_artifact
    from pipe.tslm.predict import Predictor
    directory = ctx["output"] / "validation"
    if (previous := finished(directory, preregistration_sha256=ctx["preregistration_sha256"])) is not None:
        return previous
    final = finished(ctx["output"] / "final/tslm", preregistration_sha256=ctx["preregistration_sha256"])
    c1 = finished(ctx["output"] / "final/c1", preregistration_sha256=ctx["preregistration_sha256"])
    if final is None or c1 is None or (final["hostname"] == platform.node() and final["pid"] == os.getpid()):
        raise ValueError("Deux refits finis et un nouveau processus sont requis avant validation")
    bundle = ctx["output"] / "final/tslm/bundle"
    verify_runtime(ctx["gates"][final["variant"]]["runtime"])
    predictor = Predictor(bundle)
    if sha256_file(bundle / "checksums.json") != final["checkpoint_checksums_sha256"]:
        raise ValueError("Bundle final modifié")
    # Première lecture des valeurs/WAV de validation officielle dans cette campagne.
    rows, series, amplitudes, X = load_fold(ctx["registration"]["paths"], ctx["split"], "val")
    directory.mkdir(parents=True, exist_ok=False)
    scores = score_rows(predictor.model, rows, series, amplitudes, predictor.metadata)
    scaler, classifier = v2_c1.load_checkpoint(ctx["output"] / "final/c1/checkpoint.pkl",
                                             expected_sha256=c1["checkpoint_sha256"], trusted=True)
    c1_values = v2_c1.predict_probability(scaler, classifier, X)
    c1_scores = {r["clip_id"]: float(p) for r, p in zip(rows, c1_values)}
    thresholds = {}
    for name, probabilities in (("tslm", scores), ("c1", c1_scores)):
        path = directory / f"{name}-predictions.csv"
        write_predictions(path, [r["clip_id"] for r in rows], probabilities)
        y = np.array([int(r["label"] == "leak") for r in rows])
        p = np.array([probabilities[r["clip_id"]] for r in rows])
        groups = np.array([r["group_id"] for r in rows])
        threshold = metrics.pick_threshold(y, p, groups, fold="val")
        if not np.isfinite(threshold):
            raise ValueError("Seuil validation non fini")
        thresholds[name] = float(threshold)
    evidence = {"schema_version": "pipe-threshold-evidence-v1", "fit_fold": "val",
        "threshold": thresholds["tslm"], "threshold_repr": repr(thresholds["tslm"]),
        "model_identity": checkpoint_identity(bundle), "rule": THRESHOLD_RULE,
        "split_sha256": ctx["split"].sha256,
        "validation_predictions_sha256": sha256_file(directory / "tslm-predictions.csv"),
        "threshold_method_sha256": sha256_file(ROOT / "scripts/eval/harness/metrics.py")}
    write_json(directory / "threshold-evidence.json", evidence)
    write_json(directory / "decision.json", decision_artifact(bundle, directory / "threshold-evidence.json",
                                                               predictor.metadata["model_version"] + "-decision-v1"))
    write_json(directory / "c1-threshold.json", {"schema": "pipe-v2-c1-threshold-v1", "fit_fold": "val",
        "threshold": thresholds["c1"], "threshold_repr": repr(thresholds["c1"]), "rule": THRESHOLD_RULE,
        "checkpoint_sha256": c1["checkpoint_sha256"], "split_sha256": ctx["split"].sha256,
        "validation_predictions_sha256": sha256_file(directory / "c1-predictions.csv"),
        "threshold_method_sha256": evidence["threshold_method_sha256"]})
    if sha256_file(bundle / "checksums.json") != final["checkpoint_checksums_sha256"]:
        raise ValueError("Le calcul de seuil a modifié le bundle")
    return finish(directory, {"preregistration_sha256": ctx["preregistration_sha256"],
        "fresh_process_reload": True, "n_validation_clips": len(rows), "thresholds": thresholds,
        "checkpoint_checksums_sha256": final["checkpoint_checksums_sha256"],
        "calibration": "none", "model_selected_before_validation": True, "test_or_external_data_read": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preregister", "fit-fold", "compare-c1", "select", "fit-final", "validate"))
    parser.add_argument("--output", type=Path, required=True)
    for name in PATH_ARGUMENTS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--code-revision")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.stage == "preregister":
        if args.variant is not None or args.fold is not None:
            parser.error("A/C et les trois folds sont préinscrits ensemble, sans choix partiel")
        result = preregister(args)
    else:
        if any(getattr(args, name) is not None for name in PATH_ARGUMENTS) or args.code_revision is not None:
            parser.error("Les chemins et la révision sont fixés une fois dans preregistration.json")
        if args.stage == "fit-fold" and (args.variant is None or args.fold is None):
            parser.error("fit-fold exige --variant et --fold")
        if args.stage != "fit-fold" and (args.variant is not None or args.fold is not None):
            parser.error("Aucun choix de candidat/fold permis à cette étape")
        ctx = context(args.output)
        result = (fit_tslm(ctx, args.variant, args.fold) if args.stage == "fit-fold" else
                  {"compare-c1": compare_c1, "select": select, "fit-final": fit_final, "validate": validate}[args.stage](ctx))
    print(json.dumps({"stage": args.stage, "output": str(args.output), "completed": True,
                      "result_keys": sorted(result)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
