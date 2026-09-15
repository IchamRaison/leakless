"""Exporter l'externe gelé, sans fit ni métriques : parent neuf avec run/ et audit/.

Une invocation = un modèle et une transformation. T0 TSLM conserve toutes les
sorties textuelles ; les stress réemploient strictement ses poids et son seuil.
Les 60 fenêtres de bruit restent une annexe sans cible binaire ni run primaire.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for relative in ("src", "scripts/eval", "scripts/temporal", "scripts/timenet"):
    sys.path.insert(0, str(ROOT / relative))
from export_run import sha256_file, STRESS_SHA256
import external_manifest
import audit_external_overlap as overlap
from prepare_external_holdout import INPUT_FIELDS
from harness import contract, features, split_loader
from stress import BLOCK_SAMPLES, SEED_SCHEME, STRESS_SEED, TRANSFORMS, clip_rng

MANIFEST_SHA256 = "840078010f4023a045a039167f079ef178f6b0cb56f7f280fd2a5760aad1616b"
SCHEMA = "pipe-v2-external-export-v1"
SOURCE_NAMES = (
    "scripts/tslm/export_v2_external.py", "scripts/tslm/run_v2_campaign.py",
    "scripts/tslm/v2_c1.py", "scripts/tslm/check_v2_parity.py",
    "scripts/tslm/diagnose_parity.py", "scripts/eval/external_manifest.py",
    "scripts/eval/audit_external_overlap.py", "scripts/eval/prepare_external_holdout.py",
    "scripts/eval/harness/contract.py", "scripts/eval/harness/split_loader.py",
    "scripts/eval/harness/metrics.py", "scripts/eval/harness/features.py",
    "scripts/temporal/stress.py", "src/pipe/tslm/coherent.py",
    "src/pipe/tslm/predict.py", "src/pipe/tslm/model.py", "src/pipe/tslm/preprocessing.py",
    "scripts/timenet/leakless_acoustic/connector.py")


def read_json(path):
    return external_manifest.read_json(path, bound=30_000_000)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def source_hashes():
    return {name: sha256_file(ROOT / name) for name in SOURCE_NAMES}


def finite_probability(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.floating))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Score non fini/hors [0,1] : aucune valeur de secours autorisée")
    return float(value)


def validate_threshold(receipt, model, directory, complete, rule):
    """Lire le seuil plein déjà choisi ; ne pas recalculer de seuil ni de métrique."""
    common = {"fit_fold", "threshold", "threshold_repr", "rule", "split_sha256",
              "validation_predictions_sha256", "threshold_method_sha256"}
    fields = {"schema_version", "model_identity"} if model == "tslm" else {"schema", "checkpoint_sha256"}
    value = receipt.get("threshold")
    if (set(receipt) != common | fields or receipt.get("fit_fold") != "val"
            or type(value) not in (int, float) or not math.isfinite(value)
            or receipt.get("threshold_repr") != repr(float(value))
            or receipt.get("rule") != rule
            or receipt.get("split_sha256") != split_loader.FROZEN_SPLIT_SHA256
            or receipt.get("threshold_method_sha256") != sha256_file(ROOT / "scripts/eval/harness/metrics.py")
            or receipt.get("validation_predictions_sha256") != sha256_file(directory / f"{model}-predictions.csv")
            or value != complete.get("thresholds", {}).get(model)):
        raise ValueError("Reçu de seuil validation incompatible, arrondi ou non lié aux prédictions gelées")
    if ((model == "tslm" and receipt["schema_version"] != "pipe-threshold-evidence-v1")
            or (model == "c1" and receipt["schema"] != "pipe-v2-c1-threshold-v1")):
        raise ValueError("Schéma de seuil incompatible")
    return float(value)


def validate_final_gate(report, reference, reference_sha256, identity, metadata, threshold, expected_ids):
    """Le gate de référence pré-entraînement n'est jamais une preuve du modèle final."""
    from check_v2_parity import CHECKS, SCHEMA as GATE_SCHEMA, summarize_scores, verify_reference
    if (report.get("schema") != GATE_SCHEMA or report.get("all_checks_pass") is not True
            or report.get("status") != "passed"
            or any(report.get("checks", {}).get(key) is not True for key in CHECKS)
            or report.get("reference_sha256") != reference_sha256):
        raise ValueError("Gate final PASS et référence fraîche intègre obligatoires")
    p = report["provenance"]
    if (p.get("clip_ids") != expected_ids or p.get("score_atol") != 1e-6
            or p.get("seed") != STRESS_SEED
            or p.get("manifest_sha256") != split_loader.FROZEN_SPLIT_SHA256
            or p.get("test_audio_or_cache_opened") is not False
            or p.get("quality_metrics_calculated") is not False
            or p.get("threshold_diagnostic_only") != threshold
            or p.get("scoring_spec") != metadata["scoring_spec"]
            or p.get("amplitude_evidence") != metadata.get("amplitude_evidence")
            or p.get("amplitude_evidence_version") != metadata.get("amplitude_evidence_version")
            or any(p.get(key) != identity[key] for key in (
                "checkpoint_checksums_sha256", "temporal_sha256", "preprocessing_version"))):
        raise ValueError("Gate d'un autre modèle, split, scoring ou seuil que le checkpoint final")
    required_sources = {Path(name).name: value for name, value in source_hashes().items()
                        if Path(name).name in ("model.py", "predict.py", "preprocessing.py", "connector.py",
                                               "features.py", "check_v2_parity.py", "diagnose_parity.py")}
    if any(p.get("source_sha256", {}).get(name) != value for name, value in required_sources.items()):
        raise ValueError("Sources numériques modifiées depuis le gate final")
    summary = summarize_scores(report["scores"], threshold)
    fresh = verify_reference(report, reference)
    if (summary != report.get("score_summary") or fresh != report.get("fresh_process")
            or not summary["adapter_scores_within_atol"] or not summary["batch_scores_within_atol"]
            or summary["threshold_diagnostic"]["decision_flips"]
            or not fresh["all_scores_within_atol"] or fresh["threshold_decision_flip_ids"]):
        raise ValueError("Parité finale ou décision au seuil plein non reproductible")
    return p


def campaign_context(args):
    """Contrôles de fichiers seulement ; n'appelle pas context(), fit() ou evaluate()."""
    import run_v2_campaign as campaign
    from pipe.tslm.coherent import checkpoint_identity, decision_artifact
    directory = args.campaign.resolve()
    registration_path = directory / "preregistration.json"
    registration = read_json(registration_path)
    prereg_sha = sha256_file(registration_path)
    if (registration.get("schema") != campaign.SCHEMA
            or registration.get("recipe") != campaign.RECIPE
            or registration.get("variants") != list(campaign.VARIANTS)
            or registration.get("counts") != campaign.COUNTS
            or registration.get("selection_rule") != campaign.SELECTION_RULE
            or registration.get("threshold_rule") != campaign.THRESHOLD_RULE
            or registration.get("official_validation_for_selection") is not False
            or registration.get("test_or_external_data_read") is not False
            or registration.get("identity", {}).get("source_sha256") != campaign.source_hashes()):
        raise ValueError("Campagne ou sources différentes de la préinscription")
    completed = {name: campaign.finished(directory / relative, preregistration_sha256=prereg_sha)
                 for name, relative in (("tslm", "final/tslm"), ("c1", "final/c1"), ("validation", "validation"))}
    if any(value is None for value in completed.values()):
        raise ValueError("Deux refits finaux et validation complète obligatoires avant export")
    validation = completed["validation"]
    if (validation.get("fresh_process_reload") is not True or validation.get("n_validation_clips") != 208
            or validation.get("model_selected_before_validation") is not True
            or validation.get("test_or_external_data_read") is not False or validation.get("calibration") != "none"):
        raise ValueError("Reload final et seuils validation non établis")
    bundle = directory / "final/tslm/bundle"
    identity = checkpoint_identity(bundle)
    metadata = read_json(bundle / "metadata.json")
    if (metadata.get("purpose") != "v2_selected_by_train_internal_cv"
            or metadata.get("selected_variant") != completed["tslm"].get("variant")
            or metadata.get("selected_variant") not in ("A", "C")
            or metadata.get("n_configs_compared") != 2
            or metadata.get("training_commit") != registration["code_revision"]
            or metadata.get("test_labels_not_used_for_tuning") is not True
            or metadata.get("scoring_spec") != read_json(bundle / "scoring_spec.json")
            or identity["checkpoint_checksums_sha256"] != completed["tslm"].get("checkpoint_checksums_sha256")
            or identity["checkpoint_checksums_sha256"] != validation.get("checkpoint_checksums_sha256")):
        raise ValueError("Bundle non final ou provenance incohérente")
    receipts = {name: read_json(directory / "validation" / filename) for name, filename in
                (("tslm", "threshold-evidence.json"), ("c1", "c1-threshold.json"))}
    thresholds = {name: validate_threshold(receipt, name, directory / "validation", validation,
                                           campaign.THRESHOLD_RULE) for name, receipt in receipts.items()}
    if (receipts["tslm"]["model_identity"] != identity
            or receipts["c1"]["checkpoint_sha256"] != completed["c1"]["checkpoint_sha256"]
            or sha256_file(directory / "final/c1/checkpoint.pkl") != receipts["c1"]["checkpoint_sha256"]):
        raise ValueError("Seuils attachés à d'autres poids finaux")
    decision = read_json(directory / "validation/decision.json")
    if decision != decision_artifact(bundle, directory / "validation/threshold-evidence.json", decision["decision_version"]):
        raise ValueError("Artefact de décision modifié")
    split = split_loader.load_split(args.manifests)
    from diagnose_parity import selected_ids
    gate = validate_final_gate(read_json(args.final_gate), read_json(args.final_gate_reference),
        sha256_file(args.final_gate_reference), identity, metadata, thresholds["tslm"], selected_ids(split, True))
    return {"directory": directory, "registration": registration, "preregistration_sha256": prereg_sha,
            "completed": completed, "bundle": bundle, "model_identity": identity, "metadata": metadata,
            "receipts": receipts, "thresholds": thresholds, "gate": gate}


def external_inputs(directory, manifest_path):
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Manifeste externe différent du gel autorisé")
    split = external_manifest.load_external_manifest(manifest_path)
    frozen = read_json(manifest_path)["preparation"]
    records, provenance = overlap.load_external(directory)
    if (any(provenance.get(k) != frozen[k] for k in provenance)
            or sha256_file(directory / "receipt.json") != frozen["receipt_sha256"]):
        raise ValueError("Préparation externe différente de celle auditée")
    rows = {}
    for name, filename in (("external", "inputs.csv"), ("background", "background_inputs.csv")):
        path = overlap.safe_path(directory, filename)
        if sha256_file(path) != frozen["csv_sha256"][filename]:
            raise ValueError("Mapping des fenêtres différent du manifeste externe")
        rows[name] = external_manifest.read_csv(path, INPUT_FIELDS)
    ids = [c.clip_id for c in split.clips]
    if (len(ids) != 3600 or len(set(ids)) != 3600
            or [r["clip_id"] for r in rows["external"]] != sorted(ids)
            or len(rows["background"]) != 60
            or set(ids) & {r["clip_id"] for r in rows["background"]}):
        raise ValueError("Couverture primaire/annexe incompatible")
    arrays = {str(path.relative_to(directory.resolve())): np.load(path, mmap_mode="r", allow_pickle=False)
              for _, path in records}
    # Les empreintes couvrent aussi les sources RAW et tous les tableaux, avant/après.
    sums = read_json(directory / "checksums.json")
    files = {overlap.safe_path(directory, name): digest for name, digest in sums.items()}
    files.update({directory / "checksums.json": frozen["checksums_sha256"],
                  directory / "receipt.json": frozen["receipt_sha256"], manifest_path: MANIFEST_SHA256})
    verify_files(files)
    return split, rows, arrays, files


def verify_files(expected):
    if any(sha256_file(path) != digest for path, digest in expected.items()):
        raise ValueError("Fichier source, poids ou reçu modifié pendant l'export")


def score_windows(rows, arrays, transform, predict_waveform, raw_path, *, text_audit=False):
    """Adaptateur unique : le callback reçoit seulement un tableau et 8000 Hz."""
    if len({row["clip_id"] for row in rows}) != len(rows):
        raise ValueError("clip_id répété avant toute inférence")
    probabilities, latencies, counters = {}, [], {"attempted": 0, "successful": 0, "errors": 0,
        "fallback_used": 0, "raw_format_valid": 0, "raw_band_matches_dsp": 0,
        "raw_class_matches_decision": 0}
    with Path(raw_path).open("x") as stream:
        for row in rows:
            cid = row["clip_id"]
            record = {"clip_id": cid, "transform": transform, "payload": None, "error": None}
            started = time.perf_counter()
            try:
                if cid in probabilities:
                    raise ValueError("clip_id répété")
                start, size = int(row["start_sample"]), int(row["n_samples"])
                waveform = arrays[row["array_path"]][start:start + size]
                if (size != 8000 or int(row["sample_rate"]) != 8000 or waveform.shape != (8000,)
                        or waveform.dtype != np.dtype("<f8") or not np.isfinite(waveform).all()):
                    raise ValueError("Fenêtre originale float64 finie à 8000 Hz requise")
                transformed = TRANSFORMS[transform]["fn"](waveform, clip_rng(transform, cid))
                if (transformed.shape != (8000,) or transformed.dtype != np.dtype("<f8")
                        or not np.isfinite(transformed).all()):
                    raise ValueError("Transformation officielle non finie ou incompatible")
                record["waveform_sha256"] = hashlib.sha256(np.ascontiguousarray(transformed).tobytes()).hexdigest()
                record["requantized"] = False
                # Aucun clip_id, label, chemin ou groupe n'entre dans ce callback.
                payload = predict_waveform(transformed, 8000)
                record["payload"] = payload
                probability = finite_probability(payload["probability_leak"] if text_audit else payload)
                probabilities[cid] = probability
                counters["successful"] += 1
                if text_audit:
                    counters["fallback_used"] += int(payload.get("fallback_used") is True)
                    for key in ("raw_format_valid", "raw_band_matches_dsp", "raw_class_matches_decision"):
                        counters[key] += int(payload.get("audit", {}).get(key) is True)
            except Exception as exc:
                counters["errors"] += 1
                record["error"] = {"type": type(exc).__name__, "code": getattr(exc, "code", None),
                    "message": str(exc), "coherent_audit": getattr(exc, "coherent_audit", None)}
                # JSON interdit NaN/Inf. Garder explicitement leur représentation
                # brute, jamais remplacer une probabilité invalide par zéro.
                if isinstance(record["payload"], (int, float, np.floating)) and not math.isfinite(record["payload"]):
                    record["invalid_payload_repr"] = repr(record["payload"])
                    record["payload"] = None
            record["latency_ms"] = (time.perf_counter() - started) * 1000
            latencies.append(record["latency_ms"])
            counters["attempted"] += 1
            # Les valeurs invalides restent des erreurs, jamais des scores de secours.
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if counters["attempted"] % 100 == 0:
                print(f"{transform}: {counters['attempted']}/{len(rows)} fenêtres tentées", flush=True)
    return probabilities, {**counters, "expected": len(rows), "text_audit": text_audit,
        "text_denominator_includes_errors_and_invalid": True, "warmup_performed": False,
        "first_call_included_in_latency": True, "latency_ms": {"mean": float(np.mean(latencies)),
            "p50": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95))} if latencies else None}


def state_hashes(predictor):
    from pipe.tslm.train import tensor_state_hash
    return {name: tensor_state_hash(getattr(predictor.model, name)) for name in ("encoder", "projector", "llm")}


def export(args):
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Parent de sortie neuf obligatoire ; aucun écrasement")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id) or not re.fullmatch(r"[0-9a-f]{40}", args.code_revision):
        raise ValueError("run_id sûr et code-revision SHA40 requis")
    if args.audit_text and (args.model != "tslm" or args.transform != "T0"):
        raise ValueError("Audit textuel réservé au run principal TSLM T0")
    if args.model == "c1" and args.trusted_c1_checkpoint is not True:
        raise ValueError("Autorisation explicite --trusted-c1-checkpoint requise pour le pickle produit par cette campagne")
    if sha256_file(ROOT / "scripts/temporal/stress.py") != STRESS_SHA256 or BLOCK_SAMPLES != 250:
        raise ValueError("Transformations temporelles gelées modifiées")
    ctx = campaign_context(args)
    directory = args.external.resolve()
    split, rows, arrays, input_files = external_inputs(directory, args.manifest.resolve())
    sources = source_hashes()
    receipt_path = ctx["directory"] / "validation" / ("threshold-evidence.json" if args.model == "tslm" else "c1-threshold.json")
    checkpoint = ctx["bundle"] if args.model == "tslm" else ctx["directory"] / "final/c1/checkpoint.pkl"
    text_audit = args.model == "tslm" and args.transform == "T0"
    provenance = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "transform": args.transform, "run_id": args.run_id,
        "export_commit": args.code_revision, "source_sha256": sources,
        "preregistration_sha256": ctx["preregistration_sha256"],
        "final_gate_sha256": sha256_file(args.final_gate),
        "final_gate_reference_sha256": sha256_file(args.final_gate_reference),
        "threshold_provenance_sha256": sha256_file(receipt_path),
        "external_manifest_sha256": split.sha256, "threshold": ctx["thresholds"][args.model],
        "external_preparation": read_json(args.manifest)["preparation"],
        "threshold_repr": repr(ctx["thresholds"][args.model]), "threshold_fitted_here": False,
        "metrics_calculated": False, "background_in_primary": False,
        "background_has_binary_target": False, "waveform_dtype": "float64", "requantized": False,
        "numpy_version": np.__version__, "python": platform.python_version(),
        "stress_seed": STRESS_SEED, "stress_seed_scheme": SEED_SCHEME,
        "text_audit": text_audit, "calibration": "none"}
    if args.model == "tslm":
        provenance.update(model_identity=ctx["model_identity"], **{key: ctx["model_identity"][key]
            for key in ("checkpoint_checksums_sha256", "temporal_sha256", "config_hash", "scoring_spec_sha256")},
            scoring_spec=ctx["metadata"]["scoring_spec"], n_configs_compared=2)
    else:
        provenance.update(checkpoint_sha256=ctx["receipts"]["c1"]["checkpoint_sha256"],
                          n_configs_compared=4, feature_names=list(features.C1_NAMES))
    args.output.mkdir(parents=True, exist_ok=False)
    audit = args.output / "audit"
    audit.mkdir()
    write_json(audit / "started.json", provenance)
    try:
        started = time.perf_counter()
        if args.model == "tslm":
            import run_v2_campaign as campaign
            campaign.verify_runtime(ctx["gate"]["runtime"])
            if text_audit:
                from pipe.tslm.coherent import CoherentPredictor
                backend = CoherentPredictor(checkpoint, ctx["directory"] / "validation/decision.json", receipt_path)
                predict_waveform, snapshot = backend.predict_waveform, backend.state_hashes
            else:
                from pipe.tslm.predict import Predictor
                backend = Predictor(checkpoint)
                predict_waveform, snapshot = backend.score_waveform, lambda: state_hashes(backend)
        else:
            import v2_c1
            scaler, classifier = v2_c1.load_checkpoint(checkpoint,
                expected_sha256=provenance["checkpoint_sha256"], trusted=True)
            def predict_waveform(waveform, sample_rate):
                if sample_rate != 8000:
                    raise ValueError("C1 exige 8000 Hz")
                return float(v2_c1.predict_probability(scaler, classifier, features.c1_envelope(waveform)[None, :])[0])
            def snapshot():
                return {f"{name}.{key}": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                    for name, obj in (("scaler", scaler), ("classifier", classifier))
                    for key, value in vars(obj).items() if isinstance(value, np.ndarray)}
        provenance["model_load_ms"] = (time.perf_counter() - started) * 1000
        before = snapshot()
        if args.model == "tslm" and before != ctx["gate"]["state_sha256"]:
            raise ValueError("Poids rechargés différents du gate final")
        probabilities, summaries = {}, {}
        for partition in ("external", "background"):
            probabilities[partition], summaries[partition] = score_windows(rows[partition], arrays,
                args.transform, predict_waveform, audit / f"{partition}.jsonl", text_audit=text_audit)
        after = snapshot()
        provenance.update(state_sha256_before=before, state_sha256_after=after,
                          weights_unchanged=before == after, partitions=summaries)
        if before != after:
            raise ValueError("Poids en mémoire modifiés pendant l'export")
        if source_hashes() != sources:
            raise ValueError("Code source modifié pendant l'export")
        verify_files(input_files)
        # Revalider les poids/fichiers/seuils et les reçus terminés, pas les résultats qualité.
        again = campaign_context(args)
        if (again["model_identity"] != ctx["model_identity"]
                or again["preregistration_sha256"] != ctx["preregistration_sha256"]
                or sha256_file(receipt_path) != provenance["threshold_provenance_sha256"]
                or sha256_file(args.final_gate) != provenance["final_gate_sha256"]
                or sha256_file(args.final_gate_reference) != provenance["final_gate_reference_sha256"]):
            raise ValueError("Campagne ou reçus modifiés pendant l'export")
        if any(summaries[name]["errors"] or len(probabilities[name]) != len(rows[name]) for name in rows):
            raise ValueError("Export incomplet : erreurs conservées, aucun run conforme ni score de secours produit")
        with (audit / "background-predictions.csv").open("x", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("clip_id", "probability_leak"))
            writer.writerows((cid, format(value, ".17g")) for cid, value in sorted(probabilities["background"].items()))
        run_dir = contract.write_run(args.output / "run", run_id=args.run_id,
            model_name="AcousticQwenSP V2" if args.model == "tslm" else "C1 enveloppe V2",
            checkpoint=str(checkpoint), training_commit=ctx["registration"]["code_revision"], split=split,
            threshold_rule="aucun seuil appliqué aux probabilités ; seuil validation gelé utilisé seulement pour la restitution",
            probabilities=probabilities["external"], extra={key: value for key, value in provenance.items()
                if key not in {"run_id", "threshold", "threshold_repr", "created_at", "partitions"}},
            external_manifest_name=external_manifest.NAME, external_manifest_sha256=MANIFEST_SHA256)
        run = contract.load_run(run_dir, split, folds=("external",), external_manifest_name=external_manifest.NAME,
                                external_manifest_sha256=MANIFEST_SHA256)
        # Contrôle de contrat uniquement : cette fonction ne calcule aucune métrique.
        from evaluate_predictions import load_external_threshold
        load_external_threshold(run, receipt_path)
        if {p.name for p in run_dir.iterdir()} != {"metadata.json", "predictions.csv"}:
            raise ValueError("Le run doit contenir exactement deux fichiers")
        provenance.update(status="complete", run_sha256={name: sha256_file(run_dir / name)
            for name in ("metadata.json", "predictions.csv")})
        write_json(audit / "summary.json", provenance)
        return provenance
    except Exception as exc:
        write_json(audit / "failure.json", {**provenance, "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)}})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "external", "manifest", "final-gate", "final-gate-reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, default=ROOT / "manifests")
    parser.add_argument("--model", choices=("tslm", "c1"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--transform", choices=tuple(TRANSFORMS), default="T0")
    parser.add_argument("--audit-text", action="store_true", help="Toujours activé pour TSLM T0, même sans ce drapeau")
    parser.add_argument("--trusted-c1-checkpoint", action="store_true")
    export(parser.parse_args(argv))


if __name__ == "__main__":
    main()
