"""Audit descriptif V1 figée : 208 validation puis 194 test, sans réentraînement."""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from export_run import FROZEN_AUDIT_SHA256, ROOT, sha256_file
from harness import contract, metrics, split_loader
from evaluate_predictions import vectors


def load_expected_audio_md5(manifests):
    """Empreintes d'intégrité uniquement côté adaptateur, jamais entrées modèle."""
    path = Path(manifests) / "split_v2_audit.csv"
    if sha256_file(path) != FROZEN_AUDIT_SHA256:
        raise ValueError("Mapping audio différent du manifeste gelé")
    expected = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            clip_id, digest = row["clip_id"], row["md5"]
            if (clip_id in expected or len(digest) != 32
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Identifiant répété ou MD5 invalide dans l'audit")
            expected[clip_id] = digest
    return expected


def verify_audio_bytes(raw, clip_id, expected_md5):
    if hashlib.md5(raw).hexdigest() != expected_md5[clip_id]:
        raise ValueError(f"Intégrité audio différente du WAV gelé : {clip_id}")


def validation_threshold(split, run, report_path, supplied=None):
    """Réutiliser le seuil plein du harness, vérifier sa dérivation sur val seule."""
    report = json.loads(Path(report_path).read_text())
    source = report["runs"][run.run_id]
    if (report["split"]["sha256"] != split.sha256 or source["run_id"] != run.run_id
            or any(source[key] != run.metadata[key] for key in ("training_commit", "config_hash"))):
        raise ValueError("Rapport de seuil provenant d'un autre split ou modèle")
    y, scores, groups, _ = vectors(split, run, "val")
    full = float(metrics.pick_threshold(y, scores, groups, fold="val"))
    declared = float(source["folds"]["val"]["threshold"])
    if (not math.isfinite(full) or declared != full
            or source["threshold_recomputed_on_val"] != round(full, 6)
            or (supplied is not None and supplied != full)):
        raise ValueError("Seuil différent du seuil validation exact ; ne pas employer l'arrondi du rapport")
    return full


def exception_record(exc):
    return {"type": type(exc).__name__, "code": getattr(exc, "code", None), "message": str(exc)}


@contextmanager
def capture_generation(model):
    """Observer le retour exact sans modifier arguments, valeur renvoyée ou erreurs."""
    original, calls = model.generate, []

    def capture(*args, **kwargs):
        started = time.perf_counter()
        event = {"returned": None, "error": None}
        try:
            value = original(*args, **kwargs)
            event["returned"] = value
            return value
        except Exception as exc:
            event["error"] = exception_record(exc)
            raise
        finally:
            event["elapsed_ms"] = (time.perf_counter() - started) * 1000
            calls.append(event)

    model.generate = capture
    try:
        yield calls
    finally:
        model.generate = original


def grade_text(payload, calls, band, saved_probability, threshold, pattern):
    """Une panne/abstention ne disparaît pas du dénominateur de fidélité textuelle."""
    returned = calls[0]["returned"] if len(calls) == 1 else None
    exact = returned[0] if isinstance(returned, (list, tuple)) and len(returned) == 1 else None
    match = pattern.fullmatch(exact.strip()) if isinstance(exact, str) else None
    valid = bool(payload is not None and match is not None and payload.get("abstained") is False)
    generated_class = match[1] if valid else None
    decision = "leak" if saved_probability >= threshold else "no_leak"
    expected = f"Greatest mean spectral energy: {band} Hz." if band is not None else None
    return {"generated_text_exact": exact, "format_valid": valid,
            "generated_class": generated_class,
            "abstained": payload.get("abstained") if payload is not None else None,
            "expected_measured_description": expected,
            "correct_band": bool(valid and expected is not None and match[2] == expected),
            "saved_t0_probability_leak": saved_probability, "saved_t0_threshold_decision": decision,
            "class_decision_disagreement": (generated_class != decision) if valid else None}


def latency_summary(values):
    if not values:
        return {"n": 0, "median_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    return {"n": len(values), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "min_ms": min(values), "max_ms": max(values)}


def summarize(records, expected_counts):
    """Rapports séparés, taux de fidélité sur tous les clips, erreurs comprises."""
    result = {}
    for fold in ("val", "test"):
        rows = [row for row in records if row["fold"] == fold]
        total = len(rows)
        if total != expected_counts[fold] or len({row["clip_id"] for row in rows}) != total:
            raise ValueError(f"Couverture incomplète ou doublons : {fold}")
        errors = [row for row in rows if row["error"] is not None]
        valid = sum(row["format_valid"] for row in rows)
        correct = sum(row["correct_band"] for row in rows)
        comparable = sum(row["class_decision_disagreement"] is not None for row in rows)
        disagreements = sum(row["class_decision_disagreement"] is True for row in rows)
        result[fold] = {"n_expected": expected_counts[fold], "n_attempted": total,
            "n_errors": len(errors), "n_abstentions": sum(row["abstained"] is True for row in rows),
            "n_valid_format": valid, "valid_format_rate_all_clips": valid / total,
            "n_correct_band": correct, "correct_band_rate_all_clips": correct / total,
            "n_wrong_band_among_valid": valid - correct,
            "n_no_usable_text": total - valid,
            "n_comparable_class_decisions": comparable, "n_class_disagreements": disagreements,
            "class_disagreement_rate_comparable": disagreements / comparable if comparable else None,
            "n_class_comparisons_unavailable": total - comparable,
            "error_counts": dict(Counter(row["error"]["type"] for row in errors)),
            "error_clip_ids": [row["clip_id"] for row in errors],
            "api_latency_after_warmup_all_attempts": latency_summary(
                [row["api_elapsed_ms"] for row in rows if row["api_elapsed_ms"] is not None]),
            "api_latency_after_warmup_successful_payloads": latency_summary(
                [row["api_elapsed_ms"] for row in rows if row["api_payload"] is not None]),
            "generation_latency_after_warmup_successful_calls": latency_summary(
                [call["elapsed_ms"] for row in rows for call in row["generation_calls"] if call["error"] is None])}
    return result


def audit(args):
    if args.output.exists():
        raise FileExistsError("Dossier d'audit existant : refus d'écraser")
    if len(args.code_revision) != 40 or any(c not in "0123456789abcdef" for c in args.code_revision):
        raise ValueError("Révision complète du code d'audit requise")
    expected_md5 = load_expected_audio_md5(args.manifests)
    split = split_loader.load_split(args.manifests)
    if set(expected_md5) != {clip.clip_id for clip in split.clips}:
        raise ValueError("Couverture des empreintes audio différente du split gelé")
    run = contract.load_run(args.run, split, folds=("val", "test"))
    with (args.run / "predictions.csv").open(newline="") as stream:
        if next(csv.reader(stream)) != ["clip_id", "probability_leak"]:
            raise ValueError("Deux colonnes exactes requises dans le run T0")
    clips = [clip for fold in ("val", "test") for clip in sorted(split.fold(fold), key=lambda c: c.clip_id)]
    counts = dict(Counter(clip.fold for clip in clips))
    if counts != {"val": 208, "test": 194} or set(run.probabilities) != {clip.clip_id for clip in clips}:
        raise ValueError("Le run doit couvrir exactement les 402 clips val/test")
    if run.metadata.get("transform") != "T0":
        raise ValueError("Le désaccord de classe doit utiliser le run T0 original")
    threshold = validation_threshold(split, run, args.threshold_provenance, args.threshold)
    checkpoint = args.checkpoint.resolve()
    if sha256_file(checkpoint / "checksums.json") != run.metadata["checkpoint_checksums_sha256"]:
        raise ValueError("Checkpoint différent du run T0")
    model_sources = ("src/pipe/tslm/model.py", "src/pipe/tslm/predict.py", "src/pipe/tslm/preprocessing.py",
                     "scripts/timenet/leakless_acoustic/connector.py")
    if any(sha256_file(ROOT / name) != run.metadata["source_sha256"][name] for name in model_sources):
        raise ValueError("Code du modèle/preprocessing différent du T0 figé")
    from pipe.tslm.predict import OUTPUT_PATTERN, Predictor
    from pipe.tslm.preprocessing import decode_wav, measured_band, preprocess_audio

    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {"status": "running", "purpose": "frozen_v1_text_audit_no_model_selection",
        "started_at": datetime.now(timezone.utc).isoformat(), "audit_commit": args.code_revision,
        "run_id": run.run_id, "checkpoint": str(checkpoint),
        "checkpoint_checksums_sha256": run.metadata["checkpoint_checksums_sha256"],
        "temporal_sha256": run.metadata["temporal_sha256"], "config_hash": run.metadata["config_hash"],
        "scoring_spec": run.metadata["scoring_spec"], "training_commit": run.metadata["training_commit"],
        "split_sha256": split.sha256, "split_audit_sha256": FROZEN_AUDIT_SHA256,
        "run_files_sha256": {name: sha256_file(args.run / name) for name in ("metadata.json", "predictions.csv")},
        "threshold": threshold, "threshold_repr": repr(threshold),
        "threshold_source": "harness metrics.json runs[run_id].folds.val.threshold, verified on val only",
        "threshold_provenance_sha256": sha256_file(args.threshold_provenance),
        "threshold_rule": "harness.metrics.pick_threshold, validation cluster median macro-F1, score >= threshold",
        "audio_integrity_rule": "MD5 des octets WAV vérifié par clip_id dans split_v2_audit.csv dont le SHA256 est gelé",
        "fold_order": ["val", "test"], "expected_counts": counts, "warmup_included_in_metrics": False,
        "numpy_version": np.__version__,
        "source_sha256": {name: sha256_file(ROOT / name) for name in (*model_sources,
            "scripts/tslm/audit_text.py", "scripts/tslm/export_run.py", "scripts/eval/harness/metrics.py",
            "scripts/eval/evaluate_predictions.py", "scripts/eval/harness/contract.py", "scripts/eval/harness/split_loader.py")}}

    def save_provenance():
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    save_provenance()
    try:
        loading_started = time.perf_counter()
        try:
            predictor = Predictor(checkpoint, device=args.device)
        finally:
            provenance["checkpoint_load_elapsed_ms"] = (time.perf_counter() - loading_started) * 1000
        if (predictor.model.scoring_spec() != run.metadata["scoring_spec"]
                or predictor.metadata["config_hash"] != run.metadata["config_hash"]):
            raise ValueError("Modèle rechargé différent du run T0")
        expected = json.loads((checkpoint / "reload-expected.json").read_text())
        if split.by_id()[expected["clip_id"]].fold != "val":
            raise ValueError("Le warmup doit être l'exemple de développement figé")
        records = []
        with capture_generation(predictor.model) as calls:
            try:
                warmup_raw = (checkpoint / "reload-example.wav").read_bytes()
                verify_audio_bytes(warmup_raw, expected["clip_id"], expected_md5)
                warmup = {"clip_id": expected["clip_id"], "input_sha256": hashlib.sha256(warmup_raw).hexdigest(),
                          "api_payload": predictor.predict(warmup_raw).model_dump(mode="json"), "error": None}
            except Exception as exc:
                warmup = {"clip_id": expected["clip_id"], "api_payload": None, "error": exception_record(exc)}
            warmup["generation_calls"] = list(calls)
            provenance["warmup"] = warmup
            save_provenance()
            with (args.output / "raw.jsonl").open("x") as stream:
                for clip in clips:
                    calls.clear()
                    record = {"clip_id": clip.clip_id, "fold": clip.fold, "input_sha256": None,
                              "audio_md5_verified": False,
                              "dsp_band": None, "api_payload": None, "api_elapsed_ms": None,
                              "generation_calls": [], "error": None}
                    stage = "read_and_dsp"
                    try:
                        path = (args.data_root.resolve() / split.path_of(clip.clip_id)).resolve()
                        if not path.is_relative_to(args.data_root.resolve()):
                            raise ValueError("Chemin audio hors data-root")
                        raw = path.read_bytes()
                        record["input_sha256"] = hashlib.sha256(raw).hexdigest()
                        stage = "audio_integrity"
                        verify_audio_bytes(raw, clip.clip_id, expected_md5)
                        record["audio_md5_verified"] = True
                        stage = "dsp"
                        record["dsp_band"] = measured_band(preprocess_audio(decode_wav(raw), 8000))
                        stage = "predict"
                        started = time.perf_counter()
                        try:
                            record["api_payload"] = predictor.predict(raw).model_dump(mode="json")
                        finally:
                            record["api_elapsed_ms"] = (time.perf_counter() - started) * 1000
                    except Exception as exc:
                        record["error"] = {"stage": stage, **exception_record(exc)}
                    record["generation_calls"] = list(calls)
                    record.update(grade_text(record["api_payload"], calls, record["dsp_band"],
                                             run.probabilities[clip.clip_id], threshold, OUTPUT_PATTERN))
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    records.append(record)
                    if len(records) % 20 == 0 or len(records) == len(clips):
                        print(f"Audit texte : {len(records)}/{len(clips)}, fold {clip.fold}", flush=True)
        summary = {"purpose": provenance["purpose"], "threshold": threshold,
                   "warmup_included": False, "warmup_call_succeeded": warmup["error"] is None,
                   "folds": summarize(records, counts),
                   "limitations": ["Description gabarit parmi quatre bandes, pas justification causale de fuite.",
                       "Désaccord classe/score : cohérence de sorties, pas exactitude de classification.",
                       "Latences séquentielles sur clips, pas preuve de débit soutenu ni validation terrain."],
                   "raw_sha256": sha256_file(args.output / "raw.jsonl")}
        with (args.output / "summary.json").open("x") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        provenance["status"] = "complete"
        provenance["summary_sha256"] = sha256_file(args.output / "summary.json")
        provenance["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_provenance()
        print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    except Exception as exc:
        provenance["status"] = "failed"
        provenance["fatal_error"] = exception_record(exc)
        save_provenance()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data-root", "manifests", "run", "output", "threshold-provenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--threshold", type=float, help="Contrôle facultatif, valeur exacte requise (pas l'arrondi)")
    parser.add_argument("--device", default="cuda")
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
