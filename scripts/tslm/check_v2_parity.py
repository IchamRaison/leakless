"""Porte V2 : parité numérique développement uniquement, sans métrique qualité."""
import argparse
from datetime import datetime, timezone
from importlib import metadata as packages
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np

from diagnose_parity import (ROOT, SCORE_ATOL, TRAIN_CASE, array_difference,
                             array_info, load_expected_audio_md5, score_spread,
                             selected_ids, sha256_file, split_loader,
                             verify_audio_bytes, write_json)

SCHEMA = "pipe-parity-v2"
SEED = 20260912
BATCH_SIZES = (1, 2, 4)
ADAPTERS = ("series", "waveform", "wav_bytes")
CHECKS = ("canonical_inputs_exact", "bundle_unchanged", "adapter_scores_within_atol",
          "batch_scores_within_atol", "fresh_process_verified")


def require_canonical(metadata, canonical_version):
    if metadata.get("preprocessing_version") != canonical_version:
        raise ValueError("Le gate refuse un bundle legacy : version canonique explicite requise")


def exact_arrays(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)


def compare_routes(routes):
    canonical = routes["canonical"]
    comparisons = {name: array_difference(canonical, value)
                   for name, value in routes.items() if name != "canonical"}
    return {"exact": all(exact_arrays(canonical, value) for value in routes.values()),
            "comparisons": comparisons,
            "series": {name: array_info(value) for name, value in routes.items()}}


def prepare_inputs(args):
    from leakless_acoustic import connector
    from pipe.tslm import model, predict, preprocessing
    from pipe.tslm.campaign import load_cache
    from timenet.reader.reader import TimeFReader
    from timenet.registry.version import DatasetVersion

    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    require_canonical(metadata, preprocessing.CANONICAL_VERSION)
    split = split_loader.load_split(args.manifests)
    ids = selected_ids(split, True)
    expected_md5 = load_expected_audio_md5(args.manifests)
    if set(expected_md5) != {c.clip_id for c in split.clips}:
        raise ValueError("Empreintes audio incomplètes")
    use_amplitude = metadata.get("amplitude_evidence", False)
    if type(use_amplitude) is not bool:
        raise ValueError("Politique d'entrée booléenne requise")
    caches, historical, amplitudes, amplitude_texts = {}, {}, {}, {}
    for fold in ("train", "val"):
        rows = [{"clip_id": c.clip_id, "fold": fold} for c in split.fold(fold)]
        caches.update(load_cache(args.prepared, fold, rows, preprocessing.CANONICAL_VERSION))
        historical.update(load_cache(args.historical_prepared, fold, rows, preprocessing.VERSION))
        if use_amplitude:
            with np.load(args.prepared / f"{fold}.npz", allow_pickle=False) as cache:
                if str(cache["amplitude_evidence_version"]) != preprocessing.AMPLITUDE_EVIDENCE_VERSION:
                    raise ValueError("Version des mesures d'amplitude du cache incompatible")
                values, texts = cache["amplitude_features"], cache["amplitude_text"]
                if (values.shape != (len(rows), 9) or values.dtype != np.float64
                        or texts.shape != (len(rows),) or not np.isfinite(values).all()):
                    raise ValueError("Mesures/texte d'amplitude du cache incompatibles")
                amplitudes.update(zip(cache["ids"].tolist(), values))
                amplitude_texts.update(zip(cache["ids"].tolist(), texts.tolist()))
    timef = {}
    with TimeFReader(DatasetVersion.open_local(args.timef_version)) as reader:
        # Filtrage avant lecture des signaux ; ni tasks, ni verify(), ni test.
        for record in reader.iter_records(record_ids=ids, with_annotations=False):
            if record.record_id not in ids or record.record_id in timef:
                raise ValueError("Record TimeF inattendu ou répété")
            timef[record.record_id] = record.time_series[0].to_numpy()
    if set(timef) != set(ids):
        raise ValueError("Couverture TimeF incomplète")
    inputs, records = {}, []
    root = args.data_root.resolve()
    for cid in ids:
        path = (root / split.path_of(cid)).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Chemin WAV hors data-root")
        raw = path.read_bytes()
        verify_audio_bytes(raw, cid, expected_md5)
        waveform = preprocessing.decode_wav(raw)
        canonical = predict.preprocess_for_model(waveform, 8000, metadata)
        routes = {"canonical": canonical, "canonical_cache": caches[cid],
                  "historical_cache": historical[cid],
                  "timef": preprocessing.band_series(timef[cid]),
                  "explicit_canonical": preprocessing.preprocess_audio(
                      waveform, 8000, version=preprocessing.CANONICAL_VERSION)}
        route_check = compare_routes(routes)
        amplitude = None
        if use_amplitude:
            amplitude_routes = {"canonical": preprocessing.amplitude_features(waveform, 8000),
                                "canonical_cache": amplitudes[cid],
                                "timef": preprocessing.amplitude_from_normalized(timef[cid])}
            amplitude_check = compare_routes(amplitude_routes)
            texts = {name: preprocessing.amplitude_text(value) for name, value in amplitude_routes.items()}
            texts["serialized_cache"] = amplitude_texts[cid]
            amplitude_check["texts"] = texts
            amplitude_check["text_exact"] = len(set(texts.values())) == 1
            route_check["amplitude"] = amplitude_check
            route_check["exact"] &= amplitude_check["exact"] and amplitude_check["text_exact"]
            amplitude = amplitudes[cid]
        inputs[cid] = (raw, waveform, canonical, amplitude)
        records.append({"clip_id": cid, "input_sha256": hashlib.sha256(raw).hexdigest(),
                        "waveform": array_info(waveform), "timef": array_info(timef[cid]),
                        **route_check})
    sources = {"check_v2_parity.py": Path(__file__),
               "diagnose_parity.py": Path(__file__).with_name("diagnose_parity.py"),
               "model.py": Path(model.__file__), "predict.py": Path(predict.__file__),
               "preprocessing.py": Path(preprocessing.__file__),
               "connector.py": Path(connector.__file__),
               "features.py": ROOT / "scripts/eval/harness/features.py"}
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(),
                  "process": {"hostname": platform.node(), "pid": os.getpid()},
                  "seed": SEED, "score_atol": SCORE_ATOL,
                  "preprocessing_version": preprocessing.CANONICAL_VERSION,
                  "amplitude_evidence": use_amplitude,
                  "amplitude_evidence_version": metadata.get("amplitude_evidence_version"),
                  "historical_preprocessing_version": preprocessing.VERSION,
                  "clip_ids": ids, "fixed_train_clip_id": TRAIN_CASE,
                  "manifest_sha256": split.sha256,
                  "cache_sha256": {f: sha256_file(args.prepared / f"{f}.npz") for f in ("train", "val")},
                  "historical_cache_sha256": {f: sha256_file(args.historical_prepared / f"{f}.npz")
                                              for f in ("train", "val")},
                  "timef_manifest_sha256": sha256_file(args.timef_version / "manifest.json"),
                  "checkpoint_checksums_sha256": sha256_file(args.checkpoint / "checksums.json"),
                  "temporal_sha256": sha256_file(args.checkpoint / "temporal.pt"),
                  "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
                  "test_audio_or_cache_opened": False, "quality_metrics_calculated": False,
                  "threshold_diagnostic_only": args.threshold}
    return inputs, records, provenance


def score_batches(ids, series, score_batch):
    """Exécuter de vrais batches ; rattacher les résultats aux IDs hors modèle."""
    if len(ids) != len(set(ids)) or set(ids) != set(series):
        raise ValueError("Couverture des entrées batch invalide")
    result = {}
    for order_name, ordered in (("forward", ids), ("reversed", list(reversed(ids)))):
        for size in BATCH_SIZES:
            scores = {}
            for start in range(0, len(ordered), size):
                chunk = ordered[start:start + size]
                values = list(score_batch([series[cid] for cid in chunk]))
                if len(values) != len(chunk):
                    raise ValueError("Le scorer batch a omis ou ajouté un clip")
                for cid, value in zip(chunk, values, strict=True):
                    score_spread([value])
                    scores[cid] = float(value)
            result[f"batch_{size}_{order_name}"] = scores
            print(json.dumps({"stage": "batch_done", "batch_size": size,
                              "order": order_name, "clips": len(scores)}), flush=True)
    return result


def summarize_scores(scores, threshold=None):
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("Seuil de diagnostic fini requis")
    expected = set(ADAPTERS) | {f"batch_{size}_{order}" for size in BATCH_SIZES
                                for order in ("forward", "reversed")}
    adapters, all_paths, flips = {}, {}, []
    for cid, values in scores.items():
        if set(values) != expected:
            raise ValueError("Une interface ou un batch manque dans le gate")
        adapters[cid] = score_spread([values[name] for name in ADAPTERS])
        all_paths[cid] = score_spread(list(values.values()))
        if threshold is not None:
            decisions = {name: value >= threshold for name, value in values.items()}
            if len(set(decisions.values())) > 1:
                flips.append({"clip_id": cid, "scores": values, "decisions_leak": decisions})
    if not scores:
        raise ValueError("Gate sans scores")
    worst = max(all_paths, key=lambda cid: all_paths[cid]["max_abs"])
    return {"adapter_scores_within_atol": all(x["within_atol"] for x in adapters.values()),
            "batch_scores_within_atol": all(x["within_atol"] for x in all_paths.values()),
            "max_abs": all_paths[worst]["max_abs"], "worst_clip_id": worst,
            "out_of_tolerance_ids": [cid for cid, item in all_paths.items() if not item["within_atol"]],
            "threshold_diagnostic": {"threshold": threshold, "selected_here": False,
                                     "decision_flips": flips,
                                     "near_threshold_ids": [] if threshold is None else [
                                         cid for cid, values in scores.items()
                                         if min(abs(x - threshold) for x in values.values()) <= SCORE_ATOL]}}


def verify_reference(report, reference):
    """Une référence de même identité doit provenir d'un autre processus."""
    if reference.get("schema") != SCHEMA:
        raise ValueError("Schéma de référence incompatible")
    if not all(reference.get("checks", {}).get(key) is True for key in CHECKS[:-1]):
        raise ValueError("Le premier passage n'a pas validé tous les contrôles locaux")
    current, previous = report["provenance"], reference["provenance"]
    identity = ("preprocessing_version", "historical_preprocessing_version", "seed", "score_atol",
                "clip_ids", "manifest_sha256", "cache_sha256", "historical_cache_sha256",
                "checkpoint_checksums_sha256", "temporal_sha256", "timef_manifest_sha256",
                "source_sha256", "runtime", "scoring_spec", "state_sha256",
                "threshold_diagnostic_only", "amplitude_evidence", "amplitude_evidence_version",
                "amplitude_tokens")
    for key in identity:
        if current[key] != previous[key]:
            raise ValueError(f"Identité de référence différente : {key}")
    if (current["process"]["hostname"] != previous["process"]["hostname"]
            or current["process"]["pid"] == previous["process"]["pid"]):
        raise ValueError("Deux processus distincts sur le même hôte sont requis")
    ids = current["clip_ids"]
    for candidate in (report, reference):
        if [row["clip_id"] for row in candidate["records"]] != ids or set(candidate["scores"]) != set(ids):
            raise ValueError("Couverture de référence incomplète ou répétée")
    if report["records"] != reference["records"]:
        raise ValueError("Les tenseurs ou WAV de référence ont changé")
    deltas, flips = {}, []
    threshold = current["threshold_diagnostic_only"]
    for cid in ids:
        a, b = report["scores"][cid], reference["scores"][cid]
        if set(a) != set(b):
            raise ValueError("Chemins de score de référence différents")
        # Borne commune aux interfaces, tailles, ordres ET aux deux processus.
        deltas[cid] = score_spread([*a.values(), *b.values()])["max_abs"]
        if threshold is not None and len({x >= threshold for x in [*a.values(), *b.values()]}) > 1:
            flips.append(cid)
    worst = max(deltas, key=deltas.get)
    return {"fresh_process": True, "all_scores_within_atol": max(deltas.values()) <= SCORE_ATOL,
            "max_abs": deltas[worst], "worst_clip_id": worst,
            "out_of_tolerance_ids": [cid for cid, delta in deltas.items() if delta > SCORE_ATOL],
            "threshold_decision_flip_ids": flips}


def run_scores(args, inputs, provenance):
    import torch
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate as collate
    from pipe.tslm.predict import Predictor, model_input_for_model
    from pipe.tslm.preprocessing import amplitude_text
    from pipe.tslm.train import tensor_state_hash

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    runtime = {"python": platform.python_version(), "numpy": np.__version__,
               "torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device,
               "cudnn": torch.backends.cudnn.version(),
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "cudnn_benchmark": torch.backends.cudnn.benchmark,
               "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
               "cuda_device": torch.cuda.get_device_name() if args.device.startswith("cuda") else None}
    for package in ("transformers", "scikit-learn", "timenet", "opentslm"):
        try:
            runtime[package] = packages.version(package)
        except packages.PackageNotFoundError:
            runtime[package] = "not-installed-as-distribution"
    provenance["runtime"] = runtime
    predictor = Predictor(args.checkpoint, device=args.device)
    model = predictor.model
    provenance["scoring_spec"] = model.scoring_spec()
    if json.loads((args.checkpoint / "scoring_spec.json").read_text()) != provenance["scoring_spec"]:
        raise ValueError("Spécification des scores différente du bundle")
    components = {"encoder": model.encoder, "projector": model.projector, "llm": model.llm}
    provenance["state_sha256"] = {name: tensor_state_hash(part) for name, part in components.items()}
    scores, token_checks = {}, {}
    for index, (cid, (raw, waveform, series, amplitude)) in enumerate(inputs.items()):
        if amplitude is not None:
            # Les mesures et le texte sont déjà comparés entre les trois routes.
            # Empreinte du fragment isolé, pas des tokens du chat complet
            # entrelacés avec les embeddings acoustiques.
            text = amplitude_text(amplitude)
            ids = model.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                raise ValueError("Mesures d'amplitude sans tokens")
            token_checks[cid] = {"scope": "isolated_amplitude_text_not_full_chat_prompt", "n_tokens": len(ids),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "tokens_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        scores[cid] = {"series": predictor.score_series(series, amplitude_features=amplitude),
                       "waveform": predictor.score_waveform(waveform, 8000),
                       "wav_bytes": predictor.score(raw)}
        if index % 20 == 0:
            print(json.dumps({"stage": "adapters", "completed": index + 1, "total": len(inputs)}), flush=True)

    def score_batch(entries):
        examples = [model_input_for_model(series, predictor.metadata, amplitude_features=amplitude)
                    for series, amplitude in entries]
        if any(set(x) != {"pre_prompt", "post_prompt", "time_series", "time_series_text"} for x in examples):
            raise ValueError("Métadonnée ou label dans l'entrée modèle")
        return model.score_probability_leak(collate(examples, normalize=False))

    provenance["amplitude_tokens"] = token_checks
    batches = score_batches(list(inputs), {cid: values[2:] for cid, values in inputs.items()}, score_batch)
    for name, values in batches.items():
        for cid, value in values.items():
            scores[cid][name] = value
    unchanged = (provenance["state_sha256"] == {name: tensor_state_hash(part) for name, part in components.items()}
                 and provenance["scoring_spec"] == model.scoring_spec()
                 and provenance["checkpoint_checksums_sha256"] == sha256_file(args.checkpoint / "checksums.json")
                 and provenance["temporal_sha256"] == sha256_file(args.checkpoint / "temporal.pt"))
    return scores, unchanged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "prepared", "historical-prepared", "timef-version", "data-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, default=ROOT / "manifests")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--threshold", type=float, help="Seuil déjà figé, diagnostic seul ; aucun seuil choisi ici")
    args = parser.parse_args(argv)
    if args.threshold is not None and not np.isfinite(args.threshold):
        parser.error("--threshold doit être fini (borne externe permise par pick_threshold)")
    # Sortie neuve obligatoire : ne jamais remplacer les preuves V1/V2 existantes.
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "all_checks_pass": False, "checks": {name: False for name in CHECKS},
              "purpose": "development_numerical_parity_only_no_quality_metrics_no_threshold_selection"}
    stage = "prepare_inputs"
    try:
        inputs, records, provenance = prepare_inputs(args)
        report.update(provenance=provenance, records=records)
        report["checks"]["canonical_inputs_exact"] = all(row["exact"] for row in records)
        write_json(args.output / "inputs.json", {"provenance": provenance, "records": records})
        if not report["checks"]["canonical_inputs_exact"]:
            raise ValueError("Les 209 entrées canoniques ne sont pas identiques aux caches et TimeF")
        stage = "score_interfaces_and_batches"
        scores, unchanged = run_scores(args, inputs, provenance)
        summary = summarize_scores(scores, args.threshold)
        report.update(scores=scores, score_summary=summary)
        report["checks"].update(bundle_unchanged=unchanged,
                                adapter_scores_within_atol=summary["adapter_scores_within_atol"],
                                batch_scores_within_atol=summary["batch_scores_within_atol"])
        if args.reference:
            stage = "fresh_process_reference"
            report["reference_sha256"] = sha256_file(args.reference)
            comparison = verify_reference(report, json.loads(args.reference.read_text()))
            report["fresh_process"] = comparison
            report["checks"]["fresh_process_verified"] = comparison["all_scores_within_atol"]
        report["all_checks_pass"] = all(report["checks"].values())
        report["status"] = ("passed" if report["all_checks_pass"] else
                            "awaiting_fresh_process" if all(report["checks"][key] for key in CHECKS[:-1])
                            and not args.reference else "failed")
    except Exception as exc:
        report.update(status="failed", error={"stage": stage, "type": type(exc).__name__, "message": str(exc)})
    write_json(args.output / "report.json", report)
    print(json.dumps({"report": str(args.output / "report.json"), "status": report["status"],
                      "all_checks_pass": report["all_checks_pass"], "checks": report["checks"]}), flush=True)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
