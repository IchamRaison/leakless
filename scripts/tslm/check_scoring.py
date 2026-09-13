"""Sanity du score sur quatre clips train : aucune métrique ni lecture de cache val/test."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

from pipe.tslm.predict import PredictionError, Predictor, sha256_file
from pipe.tslm.prepare import MANIFEST_HASHES, load_manifest, safe_member_path
from pipe.tslm.preprocessing import VERSION, decode_wav, model_input, preprocess_audio, target_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Rapport existant : choisir un nouveau fichier")
    started = time.monotonic()
    train_rows = sorted((row for row in load_manifest(args.manifests) if row["fold"] == "train"),
                        key=lambda row: row["clip_id"])
    chosen = train_rows[:4]
    train_ids = {row["clip_id"] for row in train_rows}
    with np.load(args.prepared / "train.npz", allow_pickle=False) as cache:
        if str(cache["preprocessing_version"]) != VERSION:
            raise ValueError("Cache de mauvaise version")
        ids = cache["ids"].tolist()
        if len(ids) != len(set(ids)) or set(ids) != train_ids or len(chosen) != 4:
            raise ValueError("Le cache doit contenir exactement les IDs du train gelé")
        cached = {row["clip_id"]: cache["series"][ids.index(row["clip_id"])].copy() for row in chosen}
    predictor = Predictor(args.checkpoint)
    spec = predictor.model.scoring_spec()
    samples = []
    for row in chosen:
        sample_started = time.monotonic()
        raw = safe_member_path(row["path"], args.data_root).read_bytes()
        waveform = decode_wav(raw)
        series = preprocess_audio(waveform, 8000)
        np.testing.assert_allclose(cached[row["clip_id"]], series, rtol=1e-6, atol=1e-6)
        example = model_input(series)
        if set(example) != {"pre_prompt", "post_prompt", "time_series", "time_series_text"}:
            raise ValueError("Métadonnées dans l'entrée du modèle")
        for label, candidate_ids in zip(("leak", "no_leak"), spec["class_token_ids"]):
            # Les deux réponses artificielles testent la frontière de tokenisation,
            # sans consulter la vérité terrain du clip.
            full_ids = predictor.model.tokenizer.encode(
                target_text(label, series) + predictor.model.get_eos_token(), add_special_tokens=False)
            if full_ids[:len(candidate_ids)] != candidate_ids:
                raise ValueError(f"Frontière de tokens de classe incohérente : {label}")
        scores = {"wav": predictor.score(raw), "waveform": predictor.score_waveform(waveform),
                  "series": predictor.score_series(series), "repeat": predictor.score_series(series)}
        batch = extend_time_series_to_match_patch_size_and_aggregate([example], normalize=False)
        logprobs = predictor.model.score_class_logprobs(batch)[0].cpu().tolist()
        if not all(math.isfinite(value) for value in logprobs):
            raise ValueError("Log-probabilités non finies")
        maximum = max(logprobs)
        numerators = [math.exp(value - maximum) for value in logprobs]
        scores["manual_softmax"] = numerators[0] / sum(numerators)
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in scores.values()):
            raise ValueError("Probabilité non finie ou hors [0,1]")
        np.testing.assert_allclose(list(scores.values()), scores["wav"], rtol=0, atol=1e-6)
        samples.append({"clip_id": row["clip_id"], "fold": "train",
                        "input_sha256": hashlib.sha256(raw).hexdigest(),
                        "probabilities": scores, "class_logprobs": logprobs,
                        "max_timef_cache_feature_difference": float(np.max(np.abs(cached[row["clip_id"]] - series))),
                        "max_probability_difference": max(abs(value - scores["wav"]) for value in scores.values()),
                        "all_routes_elapsed_seconds": time.monotonic() - sample_started})
        print(json.dumps({"train_clip_checked": row["clip_id"]}), flush=True)

    rejected = []
    for name, invoke, expected in (
        ("invalid_wav", lambda: predictor.score(b"not a WAV"), "unsupported_audio"),
        ("constant_signal", lambda: predictor.score_waveform(np.ones(8000)), "silent_audio"),
    ):
        try:
            invoke()
        except PredictionError as exc:
            if exc.code != expected:
                raise
            rejected.append(name)
        else:
            raise RuntimeError(f"Entrée invalide acceptée : {name}")
    try:
        predictor.model.score_probability_leak([{**example, "clip_id": "forbidden"}])
    except ValueError:
        rejected.append("metadata_in_model_input")
    else:
        raise RuntimeError("Métadonnées acceptées par le score")

    probabilities = [sample["probabilities"]["wav"] for sample in samples]
    report = {"purpose": "train_only_scoring_sanity_not_model_selection_or_final_evaluation",
              "folds_read_for_signals": ["train"], "cache_files_read": ["train.npz"],
              "model_version": predictor.metadata["model_version"],
              "checkpoint_code_revision": predictor.metadata["code_revision"],
              "base_model": predictor.metadata["base_model"],
              "base_revision": predictor.metadata["base_revision"],
              "preprocessing_version": VERSION, "scoring_spec": spec,
              "checkpoint_checksums_sha256": sha256_file(args.checkpoint / "checksums.json"),
              "train_cache_sha256": sha256_file(args.prepared / "train.npz"),
              "manifest_sha256": MANIFEST_HASHES["split_v2.csv"],
              "script_sha256": sha256_file(Path(__file__)),
              "samples": samples, "invalid_inputs_rejected": rejected,
              "tokenization_boundary_verified": True, "probability_tolerance": 1e-6,
              "max_feature_difference": max(sample["max_timef_cache_feature_difference"] for sample in samples),
              "train_distribution_diagnostic_only": {"n_unique": len(set(probabilities)),
                  "min": min(probabilities), "max": max(probabilities),
                  "n_exact_zero_or_one": sum(value in (0., 1.) for value in probabilities)},
              "elapsed_seconds": time.monotonic() - started}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
