"""Processus neuf : vérifier score, texte et trois chemins numériques sur développement."""
import argparse
import json
import math
from pathlib import Path

from export_run import sha256_file

SCORE_ATOL = 1e-6


def close_score(actual, expected):
    return (math.isfinite(actual) and math.isfinite(expected) and 0 <= actual <= 1
            and 0 <= expected <= 1 and abs(actual - expected) <= SCORE_ATOL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Rapport déjà présent : choisir un nouveau chemin")
    from pipe.tslm.predict import PredictionError, Predictor
    from pipe.tslm.preprocessing import decode_wav, preprocess_audio

    predictor = Predictor(args.checkpoint, device=args.device)
    expected = json.loads((args.checkpoint / "reload-expected.json").read_text())
    spec = json.loads((args.checkpoint / "scoring_spec.json").read_text())
    if predictor.model.scoring_spec() != spec:
        raise RuntimeError("Scoring différent du checkpoint figé")
    raw = (args.checkpoint / "reload-example.wav").read_bytes()
    score = float(predictor.score(raw))
    if not close_score(score, float(expected["probability_leak"])):
        raise RuntimeError("Score différent après rechargement frais")
    waveform = decode_wav(raw)
    path_scores = {"wav": score, "waveform": float(predictor.score_waveform(waveform, 8000)),
                   "series": float(predictor.score_series(preprocess_audio(waveform, 8000)))}
    if any(not close_score(value, score) for value in path_scores.values()):
        raise RuntimeError("Entrées WAV/waveform/series incohérentes sur développement")
    got = predictor.predict(raw).model_dump(mode="json")
    for key in ("prediction", "description", "abstained", "observations", "input_sha256", "model_version"):
        if got[key] != expected["prediction"][key]:
            raise RuntimeError(f"Génération différente après rechargement : {key}")
    try:
        predictor.score(b"not a WAV")
    except PredictionError as exc:
        if exc.code != "unsupported_audio":
            raise RuntimeError("Erreur inattendue pour audio invalide") from exc
    else:
        raise RuntimeError("Audio invalide accepté")
    report = {"fresh_process_reload": True, "score_match": True,
              "deterministic_output_match": True, "input_paths_match": True,
              "invalid_audio_rejected": True, "score_atol": SCORE_ATOL,
              "probability_leak": score, "input_path_scores": path_scores,
              "prediction": got, "scoring_spec": spec,
              "checkpoint_checksums_sha256": sha256_file(args.checkpoint / "checksums.json")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
