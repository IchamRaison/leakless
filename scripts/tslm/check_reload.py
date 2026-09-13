"""Exécuter dans un processus neuf après train ; inférence sans réseau ni labels."""
import argparse
import json
from pathlib import Path

from pipe.tslm.predict import PredictionError, Predictor, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictor = Predictor(args.checkpoint)
    raw = (args.checkpoint / "reload-example.wav").read_bytes()
    got = predictor.predict(raw).model_dump(mode="json")
    expected = json.loads((args.checkpoint / "training-report.json").read_text())["after_validation_prediction"]
    for key in ("prediction", "description", "abstained", "observations", "input_sha256", "model_version"):
        if got[key] != expected[key]:
            raise RuntimeError(f"Reload non identique : {key}")
    try:
        predictor.predict(b"not a WAV")
    except PredictionError as exc:
        assert exc.code == "unsupported_audio"
    else:
        raise RuntimeError("WAV invalide accepté")
    report = {"fresh_process_reload": True, "deterministic_output_match": True,
              "invalid_audio_rejected": True, "prediction": got,
              "checkpoint_checksums_sha256": sha256_file(args.checkpoint / "checksums.json")}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
