"""Recette réelle du prototype fold0 : API WAV, reload et texte sur train fixé."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/tslm")]
import run_v2_campaign as campaign
from pipe.tslm.campaign import write_json


def check(root, checkpoint):
    from pipe.tslm.predict import Predictor, PredictionError
    expected = campaign.read_json(checkpoint / "training-result.json")
    split = campaign.split_loader.load_split(ROOT / "manifests")
    train_ids = {row["clip_id"] for row in campaign.diagnostic.train_rows(split)}
    ids = expected["reload_train_ids"]
    assert len(ids) == 4 and set(ids) <= train_ids
    model = Predictor(checkpoint / "bundle")
    assert model.metadata["fold_id"] == 0
    md5 = campaign.diagnostic.load_expected_audio_md5(ROOT / "manifests")
    result = {"bundle_sha256": campaign.sha256_file(checkpoint / "bundle/checksums.json"),
              "prototype_fold": 0, "optimizer_steps": 0,
              "official_validation_test_external_read": False, "clips": []}
    for index, cid in enumerate(ids):
        data_root = (root / "data/extracted").resolve()
        path = (data_root / split.path_of(cid)).resolve()
        assert path.is_relative_to(data_root)
        raw = path.read_bytes()
        campaign.diagnostic.verify_audio_bytes(raw, cid, md5)
        start = time.monotonic()
        score = model.score(raw)
        delta = abs(score - expected["reload_probabilities"][cid])
        assert delta <= 1e-6
        item = {"clip_id": cid, "score": score, "absolute_difference": delta,
                "score_seconds": time.monotonic() - start}
        if index == 0:
            item["prediction"] = model.predict(raw).model_dump(mode="json")
        result["clips"].append(item)
    try:
        model.score(b"invalid wav")
    except PredictionError as error:
        assert error.code == "unsupported_audio"
        result["invalid_wav_rejected"] = True
    else:
        raise AssertionError("WAV invalide accepté")
    result["passed"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.root, args.checkpoint)
    write_json(args.output, result)
    print(result, flush=True)
