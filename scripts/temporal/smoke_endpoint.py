"""Recette HTTP finie avec poids réels, train seul et chronologie artificielle explicite."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import time
import wave

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pipe.temporal_model import C1Detector, decode_pcm, digest
from pipe.sequence_model import load_sequence, sequence_probability
from run_campaign import split_loader, load_expected_audio_md5, verify_audio_bytes
from replay_endpoint import Receiver, replay


def write(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((args.run / "bundle-receipt.json").read_text())
    detector = C1Detector(args.run / "bundle", receipt["bundle_sha256"])
    split = split_loader.load_split(ROOT / "manifests")
    md5 = load_expected_audio_md5(ROOT / "manifests")
    candidates = []
    for clip in sorted(split.fold("train"), key=lambda r: r.clip_id):
        source = (args.data_root / split.path_of(clip.clip_id)).resolve()
        if not source.is_relative_to(args.data_root.resolve()):
            raise ValueError("Chemin hors racine")
        raw = source.read_bytes()
        verify_audio_bytes(raw, clip.clip_id, md5)
        features, p = detector.score(decode_pcm(raw))
        candidates.append((p, clip.clip_id, source, features))
    low, high = min(candidates, key=lambda r:r[0]), max(candidates, key=lambda r:r[0])
    policy = detector.metadata["config"]["policy"]
    if low[0] > policy["close_threshold"] or high[0] < policy["open_threshold"]:
        raise ValueError("Le train ne fournit pas les scores nécessaires à ce scénario ; ne pas changer les seuils")
    order = [low] * 3 + [high] * 4 + [low] * 5
    entries = [{"path": str(row[2]), "audio_sha256": digest(row[2]), "provenance": "train",
                "source_offset_samples": i * 8000} for i, row in enumerate(order)]
    write(args.output / "scenario.json", {"schema_version":1,"device_id":"smoke-c1",
        "source_mode":"replay", "entries":entries, "incidents":[], "consumer_seconds":0,
        "selection":"train score extrema repeated; artificial chronology; software check only"})
    write(args.output / "reference.json", {"clips":[row[1] for row in order],
        "scores":[row[0] for row in order], "selection":"deliberate train extrema, not quality evaluation"})
    result = replay(args.output / "scenario.json", args.output / "replay", receiver=Receiver(args.endpoint))
    rows = [json.loads(line) for line in (args.output / "replay/events.jsonl").read_text().splitlines()]
    received = [r for r in rows if r["event"] == "received"]
    assert len(received) == 12 and result["status"] == "completed"
    scores = np.asarray([r["prediction"]["probability_leak"] for r in received])
    delta = float(abs(scores - [row[0] for row in order]).max())
    assert delta == 0
    for row, reference in zip(received, order):
        np.testing.assert_array_equal(row["prediction"]["features"], reference[3])
    previews = [r["prediction"]["preview"] for r in received if r["prediction"]["preview"]]
    assert [p["kind"] for p in previews] == ["opened", "ended"]
    assert all(not p["sent"] and p["delivery"] == "preview_only" for p in previews)
    latency = [r["prediction"]["latency_ms"] for r in received]
    with httpx.Client(base_url=args.endpoint, timeout=30, trust_env=False) as client:
        state = client.get(f"/temporal/sessions/{result['session_id']}").raise_for_status().json()
        assert state["health"] == "ended" and len(state["events"]) == 1 and len(state["previews"]) == 2
        registration = json.loads((args.run / "registration.json").read_text())
        row = next(r for r in registration["records"] if r["partition"] == "train" and r["full_scale_samples"] == 0)
        signal = np.load(Path(registration["external"]) / row["array_path"], allow_pickle=False)
        raw = BytesIO()
        with wave.open(raw, "wb") as wav:
            wav.setparams((1, 4, 8000, 240000, "NONE", "none"))
            wav.writeframes(np.rint(signal * 2**31).astype("<i4").tobytes())
        np.testing.assert_array_equal(decode_pcm(raw.getvalue(), seconds=30), signal)
        model, mean, scale = load_sequence(detector)
        expected = sequence_probability(model, detector.sequence_inputs(signal), mean, scale)
        started = time.perf_counter()
        response = client.post("/temporal/sequence", files={"file":("sequence.wav",raw.getvalue(),"audio/wav")})
        response.raise_for_status()
        actual = response.json()
        sequence_delta = abs(actual["probability_leak"] - expected)
        assert sequence_delta <= 1e-5 and actual["experimental"]
        sequence_seconds = time.perf_counter() - started
    report = {"status":"passed", "real_checkpoint":True, "n_windows":12, "artificial_chronology":True,
        "scientific_quality_evaluation":False, "train_only":True, "c1_http_max_diff":delta,
        "window_server_latency_p95_ms":float(np.quantile(latency,.95)),
        "window_server_latency_max_ms":max(latency), "n_events":1,"n_previews":2,"external_messages_sent":0,
        "sequence_http_max_diff":sequence_delta,"sequence_http_seconds_including_warmup":sequence_seconds,
        "sequence_train_recording":row["recording_id"],"bundle_sha256":receipt["bundle_sha256"],
        "session_id":result["session_id"]}
    write(args.output / "report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--endpoint",default="http://127.0.0.1:8019")
    main(parser.parse_args())
