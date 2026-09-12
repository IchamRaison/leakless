"""WAV synthétiques réservés aux tests d'interface, jamais exemples scientifiques."""

from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from pydantic import ValidationError

from pipe.api.main import app
from pipe.api.audio import MAX_BYTES
from pipe.contracts import Prediction


def wav(samples=None, rate=16000, subtype="PCM_16"):
    if samples is None:
        samples = .25 * np.sin(2 * np.pi * 1000 * np.arange(rate) / rate)
    out = BytesIO()
    sf.write(out, samples, rate, format="WAV", subtype=subtype)
    return out.getvalue()


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def upload(client, raw=None, name="hidden_leak_name.wav"):
    return client.post("/samples", files={"file": (name, wav() if raw is None else raw, "audio/wav")})


def test_round_trip_and_visualization_identity(client):
    raw = wav()
    response = upload(client, raw)
    assert response.status_code == 201
    sample = response.json()
    assert "leak" not in response.text
    assert sample["duration_seconds"] == 1
    assert sample["execution_mode"] == "development_fixture"
    assert sample["label_available"] is False
    audio = client.get(f"/samples/{sample['sample_id']}/audio")
    assert audio.content == raw
    assert audio.headers["x-input-sha256"] == sample["input_sha256"]
    visual = client.get(f"/samples/{sample['sample_id']}/visualization").json()
    assert visual["input_sha256"] == sample["input_sha256"]
    assert len(visual["waveform"]["min"]) <= 800
    matrix = np.array(visual["spectrogram"]["power_db"])
    assert matrix.shape[0] <= 256 and matrix.shape[1] <= 400
    assert np.isfinite(matrix).all()
    # Le pic spectral du signal de test se situe bien à 1 kHz.
    peak_hz = visual["spectrogram"]["frequencies_hz"][matrix.mean(axis=1).argmax()]
    assert abs(peak_hz - 1000) < 40
    assert client.get("/samples").json() == [sample]


@pytest.mark.parametrize("raw,name,status", [
    (b"not a wave", "audio.wav", 422),
    (b"", "empty.wav", 422),
    (b"anything", "audio.mp3", 415),
    (b"x" * (MAX_BYTES + 1), "large.wav", 413),
])
def test_invalid_uploads(client, raw, name, status):
    response = upload(client, raw, name)
    assert response.status_code == status
    assert response.json()["error"]["message"]
    assert client.get("/samples").json() == []


@pytest.mark.parametrize("samples,rate,subtype", [
    (np.zeros(16000 * 31), 16000, "PCM_16"),
    (np.zeros((160, 3)), 16000, "PCM_16"),
    (np.zeros(100), 4000, "PCM_16"),
    (np.array([np.nan, 1.0]), 16000, "FLOAT"),
    (np.array([np.inf, 1.0]), 16000, "FLOAT"),
    (np.zeros(0), 16000, "PCM_16"),
])
def test_invalid_signal_properties(client, samples, rate, subtype):
    response = upload(client, wav(samples, rate, subtype))
    assert response.status_code == 422


def test_silence_short_clip_and_stereo(client):
    for signal in [np.zeros(1600), np.zeros(1), np.zeros((1600, 2))]:
        sample = upload(client, wav(signal)).json()
        assert sample["warnings"]
        visual = client.get(f"/samples/{sample['sample_id']}/visualization")
        assert visual.status_code == 200
        assert np.isfinite(visual.json()["spectrogram"]["power_db"]).all()


def test_no_fake_model_label_or_evaluation(client):
    health = client.get("/health").json()
    assert all(not model["available"] for model in health["models"])
    sample = upload(client).json()
    for model in ("tslm", "baseline"):
        response = client.post("/predict", json={"sample_id": sample["sample_id"], "model_name": model, "request_id": "test"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"
        assert "prediction" not in response.json()
    assert client.get(f"/samples/{sample['sample_id']}/label").status_code == 404
    assert client.get("/evaluation").status_code == 404


def test_unknown_ids_and_arbitrary_paths(client):
    assert client.get("/samples/not-found/audio").status_code == 404
    assert client.post("/predict", json={"sample_id": "../private", "model_name": "tslm", "request_id": "test"}).status_code == 422
    assert client.post("/predict", json={"sample_id": "a" * 24, "model_name": "tslm", "request_id": "test", "ground_truth": "leak"}).status_code == 422


def test_remove_deduplicate_and_restart(client):
    sample = upload(client).json()
    assert upload(client, name="other.wav").json()["sample_id"] == sample["sample_id"]
    assert len(client.get("/samples").json()) == 1
    assert client.delete(f"/samples/{sample['sample_id']}").status_code == 204
    assert client.get(f"/samples/{sample['sample_id']}/audio").status_code == 404
    assert client.get("/samples").json() == []


def test_lifespan_discards_uploads():
    with TestClient(app) as client:
        upload(client)
    with TestClient(app) as client:
        assert client.get("/samples").json() == []


def test_bounded_library(client):
    for index in range(16):
        assert upload(client, wav(np.full(80, index / 32))).status_code == 201
    assert upload(client, wav(np.full(80, .9))).status_code == 409


def test_streamed_upload_limit(client):
    response = client.post("/samples", content=(b"x" * 1024 * 1024 for _ in range(9)))
    assert response.status_code == 413


def test_prediction_rejects_ground_truth_and_nan():
    example = dict(schema_version="0.1", sample_id="a" * 24, input_sha256="b" * 64,
                   model_name="tslm", model_version="test-only", preprocessing_version="test-only",
                   prediction=None, score_type="none", abstained=True, latency_ms=1,
                   execution_mode="development_fixture")
    assert Prediction(**example).prediction is None
    with pytest.raises(ValidationError):
        Prediction(**example, ground_truth="leak")
    with pytest.raises(ValidationError):
        Prediction(**{**example, "latency_ms": float("nan")})
