"""Tests de l'adaptateur UI, sans charger PyTorch ni un checkpoint réel."""

from io import BytesIO
import json

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from pipe.api.audio import decode_audio
from pipe.api.model_service import ModelService, load_tslm, predict_tslm


def wav():
    output = BytesIO()
    samples = 0.2 * np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)
    sf.write(output, samples, 8000, format="WAV", subtype="PCM_16")
    return output.getvalue()


def coherent_result():
    return {
        "schema_version": "pipe-coherent-v1",
        "input_sha256": "c" * 64,
        "model_version": "icham-v2-test",
        "preprocessing_version": "prep-v2-test",
        "probability_leak": 0.625,
        "score_type": "raw",
        "calibration": "none",
        "prediction": "leak",
        "threshold": 0.6,
        "decision_version": "decision-v2-test",
        "decision_artifact_sha256": "d" * 64,
        "dominant_band_hz": "0-1000 Hz",
        "description": "LEAK; greatest mean spectral energy: 0-1000 Hz",
        "description_source": "dsp_template_fallback",
        "fallback_used": True,
        "fallback_reasons": ["invalid_generated_format"],
        "warnings": ["Score brut non calibré."],
        "audit": {"private": "must stay server-side"},
        "latency_ms": 12.4,
    }


class FakePredictor:
    def predict(self, raw):
        assert raw.startswith(b"RIFF")
        return coherent_result()


def test_missing_configuration_does_not_import_the_model():
    service = load_tslm({})
    assert service.available is False
    assert service.device == "cpu"
    assert "PIPE_TSLM_CHECKPOINT" in service.reason


def test_loader_builds_one_predictor_from_configured_artifacts(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"model_identity": {"model_version": "v2"}}))
    evidence = tmp_path / "validation-evidence.json"
    evidence.write_text("{}")
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return FakePredictor()

    service = load_tslm(
        {
            "PIPE_TSLM_CHECKPOINT": str(checkpoint),
            "PIPE_TSLM_DECISION": str(decision),
            "PIPE_TSLM_VALIDATION_EVIDENCE": str(evidence),
            "PIPE_TSLM_DEVICE": "cpu",
        },
        factory=factory,
    )
    assert service.available is True
    assert service.version == "v2"
    assert len(calls) == 1
    assert calls[0]["checkpoint"] == checkpoint


def test_prediction_maps_v2_provenance_without_exposing_audit():
    audio = decode_audio(wav())
    response = predict_tslm(
        ModelService(True, "cpu", "v2", predictor=FakePredictor()),
        audio=audio,
        request_id="ui-request-1",
    )
    body = response.model_dump(mode="json")
    assert body["request_id"] == "ui-request-1"
    assert body["sample_id"] == audio.metadata.sample_id
    assert body["input_sha256"] == audio.metadata.input_sha256
    assert body["model_input_sha256"] == "c" * 64
    assert body["class_scores"] == {"leak": 0.625, "no_leak": 0.375}
    assert body["score_type"] == "raw"
    assert body["fallback_used"] is True
    assert "audit" not in body


def test_predict_endpoint_returns_live_v2_response(monkeypatch):
    from pipe.api import main

    service = ModelService(True, "cpu", "icham-v2-test", predictor=FakePredictor())
    monkeypatch.setattr(main, "load_tslm", lambda: service)
    with TestClient(main.app) as client:
        sample = client.post(
            "/samples", files={"file": ("recording.wav", wav(), "audio/wav")}
        ).json()
        health = client.get("/health").json()
        assert health["models"][0]["available"] is True
        response = client.post(
            "/predict",
            json={
                "sample_id": sample["sample_id"],
                "model_name": "tslm",
                "request_id": "browser-request-1",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "browser-request-1"
    assert body["input_sha256"] == sample["input_sha256"]
    assert body["execution_mode"] == "live"
    assert "audit" not in body
