"""Contrat HTTP et fautes de flux ; faux détecteur réservé aux tests logiciels."""
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from threading import Lock
from unittest.mock import patch
import wave

from fastapi.testclient import TestClient
import numpy as np

from pipe.api.main import app
from pipe.temporal_model import acoustic_features, decode_pcm
from pipe.temporal_tracking import Tracker


def test_endpoint_contract_quality_dedup_and_auth(tmp_path, monkeypatch):
    for name in ("BUNDLE", "SHA256", "DB", "TOKEN"):
        monkeypatch.delenv("PIPE_TEMPORAL_" + name, raising=False)
    with TestClient(app) as client:
        assert client.get("/temporal/health").json()["available"] is False
        assert client.post("/temporal/sessions", json={}).status_code == 503
        class Detector:
            version = "test-only"
            def score(self, signal):
                return acoustic_features(signal), .91
            def sequence_inputs(self, signal):
                raise ValueError("Invalid sequence for fixture")
        policy = json.loads((Path(__file__).resolve().parents[2] / "configs/temporal/c1_lstm.json").read_text())["policy"]
        app.state.temporal = {"detector": Detector(), "tracker": Tracker(tmp_path / "db", policy, "test"),
                              "lock": Lock(), "sequence": None}
        with patch("pipe.temporal_tracking.time.time", return_value=100.):
            session = client.post("/temporal/sessions", json={}).json()["session_id"]
        assert client.post("/temporal/sessions", json={"source_mode": "live"}).status_code == 422
        raw = BytesIO()
        with wave.open(raw, "wb") as wav:
            wav.setparams((1, 2, 8000, 8000, "NONE", "none"))
            wav.writeframes((np.sin(np.arange(8000)) * 10000).astype("<i2").tobytes())
        def submit(seq, data=raw.getvalue(), stamp=None):
            with patch("pipe.api.temporal.time.time", return_value=101. + seq):
                return client.post(f"/temporal/sessions/{session}/windows",
                    data={"sequence": seq, "source_end_at": stamp or datetime.fromtimestamp(101 + seq, timezone.utc).isoformat()},
                    files={"file": ("irrelevant_label.wav", data, "audio/wav")})
        for seq in range(3):
            result = submit(seq)
            assert result.status_code == 200, result.text
            assert result.json()["probability_leak"] == .91
        assert result.json()["preview"]["sent"] is False
        assert submit(2).json()["duplicate"] is True
        assert submit(2, b"different").status_code == 422
        assert submit(3, stamp="2026-01-01T00:00:00").status_code == 422
        invalid = submit(3, b"invalid").json()
        assert invalid["probability_leak"] is None
        assert invalid["state"]["health"] == "degraded"
        assert invalid["state"]["active"]["coverage_interrupted"] is True
        assert submit(4, b"x" * 65537).status_code == 413
        assert client.post("/temporal/sequence", files={"file": ("x.wav", b"invalid")}).status_code == 422
        app.state.temporal["lock"].acquire()
        assert client.post("/temporal/sequence", files={"file": ("x.wav", b"invalid")}).status_code == 409
        app.state.temporal["lock"].release()
        monkeypatch.setenv("PIPE_TEMPORAL_TOKEN", "test-only-not-a-secret")
        assert client.get("/temporal/health").status_code == 401
        assert client.get("/temporal/health", headers={"Authorization": "Bearer test-only-not-a-secret"}).status_code == 200
