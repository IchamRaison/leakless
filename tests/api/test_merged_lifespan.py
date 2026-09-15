"""Régression de fusion : studio, démo et API temporelle coexistent."""

from fastapi.testclient import TestClient

from pipe.api import main


def test_demo_alerts_and_optional_models_share_lifespan(monkeypatch):
    for name in (
        "PIPE_TEMPORAL_BUNDLE", "PIPE_TEMPORAL_SHA256", "PIPE_TEMPORAL_DB",
        "PIPE_TSLM_CHECKPOINT", "PIPE_TSLM_DECISION", "PIPE_TSLM_VALIDATION_EVIDENCE",
        "PIPE_TEMPORAL_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    shutdowns = []
    original = main.temporal.shutdown

    def shutdown(app):
        shutdowns.append(app)
        original(app)

    monkeypatch.setattr(main.temporal, "shutdown", shutdown)
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
        temporal = client.get("/temporal/health")
        assert temporal.status_code == 200
        assert temporal.json()["available"] is False
        assert temporal.json()["delivery"] == "preview_only"
        assert client.get("/incidents/current").status_code == 200
        incident = client.post("/incidents", json={
            "strongest_sensor": 1, "position": {"x": 10, "y": 20},
        })
        assert incident.status_code == 201
        assert incident.json()["created"] is True
        unavailable = client.post("/temporal/sessions", json={"source_mode": "replay"})
        assert unavailable.status_code == 503
        assert unavailable.json()["error"]["code"] == "temporal_unavailable"
    assert shutdowns == [main.app]
    assert main.app.state.samples == {}
