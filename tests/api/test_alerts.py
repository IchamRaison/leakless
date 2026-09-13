"""Alertes côté serveur : persistance, dédoublonnage, Telegram simulé. Aucun envoi réel."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

import pipe.api.main as main
from pipe.api.alerts import (
    ALERT_FAILED,
    ALERT_SENT,
    ANOMALY_PENDING,
    RESOLVED,
    IncidentStore,
    TelegramTransport,
    TransportError,
    alert_text,
)

SECRET = "123456:SECRET-TOKEN-never-shown"


class Clock:
    def __init__(self, now=1_789_300_000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeTelegram:
    name = "telegram"
    configured = True

    def __init__(self, fail=False):
        self.fail = fail
        self.messages = []

    def send(self, text):
        self.messages.append(text)
        if self.fail:
            raise TransportError("Telegram rejected the message: Bad Request: chat not found.")
        return str(9000 + len(self.messages))


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def telegram():
    return FakeTelegram()


@pytest.fixture
def store(clock, telegram, tmp_path):
    return IncidentStore(telegram, persistence_seconds=30, clock=clock, state_file=tmp_path / "incidents.json")


def inject(store, sensor=3):
    incident, created = store.inject(sensor, {"x": 1000, "y": 240})
    assert created
    return incident


def run(store, clock, seconds, step=0.5):
    for _ in range(int(seconds / step)):
        clock.advance(step)
        store.tick()


def test_anomaly_shorter_than_30_s_sends_nothing(store, clock, telegram):
    incident = inject(store)
    run(store, clock, 29.5)
    assert store.active.status == ANOMALY_PENDING
    assert telegram.messages == []
    store.resolve(incident.incident_id)
    run(store, clock, 60)
    assert telegram.messages == []
    assert store.history[0].status == RESOLVED
    assert store.history[0].delivery_status == "not_dispatched"
    assert store.history[0].alert_sent_at is None


def test_persistent_anomaly_sends_exactly_one_message(store, clock, telegram):
    incident = inject(store)
    run(store, clock, 30)
    assert len(telegram.messages) == 1
    run(store, clock, 300)
    assert len(telegram.messages) == 1
    active = store.active
    assert active.incident_id == incident.incident_id
    assert active.status == ALERT_SENT
    assert active.transport == "telegram"
    assert active.telegram_message_id == "9001"
    assert active.confirmed_at == pytest.approx(incident.started_at + 30)
    assert active.alert_sent_at is not None
    text = telegram.messages[0]
    assert text.startswith("🚨 LeakLess Alert\n\nPersistent abnormal signal detected for 30 seconds.")
    assert "Strongest response: Sensor 3" in text and "Zone: N3" in text
    assert "Action: inspection recommended." in text and "Source: injected test incident" in text
    for banned in ("probability", "confidence", "certified", "localized", "localised"):
        assert banned not in text.lower()


def test_repeated_injection_and_polling_never_duplicate(store, clock, telegram):
    first = inject(store)
    again, created = store.inject(1, {"x": 1, "y": 1})
    assert not created and again.incident_id == first.incident_id
    run(store, clock, 31)
    for _ in range(50):
        store.snapshot()
        store.tick()
    assert len(telegram.messages) == 1


def test_restart_reload_does_not_resend(clock, telegram, tmp_path):
    state = tmp_path / "incidents.json"
    store = IncidentStore(telegram, clock=clock, state_file=state)
    inject(store)
    run(store, clock, 31)
    assert len(telegram.messages) == 1
    reloaded = IncidentStore(telegram, clock=clock, state_file=state)
    run(reloaded, clock, 120)
    assert len(telegram.messages) == 1
    assert reloaded.active.status == ALERT_SENT


def test_restart_during_dispatch_is_failed_not_resent(clock, telegram, tmp_path):
    state = tmp_path / "incidents.json"
    store = IncidentStore(telegram, clock=clock, state_file=state)
    incident = inject(store)
    raw = json.loads(state.read_text())
    raw["active"]["status"] = "ALERT_DISPATCHING"
    state.write_text(json.dumps(raw))
    reloaded = IncidentStore(telegram, clock=clock, state_file=state)
    run(reloaded, clock, 120)
    assert telegram.messages == []
    assert reloaded.active.incident_id == incident.incident_id
    assert reloaded.active.status == ALERT_FAILED
    assert reloaded.active.delivery_status == "unknown"
    # Delivery may have happened: retry is refused, never a possible duplicate.
    assert reloaded.retry(incident.incident_id) is None


def test_telegram_failure_is_failed_never_sent(clock, tmp_path, caplog):
    failing = FakeTelegram(fail=True)
    store = IncidentStore(failing, clock=clock, state_file=tmp_path / "s.json")
    inject(store)
    run(store, clock, 60)
    assert len(failing.messages) == 1
    assert store.active.status == ALERT_FAILED
    assert store.active.delivery_status == "failed"
    assert store.active.alert_sent_at is None and store.active.telegram_message_id is None
    assert "chat not found" in store.active.failure_reason
    assert "alert NOT delivered" in caplog.text


def test_resolve_after_send_keeps_history(store, clock, telegram):
    incident = inject(store)
    run(store, clock, 31)
    clock.advance(200)
    store.resolve(incident.incident_id)
    assert store.active is None
    record = store.history[0]
    assert record.incident_id == incident.incident_id
    assert record.status == RESOLVED and record.delivery_status == "sent"
    assert record.telegram_message_id == "9001"
    assert [event["kind"] for event in record.events] == [
        "anomaly_started", "persistence_confirmed", "alert_sent", "resolved"]
    assert record.started_at < record.confirmed_at <= record.alert_sent_at < record.resolved_at
    snapshot = store.snapshot()
    assert snapshot["active"] is None and snapshot["history"][0]["incident_id"] == incident.incident_id


def test_unconfigured_transport_fails_clearly():
    transport = TelegramTransport(None, None)
    assert not transport.configured
    with pytest.raises(TransportError, match="TELEGRAM_BOT_TOKEN"):
        transport.send("x")


def test_telegram_transport_parses_success_and_hides_token():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 4242}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert TelegramTransport(SECRET, "777", client=client).send("hello") == "4242"
    assert seen["body"] == {"chat_id": "777", "text": "hello"}

    def unreachable(request):
        raise httpx.ConnectError("boom", request=request)

    down = httpx.Client(transport=httpx.MockTransport(unreachable))
    with pytest.raises(TransportError) as error:
        TelegramTransport(SECRET, "777", client=down).send("hello")
    assert SECRET not in str(error.value)

    rejected = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(401, json={"ok": False, "description": "Unauthorized"})))
    with pytest.raises(TransportError, match="Unauthorized"):
        TelegramTransport(SECRET, "777", client=rejected).send("hello")


def test_alert_text_format(store, clock):
    incident = inject(store)
    text = alert_text(incident, 30, incident.started_at + 30)
    assert text.splitlines()[0] == "🚨 LeakLess Alert"
    assert "Detected: " in text and text.endswith("Source: injected test incident")


def test_api_endpoints_are_read_safe_and_never_expose_secrets(monkeypatch, tmp_path, clock, telegram):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", SECRET)
    monkeypatch.setattr(main, "alert_store_factory",
                        lambda: IncidentStore(telegram, clock=clock, state_file=tmp_path / "api.json"))
    monkeypatch.setattr(main, "ALERT_TICK_SECONDS", 3600)
    with TestClient(main.app) as client:
        assert client.get("/incidents/current").json()["active"] is None
        created = client.post("/incidents", json={"strongest_sensor": 3, "position": {"x": 1000, "y": 240}})
        assert created.status_code == 201 and created.json()["created"]
        incident_id = created.json()["incident_id"]
        duplicate = client.post("/incidents", json={"strongest_sensor": 1, "position": {"x": 1, "y": 1}})
        assert duplicate.json()["incident_id"] == incident_id and not duplicate.json()["created"]
        clock.advance(31)
        client.app.state.alerts.tick()
        for _ in range(20):  # a reloading or polling browser has no side effect
            body = client.get("/incidents/current")
        assert body.json()["active"]["status"] == ALERT_SENT
        assert SECRET not in body.text and "TELEGRAM_BOT_TOKEN" not in body.text
        resolved = client.post(f"/incidents/{incident_id}/resolve").json()
        assert resolved["active"] is None and resolved["history"][0]["status"] == RESOLVED
        assert client.post("/incidents/missing/resolve").status_code == 404
        assert client.post("/incidents", json={"strongest_sensor": 9, "position": {"x": 0, "y": 0}}).status_code == 422
    assert len(telegram.messages) == 1


# -- Revue de code : régressions --------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    '{"strongest_sensor": 3, "position": {"x": "inf", "y": 240}}',
    '{"strongest_sensor": 3, "position": {"x": NaN, "y": 240}}',
    '{"strongest_sensor": 3, "position": {"x": 1000, "y": -Infinity}}',
    '{"strongest_sensor": 3, "position": {"x": 1e309, "y": 240}}',
])
def test_non_finite_positions_are_rejected_and_never_poison_state(monkeypatch, tmp_path, clock, telegram, bad):
    store = IncidentStore(telegram, clock=clock, state_file=tmp_path / "s.json")
    monkeypatch.setattr(main, "alert_store_factory", lambda: store)
    with TestClient(main.app) as client:
        response = client.post("/incidents", content=bad, headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        current = client.get("/incidents/current")
        assert current.status_code == 200 and current.json()["active"] is None
    with pytest.raises(ValueError):
        store.inject(3, {"x": float("nan"), "y": 1})
    assert store.active is None


def test_poisoned_state_file_is_ignored_on_restart(tmp_path, clock, telegram):
    state = tmp_path / "s.json"
    state.write_text('{"active": {"incident_id": "inc-x", "source": "injected_test", "strongest_sensor": 3,'
                     ' "zone": "N3", "position": {"x": Infinity, "y": 1}, "started_at": 1}, "history": []}')
    store = IncidentStore(telegram, clock=clock, state_file=state)
    assert store.active is None
    assert json.dumps(store.snapshot(), allow_nan=False)


def test_storage_failure_fails_explicitly_without_risking_a_duplicate(clock, telegram, tmp_path, monkeypatch, caplog):
    store = IncidentStore(telegram, clock=clock, state_file=tmp_path / "s.json")
    incident = inject(store)
    real_write = type(tmp_path).write_text

    def disk_full(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(type(tmp_path), "write_text", disk_full)
    run(store, clock, 31)
    # Without a durable dispatch record a restart could resend: nothing is sent, the state says why.
    assert telegram.messages == []
    assert store.active.status == ALERT_FAILED and store.active.delivery_status == "failed"
    assert "could not be saved" in store.active.failure_reason
    assert store.snapshot()["storage_ok"] is False
    assert "NOT saved" in caplog.text
    # Storage recovers: one explicit retry sends exactly once.
    monkeypatch.setattr(type(tmp_path), "write_text", real_write)
    assert store.retry(incident.incident_id) is not None
    run(store, clock, 2)
    assert len(telegram.messages) == 1 and store.active.status == ALERT_SENT


def test_unexpected_preparation_error_fails_explicitly(clock, telegram, tmp_path, monkeypatch):
    store = IncidentStore(telegram, clock=clock, state_file=tmp_path / "s.json")
    inject(store)
    import pipe.api.alerts as alerts
    monkeypatch.setattr(alerts, "alert_text", lambda *a, **k: 1 / 0)
    run(store, clock, 31)
    assert telegram.messages == []
    assert store.active.status == ALERT_FAILED
    assert "ZeroDivisionError" in store.active.failure_reason


def test_alert_loop_survives_a_failing_tick(monkeypatch):
    calls = []

    class Flaky:
        def tick(self):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")

    monkeypatch.setattr(main, "ALERT_TICK_SECONDS", 0)

    async def scenario():
        import asyncio
        task = asyncio.create_task(main.alert_loop(Flaky()))
        while len(calls) < 3:
            await asyncio.sleep(0.01)
        task.cancel()

    import asyncio
    asyncio.run(scenario())
    assert len(calls) >= 3


def test_failed_incident_can_be_retried_once_or_resolved_and_never_blocks_forever(clock, tmp_path):
    flaky = FakeTelegram(fail=True)
    store = IncidentStore(flaky, clock=clock, state_file=tmp_path / "s.json")
    incident = inject(store)
    run(store, clock, 31)
    assert store.active.status == ALERT_FAILED and len(flaky.messages) == 1
    # No automatic resend while failed.
    run(store, clock, 120)
    assert len(flaky.messages) == 1
    # Explicit retry: exactly one new attempt.
    flaky.fail = False
    assert store.retry(incident.incident_id) is not None
    run(store, clock, 5)
    assert len(flaky.messages) == 2 and store.active.status == ALERT_SENT
    assert store.active.attempts == 2
    # Retry is refused once sent: no duplicate success.
    assert store.retry(incident.incident_id) is None
    run(store, clock, 60)
    assert len(flaky.messages) == 2
    # Resolving frees the slot for a new incident.
    store.resolve(incident.incident_id)
    assert inject(store, sensor=1).incident_id != incident.incident_id


def test_failed_incident_resolves_and_api_exposes_retry(monkeypatch, tmp_path, clock):
    failing = FakeTelegram(fail=True)
    store = IncidentStore(failing, clock=clock, state_file=tmp_path / "s.json")
    monkeypatch.setattr(main, "alert_store_factory", lambda: store)
    monkeypatch.setattr(main, "ALERT_TICK_SECONDS", 3600)
    with TestClient(main.app) as client:
        incident_id = client.post("/incidents", json={"strongest_sensor": 3, "position": {"x": 1000, "y": 240}}).json()["incident_id"]
        assert client.post(f"/incidents/{incident_id}/retry").status_code == 409
        clock.advance(31)
        store.tick()
        assert client.get("/incidents/current").json()["active"]["status"] == ALERT_FAILED
        assert client.post(f"/incidents/{incident_id}/retry").status_code == 200
        resolved = client.post(f"/incidents/{incident_id}/resolve").json()
        assert resolved["active"] is None and resolved["history"][0]["delivery_status"] == "failed"
        again = client.post("/incidents", json={"strongest_sensor": 2, "position": {"x": 590, "y": 118}}).json()
        assert again["created"] is True


def test_verification_line_is_included_in_the_message(store, clock, telegram):
    incident, _ = store.inject(3, {"x": 1000, "y": 240}, verification="FINAL-E2E-20260913T1200")
    run(store, clock, 31)
    assert telegram.messages[0].endswith("Verification: FINAL-E2E-20260913T1200")


def test_env_credentials_never_leak_and_tests_never_reach_telegram(tmp_path, monkeypatch):
    from conftest import REAL_NETWORK_CALLS
    from pipe.api.alerts import store_from_env
    (tmp_path / ".env").write_text(f"TELEGRAM_BOT_TOKEN={SECRET}\nTELEGRAM_CHAT_ID=777\nALERT_PERSISTENCE_SECONDS=nan\n")
    monkeypatch.setenv("ALERT_STATE_FILE", str(tmp_path / "state.json"))
    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "ALERT_PERSISTENCE_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    clock = Clock()
    store = store_from_env(tmp_path)
    store.clock = clock
    assert store.transport.configured and store.persistence == 30.0  # nan rejected
    inject(store)
    run(store, clock, 31)
    before = len(REAL_NETWORK_CALLS)
    # The real transport was used and stopped by the guard: nothing left the machine.
    assert store.active.status == ALERT_FAILED and store.active.delivery_status == "unknown"
    snapshot = json.dumps(store.snapshot(), allow_nan=False)
    assert SECRET not in snapshot and SECRET not in (tmp_path / "state.json").read_text()
    assert REAL_NETWORK_CALLS[-1:] == ["api.telegram.org"]
    del REAL_NETWORK_CALLS[before - 1:]


def test_read_timeout_is_delivery_unknown_and_cannot_be_retried(clock, tmp_path):
    def slow(request):
        raise httpx.ReadTimeout("late", request=request)

    transport = TelegramTransport(SECRET, "777", client=httpx.Client(transport=httpx.MockTransport(slow)))
    store = IncidentStore(transport, clock=clock, state_file=tmp_path / "s.json")
    incident = inject(store)
    run(store, clock, 31)
    assert store.active.delivery_status == "unknown"
    assert store.retry(incident.incident_id) is None


def test_invalid_history_records_are_ignored_on_load(tmp_path, clock, telegram):
    state = tmp_path / "s.json"
    state.write_text('{"active": null, "history": [{"incident_id": "a", "source": "x", "strongest_sensor": 1,'
                     ' "zone": "N1", "position": {"x": 1, "y": Infinity}, "started_at": 1},'
                     ' {"incident_id": "b", "source": "x", "strongest_sensor": 1, "zone": "N1",'
                     ' "position": {"x": 1, "y": 2}, "started_at": "2026-09-13"}]}')
    store = IncidentStore(telegram, clock=clock, state_file=state)
    assert store.history == [] and json.dumps(store.snapshot(), allow_nan=False)


def test_network_guard_blocks_a_real_telegram_call():
    from conftest import REAL_NETWORK_CALLS
    before = len(REAL_NETWORK_CALLS)
    with pytest.raises(AssertionError, match="Real network call attempted"):
        TelegramTransport("000000:guard-check", "1").send("never sent")
    assert REAL_NETWORK_CALLS[before:] == ["api.telegram.org"]
    del REAL_NETWORK_CALLS[before:]  # the guard worked; nothing left the machine
