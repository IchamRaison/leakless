"""Garde-fous de test : aucun appel réseau réel, jamais Telegram, même si .env contient des secrets."""

import httpx
import pytest

import pipe.api.main as main
from pipe.api.alerts import IncidentStore

REAL_NETWORK_CALLS = []


class OfflineTransport:
    name = "telegram"
    configured = True

    def __init__(self):
        self.messages = []

    def send(self, text):
        self.messages.append(text)
        return str(len(self.messages))


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch, tmp_path):
    """Bloque le transport HTTP réel d'httpx et isole le store d'alertes de .env et de data/."""
    def blocked(self, request):
        REAL_NETWORK_CALLS.append(str(request.url.host))
        raise AssertionError(f"Real network call attempted during tests: {request.url.host}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(main, "alert_store_factory",
                        lambda: IncidentStore(OfflineTransport(), state_file=tmp_path / "incidents.json"))
    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(key, raising=False)
    yield


def pytest_sessionfinish(session, exitstatus):
    if REAL_NETWORK_CALLS:
        session.exitstatus = 1
    print(f"\nREAL_TELEGRAM_CALLS_DURING_TESTS = {sum(1 for host in REAL_NETWORK_CALLS if 'telegram' in host)}"
          f" · REAL_NETWORK_CALLS = {len(REAL_NETWORK_CALLS)}")
