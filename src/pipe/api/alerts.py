"""Incidents et alertes, décidés côté serveur.

Machine d'états : NORMAL -> ANOMALY_PENDING -> ALERT_DISPATCHING -> ALERT_SENT | ALERT_FAILED,
puis RESOLVED. Le seuil de persistance (30 s par défaut) est évalué par une boucle serveur :
l'alerte part même si aucun navigateur n'est ouvert. Un incident n'est envoyé qu'une fois,
dédupliqué par incident_id ; les lectures (polling, rechargement) n'ont aucun effet de bord.
Les secrets Telegram restent côté serveur et ne sont jamais renvoyés au client ni journalisés.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

import httpx

log = logging.getLogger("leakless.alerts")
# httpx journalise l'URL de requête, qui contient le token : jamais au-dessous de WARNING.
logging.getLogger("httpx").setLevel(logging.WARNING)

NORMAL = "NORMAL"
ANOMALY_PENDING = "ANOMALY_PENDING"
ALERT_DISPATCHING = "ALERT_DISPATCHING"
ALERT_SENT = "ALERT_SENT"
ALERT_FAILED = "ALERT_FAILED"
RESOLVED = "RESOLVED"

ZONES = {1: "N1", 2: "N2", 3: "N3"}


class TransportError(Exception):
    """Échec d'envoi exploitable : le message ne contient jamais de secret."""


class DeliveryUnknown(TransportError):
    """La requête a pu atteindre Telegram sans réponse lisible : ne jamais renvoyer automatiquement."""


class Transport(Protocol):
    name: str
    configured: bool

    def send(self, text: str) -> str: ...


class TelegramTransport:
    name = "telegram"

    def __init__(self, token: str | None, chat_id: str | None, timeout: float = 10.0,
                 client: httpx.Client | None = None):
        self._token = (token or "").strip()
        self._chat_id = (chat_id or "").strip()
        self._timeout = timeout
        self._client = client
        self.configured = bool(self._token and self._chat_id)

    def send(self, text: str) -> str:
        if not self.configured:
            raise TransportError("Telegram not configured: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID on the server.")
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        client = self._client or httpx.Client(timeout=self._timeout)
        try:
            response = client.post(url, json={"chat_id": self._chat_id, "text": text})
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # Le type suffit : l'URL de l'exception contient le token. Connexion impossible : rien n'est parti.
            raise TransportError(f"Telegram unreachable ({type(exc).__name__}).") from None
        except httpx.HTTPError as exc:
            # Envoyé peut-être, réponse perdue (timeout de lecture, coupure) : issue inconnue.
            raise DeliveryUnknown(f"Telegram delivery unknown ({type(exc).__name__}).") from None
        finally:
            if self._client is None:
                client.close()
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code != 200 or not body.get("ok"):
            description = str(body.get("description") or f"HTTP {response.status_code}")
            raise TransportError(f"Telegram rejected the message: {description}.")
        message_id = (body.get("result") or {}).get("message_id")
        if message_id is None:
            raise TransportError("Telegram answered without a message_id.")
        return str(message_id)


@dataclass
class Incident:
    incident_id: str
    source: str
    strongest_sensor: int
    zone: str
    position: dict
    started_at: float
    status: str = ANOMALY_PENDING
    confirmed_at: float | None = None
    dispatch_started_at: float | None = None
    alert_sent_at: float | None = None
    resolved_at: float | None = None
    transport: str | None = None
    delivery_status: str = "not_dispatched"
    telegram_message_id: str | None = None
    failure_reason: str | None = None
    attempts: int = 0
    retry_requested: bool = False
    verification: str | None = None
    events: list = field(default_factory=list)


def clock_text(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M:%S")


def alert_text(incident: Incident, persistence: float, sent_at: float | None = None) -> str:
    source = "injected test incident" if incident.source == "injected_test" else incident.source
    return "\n".join([
        "🚨 LeakLess Alert",
        "",
        f"Persistent abnormal signal detected for {persistence:g} seconds.",
        f"Strongest response: Sensor {incident.strongest_sensor}",
        f"Zone: {incident.zone}",
        "Action: inspection recommended.",
        f"Detected: {clock_text(incident.started_at)}",
        "",
        f"Source: {source}",
        *([f"Verification: {incident.verification}"] if incident.verification else []),
    ])


class IncidentStore:
    def __init__(self, transport: Transport, persistence_seconds: float = 30.0,
                 clock: Callable[[], float] = time.time, state_file: Path | None = None,
                 history_limit: int = 50):
        self.transport = transport
        self.persistence = persistence_seconds
        self.clock = clock
        self.state_file = state_file
        self.history_limit = history_limit
        self._lock = threading.RLock()
        self.active: Incident | None = None
        self.history: list[Incident] = []
        self.sends = 0
        self.storage_error: str | None = None
        self._load()

    # -- persistance -----------------------------------------------------------------------
    def _load(self):
        if not self.state_file or not self.state_file.exists():
            return
        try:
            raw = json.loads(self.state_file.read_text())
            history = [item for item in (_valid_incident(raw) for raw in raw.get("history", [])) if item]
            active = _valid_incident(raw["active"]) if raw.get("active") else None
        except (OSError, ValueError, TypeError, KeyError) as exc:
            log.error("Incident state unreadable (%s): starting empty.", type(exc).__name__)
            return
        self.history, self.active = history, active
        if self.active and self.active.status == ALERT_DISPATCHING:
            # Arrêt pendant un envoi : issue inconnue, jamais renvoyée automatiquement.
            self._fail(self.active, "Server restarted during dispatch; delivery unknown, not retried.", unknown=True)
            self._save()

    def _save(self) -> bool:
        """Écriture best-effort : un disque plein ne fige jamais l'état ; renvoie False en cas d'échec."""
        if not self.state_file:
            return True
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "active": asdict(self.active) if self.active else None,
                "history": [asdict(item) for item in self.history],
            }, indent=2, allow_nan=False))
            tmp.replace(self.state_file)
            self.storage_error = None
            return True
        except (OSError, TypeError, ValueError) as exc:
            self.storage_error = f"Incident state not saved ({type(exc).__name__})."
            log.error("Incident state NOT saved to %s (%s).", self.state_file, type(exc).__name__)
            return False

    # -- transitions -----------------------------------------------------------------------
    @staticmethod
    def _event(incident: Incident, kind: str, at: float, detail: str | None = None):
        incident.events.append({"kind": kind, "at": at, **({"detail": detail} if detail else {})})

    def inject(self, strongest_sensor: int, position: dict, source: str = "injected_test",
               verification: str | None = None) -> tuple[Incident, bool]:
        """Nouvel incident, ou l'incident déjà actif (aucun doublon)."""
        if not _finite_position(position):
            raise ValueError("Incident position must be finite numbers.")
        with self._lock:
            if self.active is not None:
                return self.active, False
            now = self.clock()
            incident = Incident(
                incident_id=f"inc-{uuid.uuid4().hex[:10]}", source=source,
                strongest_sensor=strongest_sensor, zone=ZONES.get(strongest_sensor, f"N{strongest_sensor}"),
                position={"x": float(position["x"]), "y": float(position["y"])}, started_at=now,
                verification=verification)
            self._event(incident, "anomaly_started", now)
            self.active = incident
            self._save()
            log.info("Incident %s started (strongest response: sensor %s).", incident.incident_id, strongest_sensor)
            return incident, True

    def resolve(self, incident_id: str) -> Incident | None:
        with self._lock:
            incident = self.active
            if incident is None or incident.incident_id != incident_id:
                return next((item for item in self.history if item.incident_id == incident_id), None)
            incident.resolved_at = self.clock()
            if incident.status == ALERT_DISPATCHING:
                # L'envoi en cours décide de l'issue ; la clôture suit son retour.
                self._save()
                return incident
            self._close(incident)
            return incident

    def retry(self, incident_id: str) -> Incident | None:
        """Nouvel essai explicite, uniquement après un échec : jamais de second envoi réussi."""
        with self._lock:
            incident = self.active
            if (incident is None or incident.incident_id != incident_id or incident.status != ALERT_FAILED
                    or incident.delivery_status != "failed"):
                # Jamais après un envoi réussi ni une issue inconnue : pas de doublon possible.
                return None
            incident.retry_requested = True
            self._event(incident, "retry_requested", self.clock())
            self._save()
            return incident

    def _close(self, incident: Incident):
        incident.status = RESOLVED
        self._event(incident, "resolved", incident.resolved_at or self.clock())
        self.history.insert(0, incident)
        del self.history[self.history_limit:]
        self.active = None
        self._save()
        log.info("Incident %s resolved (delivery: %s).", incident.incident_id, incident.delivery_status)

    def _fail(self, incident: Incident, reason: str, unknown: bool = False):
        incident.status = ALERT_FAILED
        incident.delivery_status = "unknown" if unknown else "failed"
        incident.failure_reason = reason
        incident.retry_requested = False
        self._event(incident, "alert_failed", self.clock(), reason)
        log.error("Incident %s: alert NOT delivered via %s. %s The incident stays active: "
                  "retry the notification or resolve it before a new incident can start.",
                  incident.incident_id, self.transport.name, reason)

    def tick(self) -> None:
        """Évalue la persistance ; au plus un envoi réussi par incident. Ne lève jamais."""
        incident = None
        try:
            with self._lock:
                incident = self.active
                if incident is None:
                    return
                retrying = incident.status == ALERT_FAILED and incident.retry_requested
                if incident.status != ANOMALY_PENDING and not retrying:
                    return
                now = self.clock()
                if not retrying:
                    if now - incident.started_at < self.persistence:
                        return
                    incident.confirmed_at = incident.started_at + self.persistence
                    self._event(incident, "persistence_confirmed", incident.confirmed_at)
                incident.retry_requested = False
                incident.status = ALERT_DISPATCHING
                incident.dispatch_started_at = now
                incident.transport = self.transport.name
                incident.delivery_status = "dispatching"
                incident.attempts += 1
                if not self._save():
                    # Sans trace durable de l'envoi, un redémarrage pourrait renvoyer : on n'envoie pas.
                    self._fail(incident, "Alert not sent: incident state could not be saved.")
                    return
                self.sends += 1
                text = alert_text(incident, self.persistence, now)
        except Exception as exc:  # défense : une anomalie de préparation ne fige jamais l'incident
            log.exception("Alert preparation failed.")
            if incident is not None:
                with self._lock:
                    if incident.status in (ALERT_DISPATCHING, ANOMALY_PENDING):
                        self._fail(incident, f"Alert preparation failed ({type(exc).__name__}).")
                        self._save()
            return
        # Envoi hors verrou : le polling reste servi pendant l'appel réseau.
        try:
            message_id = self.transport.send(text)
        except Exception as exc:
            reason = str(exc) if isinstance(exc, TransportError) else f"Unexpected transport error ({type(exc).__name__})."
            with self._lock:
                self._fail(incident, reason, unknown=not isinstance(exc, TransportError) or isinstance(exc, DeliveryUnknown))
                if incident.resolved_at is not None:
                    self._close(incident)
                else:
                    self._save()
            return
        try:
            self._record_sent(incident, message_id)
        except Exception:
            log.exception("Alert sent (message_id %s) but recording it failed.", message_id)

    def _record_sent(self, incident: Incident, message_id: object) -> None:
        with self._lock:
            incident.status = ALERT_SENT
            incident.delivery_status = "sent"
            incident.alert_sent_at = self.clock()
            incident.telegram_message_id = str(message_id)
            incident.failure_reason = None
            self._event(incident, "alert_sent", incident.alert_sent_at)
            log.warning("Incident %s: alert sent via %s (message_id %s).",
                        incident.incident_id, self.transport.name, message_id)
            if incident.resolved_at is not None:
                self._close(incident)
            else:
                self._save()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "server_time": self.clock(),
                "persistence_seconds": self.persistence,
                "transport": {"name": self.transport.name, "configured": self.transport.configured},
                "storage_ok": self.storage_error is None,
                "active": asdict(self.active) if self.active else None,
                "history": [asdict(item) for item in self.history[:10]],
            }


def _valid_incident(raw: object) -> Incident | None:
    """Incident relu du disque, ou None s'il est incohérent (jamais de NaN ni de type inattendu)."""
    try:
        incident = Incident(**raw)  # type: ignore[arg-type]
        times = [incident.started_at, incident.confirmed_at, incident.dispatch_started_at,
                 incident.alert_sent_at, incident.resolved_at]
        if not _finite_position(incident.position) or not all(
                t is None or (isinstance(t, (int, float)) and math.isfinite(t)) for t in times):
            raise ValueError("invalid incident")
        if incident.started_at is None:
            raise ValueError("missing started_at")
        return incident
    except (TypeError, ValueError):
        log.error("Ignoring an invalid incident record in the state file.")
        return None


def _finite_position(position: object) -> bool:
    try:
        return all(math.isfinite(float(position[key])) for key in ("x", "y"))  # type: ignore[index]
    except (TypeError, ValueError, KeyError):
        return False


def load_env_file(path: Path) -> None:
    """Charge un .env local (ignoré par git) sans écraser l'environnement existant."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def store_from_env(root: Path) -> IncidentStore:
    load_env_file(root / ".env")
    try:
        persistence = float(os.environ.get("ALERT_PERSISTENCE_SECONDS", "30"))
    except ValueError:
        persistence = 30.0
    if not math.isfinite(persistence) or persistence <= 0:
        log.error("ALERT_PERSISTENCE_SECONDS must be a finite positive number; using 30.")
        persistence = 30.0
    transport = TelegramTransport(os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID"))
    if not transport.configured:
        log.warning("Telegram transport not configured: confirmed alerts will be marked ALERT_FAILED.")
    state = Path(os.environ.get("ALERT_STATE_FILE", str(root / "data" / "incidents.json")))
    return IncidentStore(transport, persistence_seconds=persistence, state_file=state)
