"""Suivi causal persistant, politique de démonstration et aperçus sans envoi externe."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import sqlite3
import time
from uuid import uuid4


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def preview(event, kind):
    duration = event["observed_seconds"]
    coverage = " Observation interrompue ; continuité non garantie." if event["coverage_interrupted"] else ""
    message = (f"REPLAY — Signal compatible avec une fuite observé pendant {duration} secondes. "
               "Une vérification est recommandée." if kind == "opened" else
               f"REPLAY — Retour durable sous le seuil observé après {duration} secondes surveillées. "
               "Cela ne confirme pas une réparation.")
    if kind == "persistent":
        message = (f"REPLAY — Nevil, un signal compatible avec une fuite est observé à niveau élevé "
                   f"depuis {event['current_high_seconds']} secondes consécutives. Une vérification est recommandée.")
    return {"event_id": event["event_id"], "kind": kind, "message": message + coverage,
            "delivery": "preview_only", "description_source": "template", "sent": False}


class Tracker:
    def __init__(self, path, policy, model_version):
        if (not 0 <= policy["close_threshold"] < policy["open_threshold"] <= 1
                or any(type(policy[k]) is not int or policy[k] < 1 for k in
                       ("open_seconds", "close_seconds", "max_gap_seconds"))):
            raise ValueError("Politique invalide")
        self.path, self.policy, self.model_version = str(path), policy, model_version
        if "language_after_seconds" in policy and policy["language_after_seconds"] != 30:
            raise ValueError("Cette démo de langage exige strictement plus de30secondes")
        self.boot_id = uuid4().hex
        self.version = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS previews (event_id TEXT NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL,
                                                     PRIMARY KEY(event_id, kind));
                CREATE TABLE IF NOT EXISTS language_jobs (event_id TEXT NOT NULL, kind TEXT NOT NULL,
                    history TEXT NOT NULL, PRIMARY KEY(event_id, kind));
            """)
            signature = json.dumps([model_version, self.version])
            old = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
            if old and old[0] != signature:
                raise ValueError("Nouvelle politique/modèle : utiliser une nouvelle base de sessions")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('signature', ?)", (signature,))

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=3)
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, mode="replay", now=None):
        if mode not in ("replay", "development_fixture"):
            raise ValueError("Aucune acquisition live validée")
        now = time.time() if now is None else now
        state = {"session_id": uuid4().hex, "source_mode": mode, "created_at": now,
            "model_version": self.model_version, "policy_version": self.version,
            "policy_status": "software_demo_only_not_field_validated", "boot_id": self.boot_id,
            "last_sequence": -1, "last_end": None, "last_received": None, "last_hash": None,
            "health": "starting", "ended": False, "candidate_at": None, "high_count": 0,
            "low_count": 0, "active": None, "history": [], "observed_windows": 0,
            "missing_windows": 0, "quality_errors": 0, "last_response": None}
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] >= 32:
                raise ValueError("Capacité de démonstration : 32 sessions, archiver avant nouvelle campagne")
            db.execute("INSERT INTO sessions VALUES (?, ?)", (state["session_id"], json.dumps(state)))
        return self.snapshot(state["session_id"], now)

    def read(self, db, session):
        row = db.execute("SELECT state FROM sessions WHERE id=?", (session,)).fetchone()
        if row is None:
            raise KeyError("Session inconnue")
        return json.loads(row[0])

    def public(self, state, now):
        stale = now - (state["last_received"] or state["created_at"]) > self.policy["max_gap_seconds"]
        health = "ended" if state["ended"] else ("degraded" if stale or state["boot_id"] != self.boot_id else state["health"])
        return {k: v for k, v in state.items() if k not in ("boot_id", "last_hash", "history", "last_response")} | {
            "health": health, "created_at": iso(state["created_at"]), "notifications": "preview_only"}

    def snapshot(self, session, now=None):
        with self.connection() as db:
            state = self.read(db, session)
            events = [json.loads(r[0]) for r in db.execute(
                "SELECT value FROM events WHERE session_id=? ORDER BY rowid DESC LIMIT 50", (session,))]
            previews = [json.loads(r[0]) for r in db.execute(
                "SELECT previews.value FROM previews JOIN events ON events.id=previews.event_id "
                "WHERE events.session_id=? ORDER BY previews.rowid DESC LIMIT 50", (session,))]
        return self.public(state, time.time() if now is None else now) | {"events": events, "previews": previews}

    def consume(self, session, sequence, end, received, audio_hash, probability, features=None, problem=None):
        if (type(sequence) is not int or sequence < 0 or not all(math.isfinite(v) for v in (end, received))
                or not 0 <= end <= 253402300799 or not 0 <= received <= 253402300799
                or not isinstance(audio_hash, str) or len(audio_hash) != 64
                or any(c not in "0123456789abcdef" for c in audio_hash)):
            raise ValueError("Identité/horodatage invalide")
        if probability is not None and (not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("Score invalide")
        if probability is None and not problem:
            raise ValueError("Une abstention doit avoir une raison")
        if features is not None and (len(features) != 9 or not all(math.isfinite(v) for v in features)):
            raise ValueError("Neuf mesures finies requises")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            state = self.read(db, session)
            if state["ended"]:
                raise ValueError("Session terminée")
            if sequence == state["last_sequence"]:
                if end != state["last_end"] or audio_hash != state["last_hash"]:
                    raise ValueError("Même séquence avec un autre signal/horodatage")
                return state["last_response"] | {"duplicate": True, "state": self.public(state, received)}
            if sequence < state["last_sequence"]:
                raise ValueError("Fenêtre ancienne, état inchangé")
            if end > received + .25 or received - end > self.policy["max_gap_seconds"]:
                raise ValueError("Fenêtre future ou périmée")
            if state["last_end"] is not None and abs((end - state["last_end"]) - (sequence - state["last_sequence"])) > .01:
                raise ValueError("Cadence source différente de fenêtres non chevauchantes de 1 seconde")
            if end < state["created_at"] + 1 - .01:
                raise ValueError("Fenêtre reçue avant sa disponibilité")
            missing = sequence - state["last_sequence"] - 1
            gap = missing > 0 or state["boot_id"] != self.boot_id or state["health"] == "degraded" or (
                state["last_received"] is not None and received - state["last_received"] > self.policy["max_gap_seconds"])
            state["missing_windows"] += missing
            if gap or problem:
                state.update(candidate_at=None, high_count=0, low_count=0, history=[])
                if state["active"]:
                    state["active"]["coverage_interrupted"] = True
                    state["active"]["current_high_seconds"] = 0
            new_preview = None
            if problem:
                state["quality_errors"] += 1
                state["health"] = "degraded"
            else:
                state["health"] = "fresh"
                state["observed_windows"] += 1
                state["history"] = (state["history"] + [features + [probability]])[-64:] if features is not None else []
                if state["active"]:
                    event = state["active"]
                    event["observed_seconds"] += 1
                    event["last_observed_at"] = iso(end)
                    event["current_high_seconds"] = event.get("current_high_seconds", 0) + 1 if probability >= self.policy["open_threshold"] else 0
                    event["high_observed_seconds"] = event.get("high_observed_seconds", 0) + int(probability >= self.policy["open_threshold"])
                    state["low_count"] = state["low_count"] + 1 if probability <= self.policy["close_threshold"] else 0
                    if state["low_count"] >= self.policy["close_seconds"]:
                        event.update(status="ended", ended_at=iso(end))
                        new_preview = preview(event, "ended")
                        state["active"] = None
                        state["low_count"] = 0
                    elif ("language_after_seconds" in self.policy and not event.get("language_alert_created")
                          and event["current_high_seconds"] > self.policy["language_after_seconds"]):
                        event["language_alert_created"] = True
                        new_preview = preview(event, "persistent")
                    db.execute("INSERT OR REPLACE INTO events VALUES (?, ?, ?)",
                               (event["event_id"], session, json.dumps(event)))
                elif probability >= self.policy["open_threshold"]:
                    if state["candidate_at"] is None:
                        state["candidate_at"] = end - 1
                    state["high_count"] += 1
                    if state["high_count"] >= self.policy["open_seconds"]:
                        if db.execute("SELECT COUNT(*) FROM events").fetchone()[0] >= 1000:
                            raise ValueError("Capacité de journal atteinte : archiver avant poursuite")
                        event = {"event_id": uuid4().hex, "status": "active", "session_id": session,
                            "source_mode": state["source_mode"], "first_observed_at": iso(state["candidate_at"]),
                            "confirmed_at": iso(end), "last_observed_at": iso(end),
                            "observed_seconds": state["high_count"], "coverage_interrupted": False,
                            "current_high_seconds": state["high_count"], "high_observed_seconds": state["high_count"],
                            "model_version": self.model_version, "policy_version": self.version}
                        state.update(active=event, high_count=0, candidate_at=None)
                        db.execute("INSERT INTO events VALUES (?, ?, ?)", (event["event_id"], session, json.dumps(event)))
                        new_preview = preview(event, "opened")
                else:
                    state.update(candidate_at=None, high_count=0)
            if problem and state["active"]:
                event = state["active"]
                db.execute("UPDATE events SET value=? WHERE id=?", (json.dumps(event), event["event_id"]))
            if new_preview:
                if "language_after_seconds" in self.policy:
                    if new_preview["kind"] == "opened" or (new_preview["kind"] == "ended" and not event.get("language_alert_created")):
                        new_preview = None
                    else:
                        new_preview.update(recipient="Nevil", status="pending", description_source="template_pending_model",
                            duration_source="source_timestamps_and_valid_window_counts",
                            facts={k:event[k] for k in ("first_observed_at","confirmed_at","last_observed_at",
                                "observed_seconds","current_high_seconds","high_observed_seconds","coverage_interrupted")})
                        db.execute("INSERT OR IGNORE INTO language_jobs VALUES (?, ?, ?)",
                            (new_preview["event_id"],new_preview["kind"],json.dumps(state["history"])))
            if new_preview:
                db.execute("INSERT OR IGNORE INTO previews VALUES (?, ?, ?)",
                           (new_preview["event_id"], new_preview["kind"], json.dumps(new_preview)))
            state.update(last_sequence=sequence, last_end=end, last_received=received,
                         last_hash=audio_hash, boot_id=self.boot_id)
            response = {"session_id": session, "sequence": sequence, "audio_sha256": audio_hash,
                "probability_leak": probability, "calibration": "none", "features": features,
                "quality_error": problem, "duplicate": False, "state": self.public(state, received),
                "preview": new_preview}
            state["last_response"] = response
            db.execute("UPDATE sessions SET state=? WHERE id=?", (json.dumps(state, allow_nan=False), session))
        return response

    def pending_language(self):
        with self.connection() as db:
            row = db.execute("SELECT event_id, kind, history FROM language_jobs ORDER BY rowid LIMIT 1").fetchone()
        return (row[0],row[1],json.loads(row[2])) if row else None

    def language_context(self, session):
        with self.connection() as db:
            state = self.read(db,session)
        if self.public(state,time.time())["health"] != "fresh" or len(state["history"]) < 31:
            raise ValueError("31fenêtres récentes valides requises ; laisser le contexte se remplir")
        return state["history"],state["last_sequence"]

    def finish_language(self, event_id, kind, description):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT value FROM previews WHERE event_id=? AND kind=?",(event_id,kind)).fetchone()
            if not row:
                raise KeyError("Aperçu inconnu")
            value = json.loads(row[0])
            value.update(status="ready", temporal_description=description,
                description_source=description["description_source"])
            value["message"] += " " + description["description"]
            # Même job rejoué après crash : suppression et restitution sont atomiques.
            if db.execute("SELECT 1 FROM language_jobs WHERE event_id=? AND kind=?",(event_id,kind)).fetchone():
                db.execute("UPDATE previews SET value=? WHERE event_id=? AND kind=?",(json.dumps(value),event_id,kind))
                db.execute("DELETE FROM language_jobs WHERE event_id=? AND kind=?",(event_id,kind))

    def end(self, session):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            state = self.read(db, session)
            state["ended"] = True
            state["history"] = []
            if state["active"]:
                state["active"].update(status="observation_stopped", coverage_interrupted=True)
                db.execute("UPDATE events SET value=? WHERE id=?", (json.dumps(state["active"]), state["active"]["event_id"]))
            db.execute("UPDATE sessions SET state=? WHERE id=?", (json.dumps(state), session))
        return self.snapshot(session)
