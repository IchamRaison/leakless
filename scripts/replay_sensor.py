#!/usr/bin/env python3
"""Source acoustique locale finie, sans modèle ni réseau (bibliothèque standard)."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time
import uuid
import wave


def audio_bytes(path, expected):
    # Une fenêtre et ses en-têtes seulement : refuser les fichiers démesurés.
    with path.open('rb') as stream:
        payload = stream.read(65537)
    if len(payload) > 65536:
        raise ValueError(f'{path}: WAV trop volumineux')
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f'{path}: SHA-256 différent')
    try:
        with wave.open(io.BytesIO(payload), 'rb') as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                    wav.getnframes(), wav.getcomptype()) != (1, 2, 8000, 8000, 'NONE'):
                raise ValueError(f'{path}: attendu mono PCM16, 8 kHz, 8000 échantillons')
            if len(wav.readframes(8000)) != 16000:
                raise ValueError(f'{path}: WAV tronqué')
    except (wave.Error, EOFError) as exc:
        raise ValueError(f'{path}: WAV invalide') from exc
    return payload


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name}: nombre fini positif ou nul requis')
    return value


def load_scenario(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('schema_version doit valoir 1')
    device = data.get('device_id')
    if not isinstance(device, str) or not device or len(device) > 128:
        raise ValueError('device_id opaque non vide requis (128 caractères maximum)')
    if data.get('source_mode') not in ('replay', 'development_fixture'):
        raise ValueError('source_mode doit être replay ou development_fixture')
    entries = data.get('entries')
    if not isinstance(entries, list) or not entries:
        raise ValueError('entries doit être une liste non vide')
    previous = -8000
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('entrée invalide')
        offset = entry.get('source_offset_samples')
        if type(offset) is not int or offset < previous + 8000:
            raise ValueError('offsets entiers croissants, fenêtres sans chevauchement requis')
        previous = offset
        allowed = ('synthetic',) if data['source_mode'] == 'development_fixture' else ('train', 'authorized_demo')
        if entry.get('provenance') not in allowed:
            raise ValueError(f'provenance autorisée : {allowed}')
        if not isinstance(entry.get('path'), str) or not entry['path']:
            raise ValueError('path requis')
        entry['path'] = str((path.parent / entry['path']).resolve())
        audio_bytes(Path(entry['path']), entry.get('audio_sha256'))
    incidents = data.get('incidents', [])
    if not isinstance(incidents, list):
        raise ValueError('incidents doit être une liste')
    seen = set()
    for incident in incidents:
        if not isinstance(incident, dict):
            raise ValueError('incident invalide')
        seq = incident.get('sequence')
        if type(seq) is not int or not 0 <= seq < len(entries) or seq in seen:
            raise ValueError('un incident maximum par séquence existante')
        seen.add(seq)
        if incident.get('kind') not in ('drop', 'delay', 'duplicate'):
            raise ValueError('incident inconnu')
        if incident['kind'] == 'delay':
            number(incident.get('seconds'), 'delay.seconds')
    number(data.get('consumer_seconds', 0), 'consumer_seconds')
    return data


class Clock:
    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)
    utcnow = staticmethod(lambda: datetime.now(timezone.utc))


def replay(scenario_path, output, clock=None, receiver=None):
    """Le récepteur local ne fait que confirmer des octets WAV, jamais une prédiction."""
    clock = clock or Clock()
    data = load_scenario(Path(scenario_path))  # Tout valider avant de créer le run.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    session = receiver.start(data['source_mode']) if receiver else str(uuid.uuid4())
    (output / 'manifest.json').write_text(json.dumps(
        dict(data, session_id=session, chronology='artificial'), indent=2) + '\n')
    origin, utc_origin = clock.monotonic(), clock.utcnow()
    incidents = {item['sequence']: item for item in data.get('incidents', [])}
    # ponytail: métadonnées O(n) pour un scénario fini ; lecture JSONL si scénarios massifs.
    events = []
    for seq, entry in enumerate(data['entries']):
        due = (entry['source_offset_samples'] + 8000) / 8000
        incident = incidents.get(seq, {})
        arrival = due + (incident.get('seconds', 0) if incident.get('kind') == 'delay' else 0)
        events.append((arrival, seq, due, incident.get('kind')))
    events.sort()
    counts = Counter()
    pending = active = None
    busy_until = 0
    index = 0
    status = 'completed'
    with (output / 'events.jsonl').open('x') as log:
        def record(event, envelope=None, **extra):
            counts[event] += 1
            row = dict(event=event, session_id=session, at=clock.utcnow().isoformat(),
                       elapsed_seconds=clock.monotonic() - origin, **extra)
            if envelope:
                row.update({key: value for key, value in envelope.items() if key != 'payload'})
            log.write(json.dumps(row) + '\n')
            log.flush()

        def offer(envelope):
            nonlocal pending
            identity = (envelope['session_id'], envelope['sequence'])
            if any(item is not None and (item['session_id'], item['sequence']) == identity
                   for item in (active, pending)):
                record('duplicate', envelope, transport_result='ignored')
                return
            if pending is not None and pending['sequence'] > envelope['sequence']:
                record('lost', envelope, reason='older_than_pending')
                return
            if pending is not None:
                record('lost', pending, reason='replaced_pending')
            pending = envelope

        record('started')
        try:
            while index < len(events) or pending is not None or active is not None:
                now = clock.monotonic() - origin
                if active is not None and now >= busy_until:
                    prediction = receiver.window(active) if receiver else None
                    record('received', active, received_at=clock.utcnow().isoformat(),
                           transport_result='accepted', prediction=prediction)
                    active = None
                # Évacuer les échéances passées avant d'envoyer : jamais de rafale historique.
                while index < len(events) and events[index][0] <= now:
                    arrival, seq, due, kind = events[index]
                    index += 1
                    entry = data['entries'][seq]
                    base = dict(schema_version=1, device_id=data['device_id'], session_id=session,
                                sequence=seq, source_mode=data['source_mode'],
                                source_offset_samples=entry['source_offset_samples'],
                                sample_rate_hz=8000, n_samples=8000, encoding='pcm_s16le_mono',
                                scheduled_at=(utc_origin + timedelta(seconds=due)).isoformat(),
                                audio_sha256=entry['audio_sha256'])
                    if kind == 'drop':
                        record('lost', base, reason='transport_cut')
                        continue
                    payload = audio_bytes(Path(entry['path']), entry['audio_sha256'])
                    envelope = dict(base, emitted_at=clock.utcnow().isoformat(), payload=payload)
                    record('emitted', envelope, lateness_seconds=now - due)
                    offer(envelope)
                    if kind == 'duplicate':
                        # Même récepteur et même identité pour cette retransmission immédiate.
                        record('emitted', envelope, lateness_seconds=now - due, retransmission=True)
                        offer(envelope)
                    now = clock.monotonic() - origin
                if active is None and pending is not None:
                    active, pending = pending, None
                    busy_until = clock.monotonic() - origin + data.get('consumer_seconds', 0)
                    record('processing', active)
                deadlines = []
                if index < len(events):
                    deadlines.append(events[index][0])
                if active is not None:
                    deadlines.append(busy_until)
                if deadlines:
                    clock.sleep(max(0, origin + min(deadlines) - clock.monotonic()))
        except KeyboardInterrupt:
            status = 'interrupted'
        except Exception as exc:
            status = 'failed'
            record('error', error=str(exc))
            raise
        finally:
            try:
                if receiver:
                    receiver.end()
            except Exception as exc:
                status = 'failed'
                record('error', error=str(exc), phase='end_session')
                raise
            finally:
                for envelope in (active, pending):
                    if envelope is not None:
                        record('lost', envelope, reason=status)
                record('finished', status=status, unprocessed_positions=len(events) - index,
                       counters=dict(counts))
    return dict(session_id=session, status=status, counters=dict(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = replay(args.scenario, args.output)
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(2, f'Erreur : {exc}\n')
    print(json.dumps(result))
    return 130 if result['status'] == 'interrupted' else 0


if __name__ == '__main__':
    raise SystemExit(main())
