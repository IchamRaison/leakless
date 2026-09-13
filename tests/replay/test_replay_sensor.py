import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('replay_sensor', ROOT / 'scripts/replay_sensor.py')
sensor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensor)


class FakeClock:
    def __init__(self):
        self.now = 0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def utcnow(self):
        return datetime.fromtimestamp(1700000000 + self.now, timezone.utc)


def fixture(directory):
    directory = Path(directory)
    with wave.open(str(directory / 'constant.wav'), 'wb') as wav:
        wav.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
        wav.writeframes(bytes(16000))
    data = json.loads((ROOT / 'examples/replay/scenario.json').read_text())
    digest = hashlib.sha256((directory / 'constant.wav').read_bytes()).hexdigest()
    for entry in data['entries']:
        entry['audio_sha256'] = digest
    path = directory / 'scenario.json'
    path.write_text(json.dumps(data))
    return path


class ReplayTest(unittest.TestCase):
    def test_transport_and_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = fixture(base)
            data = json.loads(path.read_text())

            def run(name, **changes):
                scenario = dict(data, **changes)
                path.write_text(json.dumps(scenario))
                result = sensor.replay(path, base / name, FakeClock())
                rows = [json.loads(line) for line in (base / name / 'events.jsonl').read_text().splitlines()]
                return result, rows

            normal, rows = run('normal', incidents=[], consumer_seconds=0)
            received = [row for row in rows if row['event'] == 'received']
            self.assertEqual([r['sequence'] for r in received], list(range(8)))
            self.assertEqual([r['elapsed_seconds'] for r in received], list(range(1, 9)))
            self.assertEqual(len({r['audio_sha256'] for r in received}), 1)
            self.assertTrue(all('path' not in r and 'payload' not in r and 'provenance' not in r for r in received))
            faults, rows = run('faults')
            self.assertEqual(faults['counters']['duplicate'], 1)
            self.assertEqual(faults['counters']['lost'], 3)
            self.assertEqual(faults['counters']['received'], 5)
            late = next(r for r in rows if r['event'] == 'emitted' and r['sequence'] == 5)
            self.assertEqual(late['lateness_seconds'], 0.5)
            self.assertNotEqual(normal['session_id'], faults['session_id'])
            slow, rows = run('slow', incidents=[], consumer_seconds=3)
            self.assertGreater(slow['counters']['lost'], 0)
            self.assertEqual(slow['counters']['received'] + slow['counters']['lost'], 8)
            self.assertEqual([r for r in rows if r['event'] == 'received'][-1]['sequence'], 7)
            delayed, rows = run('delayed_old', consumer_seconds=5,
                                incidents=[{'sequence': 1, 'kind': 'delay', 'seconds': 2.5}])
            self.assertTrue(any(r.get('reason') == 'older_than_pending' for r in rows))
            self.assertEqual(delayed['counters']['received'] + delayed['counters']['lost'], 8)
            with self.assertRaises(FileExistsError):
                sensor.replay(path, base / 'slow', FakeClock())
            for invalid in [dict(data, source_mode='live'), dict(data, consumer_seconds=float('nan')),
                            dict(data, incidents=[{'sequence': 99, 'kind': 'drop'}]),
                            dict(data, entries=[dict(data['entries'][0], provenance='test')]),
                            dict(data, entries=[dict(data['entries'][0], audio_sha256='wrong')])]:
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    sensor.replay(path, base / 'invalid', FakeClock())
                self.assertFalse((base / 'invalid').exists())
            payload = (base / 'constant.wav').read_bytes()
            (base / 'constant.wav').write_bytes(payload[:-2])
            digest = hashlib.sha256(payload[:-2]).hexdigest()
            with self.assertRaisesRegex(ValueError, 'tronqué'):
                sensor.audio_bytes(base / 'constant.wav', digest)
            (base / 'constant.wav').write_bytes(b'not a wave')
            with self.assertRaisesRegex(ValueError, 'invalide'):
                sensor.audio_bytes(base / 'constant.wav', hashlib.sha256(b'not a wave').hexdigest())

    def test_clock_lag_and_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = fixture(base)
            data = json.loads(path.read_text())
            data['incidents'] = []
            path.write_text(json.dumps(data))

            class LagClock(FakeClock):
                def sleep(self, seconds):
                    self.now += seconds + (4 if self.now == 0 else 0)

            result = sensor.replay(path, base / 'lag', LagClock())
            self.assertEqual(result['counters']['lost'], 4)

            class InterruptedClock(FakeClock):
                def sleep(self, seconds):
                    raise KeyboardInterrupt

            result = sensor.replay(path, base / 'interrupted', InterruptedClock())
            self.assertEqual(result['status'], 'interrupted')
            self.assertEqual(result['counters']['finished'], 1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-demo', type=Path)
    args = parser.parse_args()
    if args.write_demo:
        args.write_demo.mkdir(parents=True, exist_ok=False)
        print(fixture(args.write_demo))
    else:
        unittest.main()
