"""Scénarios logiciels : aucune performance terrain déduite de scores injectés."""
import json
from pathlib import Path
import tempfile
import unittest

from pipe.temporal_tracking import Tracker

POLICY = {"open_threshold": .8, "close_threshold": .4, "open_seconds": 3,
          "close_seconds": 5, "max_gap_seconds": 2}


class TrackingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "state.db"
        self.tracker = Tracker(self.path, POLICY, "test-model")
        self.session = self.tracker.create(now=100)["session_id"]

    def put(self, seq, p=.9, problem=None):
        return self.tracker.consume(self.session, seq, 101 + seq, 101 + seq,
                                    "a" * 64, p, [1.] * 9 if p is not None else None, problem)

    def test_persistence_dedup_end(self):
        self.assertIsNone(self.put(0)["preview"])
        self.assertIsNone(self.put(1)["preview"])
        opened = self.put(2)
        self.assertEqual(opened["state"]["active"]["observed_seconds"], 3)
        self.assertFalse(opened["preview"]["sent"])
        self.assertTrue(self.put(2)["duplicate"])
        self.assertEqual(len(self.tracker.snapshot(self.session, 103)["previews"]), 1)
        for seq in range(3, 7):
            self.assertIsNone(self.put(seq, .2)["preview"])
        ended = self.put(7, .2)
        self.assertEqual(ended["preview"]["kind"], "ended")
        self.assertIsNone(ended["state"]["active"])
        self.assertEqual(len(self.tracker.snapshot(self.session, 108)["previews"]), 2)

    def test_brief_noise_gap_invalid_and_restart(self):
        self.put(0); self.put(1, .2)
        self.assertIsNone(self.put(2)["preview"])
        self.assertIsNone(self.put(4)["preview"])
        bad = self.put(5, None, "silence")
        self.assertEqual(bad["state"]["health"], "degraded")
        self.assertEqual(bad["state"]["missing_windows"], 1)
        self.put(6); self.put(7); event = self.put(8)["state"]["active"]
        self.tracker = Tracker(self.path, POLICY, "test-model")
        self.assertEqual(self.tracker.snapshot(self.session, 109)["health"], "degraded")
        self.assertTrue(self.put(8)["duplicate"])
        resumed = self.put(9)
        self.assertEqual(resumed["state"]["active"]["event_id"], event["event_id"])
        self.assertTrue(resumed["state"]["active"]["coverage_interrupted"])
        self.assertEqual(len(self.tracker.snapshot(self.session, 110)["previews"]), 1)
        self.assertEqual(self.tracker.snapshot(self.session, 114)["health"], "degraded")
        self.assertEqual(self.tracker.end(self.session)["active"]["status"], "observation_stopped")

    def test_bad_identity_time_score_and_version(self):
        self.put(0); self.put(1)
        before = self.tracker.snapshot(self.session, 102)
        for args in [(0, 101, 102, "a" * 64, .9), (1, 102, 102, "b" * 64, .9),
                     (2, 200, 103, "a" * 64, .9), (2, 103, 106, "a" * 64, .9),
                     (2, 103, 103, "a" * 64, float("nan")), (2, 103, 103, "z" * 64, .9)]:
            with self.assertRaises(ValueError):
                self.tracker.consume(self.session, *args)
        self.assertEqual(self.tracker.snapshot(self.session, 102), before)
        with self.assertRaises(ValueError):
            Tracker(self.path, POLICY, "another-model")
        self.tracker.end(self.session)
        with self.assertRaises(ValueError):
            self.put(2)


if __name__ == "__main__":
    unittest.main()
