"""Contrôles mécaniques sur données synthétiques, sans accès aux réserves."""
import json
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/temporal"))
sys.path.insert(0, str(ROOT / "scripts/eval"))
from run_campaign import partition
from prepare_external_holdout import expected_recordings
from pipe.temporal_model import acoustic_features, decode_pcm, digest, C1Detector, v2_c1
from pipe.sequence_model import SequenceModel


class ModelsTest(unittest.TestCase):
    def test_partition_keeps_sensors_and_counts(self):
        config = json.loads((ROOT / "configs/temporal/c1_lstm.json").read_text())
        counts, groups = {}, {}
        for record in expected_recordings().values():
            if record["subset"] != "primary":
                continue
            fold = partition(record, config)
            counts[fold] = counts.get(fold, 0) + 1
            groups.setdefault(record["condition_key"], set()).add(fold)
        self.assertEqual(counts, {"train": 40, "validation": 20, "confirmation": 60})
        self.assertTrue(all(len(folds) == 1 for folds in groups.values()))

    def test_pcm_and_exact_shared_features(self):
        raw = (np.sin(np.arange(8000)) * 10000).astype("<i2")
        data = BytesIO()
        with wave.open(data, "wb") as wav:
            wav.setparams((1, 2, 8000, 8000, "NONE", "none"))
            wav.writeframes(raw.tobytes())
        decoded = decode_pcm(data.getvalue())
        np.testing.assert_array_equal(acoustic_features(decoded), v2_c1.features.c1_envelope(raw.astype(float)))
        for invalid in (np.zeros(8000), np.ones(8000), np.full(8000, np.nan), np.zeros(7999)):
            with self.assertRaises(ValueError):
                acoustic_features(invalid)
        with self.assertRaises(ValueError):
            decode_pcm(b"invalid")

    def test_checkpoint_and_continuous_scores(self):
        rng = np.random.default_rng(12)
        X, y = rng.normal(size=(20, 9)), np.arange(20) % 2
        scaler, classifier = v2_c1.fit_final(X, y, .01)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha = v2_c1.save_checkpoint(root / "c1.pkl", scaler, classifier)
            (root / "bundle.json").write_text(json.dumps({"files": {"c1.pkl": sha}}))
            loaded = C1Detector(root, digest(root / "bundle.json"))
            np.testing.assert_array_equal(v2_c1.predict_probability(scaler, classifier, X),
                v2_c1.predict_probability(loaded.scaler, loaded.classifier, X))
            with self.assertRaises(ValueError):
                C1Detector(root, "0" * 64)

    def test_causality_streaming_gradient(self):
        torch.manual_seed(12)
        torch.set_num_threads(2)
        model = SequenceModel().eval()
        x = torch.randn(4, 30, 10)
        full, _ = model(x)
        other = x.clone(); other[:, 10:] = 123
        changed, _ = model(other)
        torch.testing.assert_close(full[:, :10], changed[:, :10], atol=1e-6, rtol=1e-6)
        state, stream = None, []
        for i in range(30):
            output, state = model(x[:, i:i+1], state)
            stream.append(output)
        torch.testing.assert_close(full, torch.cat(stream, dim=1), atol=1e-6, rtol=1e-6)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(full[:, -1], torch.tensor([0., 1., 0., 1.]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(sum(p.grad.abs().sum() for p in model.parameters())), 0)


if __name__ == "__main__":
    unittest.main()
