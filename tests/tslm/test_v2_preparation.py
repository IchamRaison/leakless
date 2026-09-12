"""Contrôle des nouvelles copies/cache ; aucune mutation de V1 ni lecture test."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
from make_v2_reference import make_reference
from pipe.tslm import prepare
from pipe.tslm import preprocessing
from pipe.tslm.preprocessing import CANONICAL_VERSION, VERSION


class V2PreparationChecks(unittest.TestCase):
    def test_reference_is_a_new_independent_identity_with_unchanged_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "v1"
            (source / "base").mkdir(parents=True)
            metadata = {key: "fixture" for key in (
                "base_model", "base_revision", "architecture", "opentslm_revision", "timenet_revision",
                "protocol", "manifest_sha256", "training_commit", "model_version")}
            metadata.update(preprocessing_version=VERSION, config_hash="a" * 64, max_new_tokens=48,
                            scoring_spec={}, training_clip_ids=["train"], training_groups=["group"])
            (source / "metadata.json").write_text(json.dumps(metadata))
            for name in ("base/config.json", "base/model.safetensors", "temporal.pt", "scoring_spec.json",
                         "QWEN-LICENSE", "QWEN-MODEL-CARD.md", "requirements-ml.lock", "PROVENANCE.json", "training-report.json"):
                (source / name).write_bytes(b"fixture")
            original = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in source.rglob("*") if p.is_file()}
            (source / "checksums.json").write_text(json.dumps(original))
            result = make_reference(source, root / "v2", "b" * 40)
            revised = json.loads((root / "v2/metadata.json").read_text())
            self.assertFalse(result["retrained"])
            self.assertTrue(revised["single_clip_acoustic_encoding"])
            self.assertEqual(revised["variant"], "A")
            self.assertIs(revised["amplitude_evidence"], False)
            self.assertNotIn("amplitude_evidence_version", revised)
            self.assertNotIn("amplitude_evidence", revised["scoring_spec"])
            self.assertEqual(revised["scoring_spec"]["acoustic_batching"], "one_clip_four_channels")
            self.assertEqual(revised["preprocessing_version"], CANONICAL_VERSION)
            self.assertNotEqual(revised["config_hash"], metadata["config_hash"])
            self.assertEqual(result["temporal_sha256"], original["temporal.pt"])
            with self.assertRaises(ValueError):
                make_reference(source, root / "bad-opt-in", "b" * 40, amplitude_evidence=1)
            self.assertFalse((root / "bad-opt-in").exists())
            result_c = make_reference(source, root / "v2-c", "b" * 40, amplitude_evidence=True)
            revised_c = json.loads((root / "v2-c/metadata.json").read_text())
            config_c = json.loads((root / "v2-c/config.json").read_text())
            self.assertEqual(revised_c["variant"], "C")
            self.assertIs(revised_c["amplitude_evidence"], True)
            self.assertIs(revised_c["single_clip_acoustic_encoding"], True)
            self.assertFalse(revised_c["retrained"])
            self.assertFalse(revised_c["use_for_cv_initialization"])
            self.assertEqual(revised_c["amplitude_evidence_version"], preprocessing.AMPLITUDE_EVIDENCE_VERSION)
            self.assertEqual(revised_c["scoring_spec"]["amplitude_evidence"], preprocessing.amplitude_spec())
            from pipe.tslm.model import AMPLITUDE_SCORING_VERSION
            self.assertEqual(revised_c["scoring_spec"]["version"], AMPLITUDE_SCORING_VERSION)
            self.assertEqual(config_c["amplitude_evidence_version"], preprocessing.AMPLITUDE_EVIDENCE_VERSION)
            self.assertNotEqual(revised_c["config_hash"], revised["config_hash"])
            self.assertNotEqual(result_c["model_version"], result["model_version"])
            self.assertEqual(result_c["temporal_sha256"], original["temporal.pt"])
            self.assertEqual(json.loads((root / "v2-c/scoring_spec.json").read_text()), revised_c["scoring_spec"])
            (root / "v2/base/model.safetensors").write_bytes(b"independent copy")
            for name, digest in original.items():
                self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), digest)
            for destination in (source, source / "nested", root / "v2"):
                with self.assertRaises(ValueError):
                    make_reference(source, destination, "b" * 40)

    def test_development_cache_requires_exact_timef_wav_parity_and_skips_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}")
            rows = [{"clip_id": f"c{i}", "group_id": f"g{i}", "fold": fold, "path": f"{i}.wav"}
                    for i, fold in enumerate(("train", "val", "test"))]
            md5_rows, records = [], []
            for i, row in enumerate(rows[:2]):
                x = np.random.default_rng(i).integers(-14000, 14000, 8000).astype("<i2")
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(8000)
                    writer.writeframes(x.tobytes())
                raw = buffer.getvalue()
                (root / row["path"]).write_bytes(raw)
                md5_rows.append(f"{row['clip_id']},{hashlib.md5(raw).hexdigest()}")
                values = prepare._normalise(x).astype(np.float32)
                records.append(SimpleNamespace(record_id=row["clip_id"], subject_ids=(row["group_id"],),
                    time_series=[SimpleNamespace(to_numpy=lambda values=values: values)]))
            (root / "split_v2_audit.csv").write_text("clip_id,md5\n" + "\n".join(md5_rows) + "\n")

            class Reader:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def iter_records(self, record_ids, with_annotations):
                    assert record_ids == ["c0", "c1"] and with_annotations is False
                    return iter(records)

            with patch.object(prepare, "load_manifest", return_value=rows), \
                 patch.object(prepare.DatasetVersion, "open_local"), \
                 patch.object(prepare, "TimeFReader", return_value=Reader()):
                report = prepare.prepare_development(root, root, root, root / "cache")
                self.assertTrue(report["canonical_vs_timef_exact"])
                self.assertEqual(report["fold_counts"], {"train": 1, "val": 1})
                self.assertTrue(report["amplitude_features_raw_vs_timef_exact"])
                self.assertTrue(report["amplitude_text_raw_vs_timef_exact"])
                self.assertEqual(report["amplitude_records_verified"], 2)
                self.assertEqual(report["amplitude_evidence_version"], preprocessing.AMPLITUDE_EVIDENCE_VERSION)
                self.assertEqual(report["source_sha256"]["features.py"], hashlib.sha256(
                    (ROOT / "scripts/eval/harness/features.py").read_bytes()).hexdigest())
                for index, fold in enumerate(("train", "val")):
                    with np.load(root / f"cache/{fold}.npz", allow_pickle=False) as cached:
                        self.assertEqual(cached["amplitude_features"].shape, (1, 9))
                        self.assertEqual(cached["amplitude_features"].dtype, np.dtype("float64"))
                        self.assertEqual(cached["amplitude_text"].dtype.kind, "U")
                        self.assertEqual(cached["amplitude_evidence_version"].item(),
                                         preprocessing.AMPLITUDE_EVIDENCE_VERSION)
                        self.assertEqual(cached["preprocessing_version"].item(), CANONICAL_VERSION)
                        waveform = prepare.decode_wav((root / rows[index]["path"]).read_bytes())
                        expected_features = preprocessing.amplitude_features(waveform, 8000)
                        np.testing.assert_array_equal(cached["amplitude_features"][0], expected_features)
                        self.assertEqual(cached["amplitude_text"][0], preprocessing.amplitude_text(expected_features))
                        np.testing.assert_array_equal(cached["series"][0],
                            preprocessing.preprocess_audio(waveform, 8000, version=CANONICAL_VERSION))
                self.assertFalse((root / "cache/test.npz").exists())
                self.assertFalse((root / "2.wav").exists())
                with self.assertRaises(FileExistsError):
                    prepare.prepare_development(root, root, root, root / "cache")
                with patch.object(preprocessing, "amplitude_features", return_value=np.ones(9)):
                    with self.assertRaises(AssertionError):
                        prepare.prepare_development(root, root, root, root / "bad-amplitude")
                self.assertFalse((root / "bad-amplitude/train.npz").exists())
                self.assertFalse((root / "bad-amplitude/preparation.json").exists())
                with patch.object(preprocessing, "amplitude_text", side_effect=["raw text", "different TimeF text"]):
                    with self.assertRaisesRegex(ValueError, "Texte des mesures"):
                        prepare.prepare_development(root, root, root, root / "bad-text")
                self.assertFalse((root / "bad-text/train.npz").exists())
                self.assertFalse((root / "bad-text/preparation.json").exists())
                records[0].time_series[0] = SimpleNamespace(to_numpy=lambda: np.zeros(8000, dtype=np.float32))
                with self.assertRaises(AssertionError):
                    prepare.prepare_development(root, root, root, root / "bad-cache")


if __name__ == "__main__":
    unittest.main()
