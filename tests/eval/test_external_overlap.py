"""Audit de données : tests synthétiques, aucune qualité de modèle calculée."""
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
sys.path.insert(0, str(ROOT / "scripts/eval"))
import audit_external_overlap as overlap  # noqa: E402


class ExternalOverlapChecks(unittest.TestCase):
    def test_fft_agrees_with_exhaustive_direct_correlation(self):
        rng = np.random.default_rng(31)
        queries, record = rng.normal(size=(3, 7)), rng.normal(size=41)
        prepared = overlap.query_spectra(queries, len(record))
        results = overlap.sliding_correlations(queries, record, prepared)
        for query, result in zip(queries, results):
            direct = [overlap.direct_correlation(query, record[start:start + len(query)])
                      for start in range(len(record) - len(query) + 1)]
            index = int(np.argmax(np.abs(direct)))
            self.assertEqual(result["offset_samples"], index)
            self.assertAlmostEqual(result["correlation"], direct[index], places=12)
        self.assertEqual(results, overlap.sliding_correlations(queries, record, workers=2))

    def test_pcm16_to_pcm32_quasicopy_gain_dc_polarity_arbitrary_offset(self):
        rng = np.random.default_rng(20260912)
        integers = rng.integers(-8000, 8001, 8000, dtype=np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(8000)
            writer.writeframes(integers.astype("<i2").tobytes())
        query = overlap.decode_pcm16(buffer.getvalue())
        record = rng.uniform(-.05, .05, 240000)
        start = 100337  # Ni une seconde ni une limite de bloc.
        record[start:start + 8000] = -.17 + (-1.6 * query)
        pcm32 = np.rint(record * 2**31).astype("<i4")
        decoded = pcm32.astype(np.float64) / 2**31
        other = rng.uniform(-.2, .2, 8000)
        copied, unrelated = overlap.sliding_correlations(np.stack([query, other]), decoded)
        self.assertEqual(copied["offset_samples"], start)
        self.assertGreater(copied["max_abs_correlation"], .999999)
        self.assertLess(copied["correlation"], -.999999)
        self.assertLess(unrelated["max_abs_correlation"], overlap.SUSPECT_CORRELATION)
        self.assertNotEqual(overlap.signal_hash(query), overlap.signal_hash(decoded[start:start + 8000]))

    def test_constant_query_and_record_are_unassessable_not_zero_similarity(self):
        rng = np.random.default_rng(8)
        query = rng.normal(size=12)
        result = overlap.sliding_correlations(query[None, :], np.ones(30))[0]
        self.assertIsNone(result["correlation"])
        self.assertEqual(result["unassessable_offsets"], 19)
        result = overlap.sliding_correlations(np.ones((1, 12)), rng.normal(size=30))[0]
        self.assertIsNone(result["max_abs_correlation"])
        self.assertEqual(result["reason"], "constant_query")
        record = np.full(200, 3.)
        record[91:103] = query + 3
        result = overlap.sliding_correlations(query[None, :], record)[0]
        self.assertEqual(result["offset_samples"], 91)
        self.assertGreater(result["unassessable_offsets"], 0)
        self.assertAlmostEqual(result["max_abs_correlation"], 1)

    def test_bounds_paths_and_old_output_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "inside").write_text("safe")
            self.assertEqual(overlap.safe_path(root, "inside"), root / "inside")
            for name in ("../bad", "/absolute", "C:bad", "a\\b", "missing"):
                with self.assertRaises(ValueError):
                    overlap.safe_path(root, name)
            args = SimpleNamespace(code_revision="a" * 40, batch_size=8, workers=1, output=root)
            with patch.object(overlap, "load_original", side_effect=AssertionError("No source read")):
                with self.assertRaises(FileExistsError):
                    overlap.audit(args)
        with self.assertRaises(ValueError):
            overlap.query_spectra(np.full((1, 8), np.nan), 40)

    def test_changed_external_receipt_rejected_before_loading_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "receipt.json").write_text(json.dumps({"status": "complete",
                "source_integrity_verified": True, "checksums_sha256": "0" * 64}))
            (root / "checksums.json").write_text("{}")
            with patch.object(np, "load", side_effect=AssertionError("No arrays read")):
                with self.assertRaises(ValueError):
                    overlap.load_external(root)

    def test_synthetic_audit_reports_pairs_suspects_and_checksums_without_quality(self):
        rng = np.random.default_rng(55)
        queries = rng.normal(size=(2, 8))
        record = rng.normal(size=35)
        record[17:25] = -2 * queries[0] + 4
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "record.npy", record)
            args = SimpleNamespace(code_revision="a" * 40, batch_size=2, workers=1,
                                   output=root / "audit", manifests=root, data_root=root, external=root)
            with patch.object(overlap, "WINDOW", 8), patch.object(overlap, "RECORD_SAMPLES", 35), \
                    patch.object(overlap, "load_original", return_value=(["a", "b"], queries, [])), \
                    patch.object(overlap, "load_external", return_value=([("new", root / "record.npy")], {})), \
                    patch("builtins.print"):
                overlap.audit(args)
            summary = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(summary["pairs_examined"], 2)
            self.assertEqual(summary["pairs_expected"], 2)
            self.assertEqual(summary["offsets_per_pair"], 28)
            self.assertEqual(summary["suspect_pairs"], 1)
            self.assertFalse(summary["automatic_exclusions"])
            suspects = [json.loads(line) for line in (args.output / "suspects.jsonl").read_text().splitlines()]
            self.assertEqual(suspects[0]["clip_id"], "a")
            self.assertEqual(suspects[0]["offset_samples"], 17)
            provenance = json.loads((args.output / "provenance.json").read_text())
            self.assertFalse(provenance["quality_metrics_computed"])
            for relative, digest in summary["files_sha256"].items():
                self.assertEqual(overlap.external.sha256_file(args.output / relative), digest)


if __name__ == "__main__":
    unittest.main()
