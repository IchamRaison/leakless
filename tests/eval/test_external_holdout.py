"""Tests CPU synthétiques : aucun modèle ni signal réel du holdout."""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import prepare_external_holdout as external  # noqa: E402


class ExternalHoldoutChecks(unittest.TestCase):
    def test_official_filename_grammar_and_dependency_groups(self):
        rows = external.expected_recordings()
        primary = [row for row in rows.values() if row["subset"] == "primary"]
        self.assertEqual(len(rows), 122)
        self.assertEqual(collections.Counter(row["label"] for row in primary),
                         {"leak": 96, "no_leak": 24})
        groups = collections.defaultdict(set)
        for row in primary:
            groups[row["condition_key"]].add(row["sensor"])
        self.assertEqual(len(groups), 60)
        self.assertTrue(all(sensors == {"H1", "H2"} for sensors in groups.values()))
        self.assertIn("Hydrophone/Branched/No-leak/BR_NL_0.18 LPS_N_H1.raw", rows)

    def test_pcm32_conversion_exact_without_clipping_or_pcm16_rounding(self):
        integers = np.array([-2147483648, -1, 0, 1, 65537, 2147483647, 123], dtype="<i4")
        with patch.object(external, "CROP_SAMPLES", 6):
            signal = external.decode_raw(integers.tobytes())
            np.testing.assert_array_equal(signal, integers[:6].astype(np.float64) / 2**31)
            self.assertEqual(signal.dtype, np.dtype("<f8"))
            self.assertEqual(signal[3], 2**-31)
            self.assertEqual(signal[5], 1 - 2**-31)
            with self.assertRaises(ValueError):
                external.decode_raw(integers[:5].tobytes())
            with self.assertRaises(ValueError):
                external.decode_raw(integers.tobytes() + b"x")

    @staticmethod
    def toy_zip(entries):
        buffer = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(buffer, "w") as archive:
                for name in entries:
                    archive.writestr(name, b"\x01\x00\x00\x00" * 2)
        buffer.seek(0)
        return zipfile.ZipFile(buffer)

    def test_zip_traversal_duplicate_unexpected_and_missing_rejected(self):
        safe = "Hydrophone/example.raw"
        with patch.object(external, "CROP_SAMPLES", 2):
            for names in ([safe, safe], ["../bad.raw"], ["/absolute.raw"],
                          ["Hydrophone\\bad.raw"], ["C:bad.raw"], ["unexpected.raw"], []):
                with self.subTest(names=names), self.toy_zip(names) as archive:
                    with self.assertRaises(ValueError):
                        external.validate_members(archive, {safe: {}})
            with self.toy_zip([safe]) as archive:
                external.validate_members(archive, {safe: {}})

    def test_zip_symlink_rejected(self):
        info = zipfile.ZipInfo("Hydrophone/example.raw")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with patch.object(external, "CROP_SAMPLES", 2), self.toy_zip([info]) as archive:
            with self.assertRaises(ValueError):
                external.validate_members(archive, {info.filename: {}})

    def test_existing_incomplete_or_symlink_output_refused_before_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "existing").mkdir()
            (root / "broken").symlink_to(root / "missing")
            with patch.object(external, "download", side_effect=AssertionError("No network")):
                for name in ("existing", "broken"):
                    with self.assertRaises(FileExistsError):
                        external.prepare(root / name, None, None, "a" * 40)

    def test_revision_must_be_explicit_full_sha(self):
        with self.assertRaises(ValueError):
            external.prepare(Path("unused"), None, None, "unknown")

    def test_synthetic_preparation_separates_labels_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "toy.zip"
            converter = root / "converter.py"
            converter.write_bytes(b"raise AssertionError('must never execute source code')\n")
            expected = {
                "Hydrophone/label_leak.raw": {"subset": "primary", "label": "leak",
                                              "condition_key": "leak_condition", "sensor": "H1"},
                "Hydrophone/background.raw": {"subset": "background", "label": "environmental_noise",
                                               "condition_key": "background", "sensor": "H1"},
            }
            with zipfile.ZipFile(source, "w") as archive:
                for index, member in enumerate(expected):
                    values = np.arange(16004, dtype="<i4") + index * 20000
                    archive.writestr(member, values.tobytes())
            archive_sha = external.sha256_file(source)
            constants = {
                "ARCHIVE_BYTES": source.stat().st_size, "ARCHIVE_SHA256": archive_sha,
                "CONVERTER_BYTES": converter.stat().st_size,
                "CONVERTER_SHA256": external.sha256_file(converter), "CROP_SAMPLES": 16000,
            }
            with patch.multiple(external, **constants), \
                    patch.object(external, "expected_recordings", return_value=expected), \
                    patch.object(external, "download", side_effect=AssertionError("No network")):
                metadata = external.prepare(root / "first", source, converter, "a" * 40)
                external.prepare(root / "second", source, converter, "a" * 40)
                with patch.object(external, "ARCHIVE_SHA256", "0" * 64):
                    with self.assertRaises(ValueError):
                        external.prepare(root / "invalid", source, converter, "a" * 40)
                self.assertFalse((root / "invalid").exists())
            self.assertFalse(metadata["model_scoring_performed"])
            self.assertFalse(metadata["training_permitted"])
            self.assertEqual(metadata["subsets"]["primary"]["windows"], 2)
            self.assertEqual(metadata["subsets"]["background"]["windows"], 2)
            first = root / "first"
            with (first / "inputs.csv").open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, external.INPUT_FIELDS)
                rows = list(reader)
            self.assertEqual([int(row["start_sample"]) for row in rows], [0, 8000])
            self.assertNotIn("leak", (first / "inputs.csv").read_text())
            signal = np.load(first / rows[0]["array_path"], allow_pickle=False, mmap_mode="r")
            np.testing.assert_array_equal(signal, np.arange(16000, dtype=np.float64) / 2**31)
            targets = list(csv.DictReader((first / "targets.csv").read_text().splitlines()))
            self.assertEqual([row["clip_id"] for row in targets], [row["clip_id"] for row in rows])
            self.assertTrue(all(row["label"] == "leak" for row in targets))
            checksums = json.loads((first / "checksums.json").read_text())
            for relative, digest in checksums.items():
                self.assertEqual(external.sha256_file(first / relative), digest)
                self.assertEqual((first / relative).read_bytes(), (root / "second" / relative).read_bytes())
            receipt = json.loads((first / "receipt.json").read_text())
            self.assertEqual(receipt["checksums_sha256"], external.sha256_file(first / "checksums.json"))
            self.assertEqual(receipt["status"], "complete")
            self.assertFalse(receipt["source_code_executed"])
            self.assertEqual(external.sha256_file(first / "sources/Hydrophone.zip"), archive_sha)


if __name__ == "__main__":
    unittest.main()
