"""Gel/lecture sur RAW et journaux synthétiques ; aucun modèle ni corpus réel."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import external_manifest as manifest


class ExternalManifestChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.external = self.root / "prepared"
        self.overlap = self.root / "overlap"
        self.overlap.mkdir()
        self.manifests = self.root / "old-manifests"
        self.manifests.mkdir()
        (self.manifests / "split_v2.csv").write_text("clip_id\nold-a\nold-b\n")
        (self.manifests / "split_v2_audit.csv").write_text("clip_id,path\nold-a,a.wav\nold-b,b.wav\n")
        self.expected = {
            "Hydrophone/leak.raw": {"subset": "primary", "label": "leak", "condition_key": "leak", "sensor": "H1"},
            "Hydrophone/normal.raw": {"subset": "primary", "label": "no_leak", "condition_key": "normal", "sensor": "H1"},
            "Hydrophone/noise.raw": {"subset": "background", "label": "environmental_noise", "condition_key": "noise", "sensor": "H1"}}
        archive, converter = self.root / "source.zip", self.root / "converter.py"
        converter.write_text("raise AssertionError('Do not execute source')\n")
        with zipfile.ZipFile(archive, "w") as handle:
            for i, member in enumerate(self.expected):
                handle.writestr(member, (np.arange(8, dtype="<i4") * (i+1) + 100*i).tobytes())
        patches = [patch.multiple(manifest.preparer,
            ARCHIVE_BYTES=archive.stat().st_size, ARCHIVE_SHA256=manifest.preparer.sha256_file(archive),
            CONVERTER_BYTES=converter.stat().st_size, CONVERTER_SHA256=manifest.preparer.sha256_file(converter),
            CROP_SAMPLES=8, WINDOW_SAMPLES=4, SAMPLE_RATE=4),
            patch.object(manifest.preparer, "expected_recordings", return_value=self.expected),
            patch.multiple(manifest.overlap, N_ORIGINAL=2, N_EXTERNAL=3, WINDOW=4, RECORD_SAMPLES=8,
                FROZEN_SPLIT_SHA256=manifest.preparer.sha256_file(self.manifests / "split_v2.csv"),
                FROZEN_AUDIT_SHA256=manifest.preparer.sha256_file(self.manifests / "split_v2_audit.csv"))]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        manifest.preparer.prepare(self.external, archive, converter, "a" * 40)
        sums = json.loads((self.external / "checksums.json").read_text())
        self.prep = {"checksums_sha256": manifest.preparer.sha256_file(self.external / "checksums.json"),
                     "metadata_sha256": sums["metadata.json"], "archive_sha256": manifest.preparer.ARCHIVE_SHA256}
        self.records = sorted(manifest.preparer.stable_id("recording", member) for member in self.expected)
        provenance = {"script_sha256": manifest.preparer.sha256_file(Path(manifest.overlap.__file__)),
            "external": self.prep, "split_sha256": manifest.overlap.FROZEN_SPLIT_SHA256,
            "audit_manifest_sha256": manifest.overlap.FROZEN_AUDIT_SHA256,
            "threshold": .995, "offset_range_inclusive": [0, 4], "sample_rate": 4,
            "labels_used": False, "model_loaded": False, "quality_metrics_computed": False,
            "original_files": [{"clip_id": cid} for cid in ("old-a", "old-b")],
            "helper_sha256": {name: manifest.preparer.sha256_file(ROOT / name) for name in manifest.HELPER_PATHS}}
        (self.overlap / "provenance.json").write_text(json.dumps(provenance))
        self.pairs = [{"clip_id": old, "recording_id": record, "correlation": .1,
                       "max_abs_correlation": .1, "offset_samples": 0, "unassessable_offsets": 0,
                       "reason": None, "suspect": False} for old in ("old-a", "old-b") for record in self.records]
        (self.overlap / "pairs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.pairs))
        (self.overlap / "suspects.jsonl").write_text("")
        (self.overlap / "exact_matches.json").write_text("[]")
        (self.overlap / "best_per_original.json").write_text(json.dumps({old: next(r for r in self.pairs if r["clip_id"] == old)
                                                                       for old in ("old-a", "old-b")}))
        self.summary = {"status": "complete", "threshold": .995, "pairs_examined": 6,
            "pairs_expected": 6, "offsets_per_pair": 5, "suspect_pairs": 0, "unassessable_pairs": 0,
            "pairs_with_unassessable_offsets": 0, "unassessable_offsets_total": 0,
            "exact_aligned_matches": 0, "automatic_exclusions": False}
        self.refresh_overlap_hashes()

    def refresh_overlap_hashes(self):
        self.summary["files_sha256"] = {name: manifest.preparer.sha256_file(self.overlap / name)
                                         for name in manifest.OVERLAP_FILES}
        (self.overlap / "summary.json").write_text(json.dumps(self.summary))

    def refresh_preparation_hashes(self, name):
        path = self.external / "checksums.json"
        sums = json.loads(path.read_text())
        sums[name] = manifest.preparer.sha256_file(self.external / name)
        path.write_text(json.dumps(sums))
        receipt = json.loads((self.external / "receipt.json").read_text())
        receipt["checksums_sha256"] = manifest.preparer.sha256_file(path)
        (self.external / "receipt.json").write_text(json.dumps(receipt))

    def freeze(self):
        return manifest.freeze_external_manifest(self.external, self.overlap, self.manifests,
                                                  self.root / manifest.NAME, "a" * 40)

    def test_full_synthetic_raw_verification_freezes_primary_only_portable_split(self):
        split = self.freeze()
        self.assertEqual(len(split.clips), 4)
        self.assertEqual({c.fold for c in split.clips}, {"external"})
        self.assertEqual({c.label for c in split.clips}, {0, 1})
        self.assertEqual(len({c.group_id for c in split.clips}), 2)
        self.assertEqual(split._paths, {})
        self.assertEqual(split.sha256, manifest.preparer.sha256_file(split.manifest_path))
        frozen = json.loads(split.manifest_path.read_text())
        self.assertEqual(len(frozen["background_annex"]["targets"]), 2)
        self.assertFalse(frozen["background_annex"]["included_in_primary"])
        with patch.object(manifest.overlap, "load_external", side_effect=AssertionError("No arrays in evaluator reader")):
            self.assertEqual(manifest.load_external_manifest(split.manifest_path), split)
        with self.assertRaises(FileExistsError):
            self.freeze()

    def test_wrong_name_existing_and_symlink_refused_before_sources(self):
        with patch.object(manifest, "validate_preparation", side_effect=AssertionError("No read")):
            with self.assertRaises(ValueError):
                manifest.freeze_external_manifest(self.external, self.overlap, self.manifests,
                                                   self.root / "split_v2.csv", "a" * 40)
            (self.root / manifest.NAME).symlink_to(self.root / "absent")
            with self.assertRaises(FileExistsError):
                self.freeze()

    def test_modified_targets_are_rejected_even_if_self_declared_hashes_are_refreshed(self):
        for name, field, value in (("targets.csv", "label", "no_leak"),
                                   ("targets.csv", "condition_group_id", "wrong"),
                                   ("background_targets.csv", "label", "leak")):
            path = self.external / name
            original = path.read_text()
            rows = list(csv.DictReader(original.splitlines()))
            row = next(r for r in rows if r[field] != value)
            row[field] = value
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=manifest.preparer.TARGET_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            self.refresh_preparation_hashes(name)
            with self.assertRaisesRegex(ValueError, "inventaire officiel"):
                manifest.validate_preparation(self.external)
            path.write_text(original)
            self.refresh_preparation_hashes(name)

    def test_wrong_columns_duplicate_ids_and_hash_only_changes_fail(self):
        path = self.external / "targets.csv"
        original = path.read_text()
        path.write_text(original + original.splitlines()[1] + "\n")
        with self.assertRaisesRegex(ValueError, "SHA"):
            manifest.validate_preparation(self.external)
        self.refresh_preparation_hashes("targets.csv")
        with self.assertRaisesRegex(ValueError, "répété"):
            manifest.validate_preparation(self.external)
        path.write_text(original.replace("condition_group_id", "metadata_for_model", 1))
        self.refresh_preparation_hashes("targets.csv")
        with self.assertRaisesRegex(ValueError, "Colonnes"):
            manifest.validate_preparation(self.external)

    def test_overlap_incomplete_duplicated_or_suspect_never_allows_freeze(self):
        for rows in (self.pairs[:-1], [*self.pairs[:-1], self.pairs[0]],
                     [{**self.pairs[0], "correlation": .999, "max_abs_correlation": .999}, *self.pairs[1:]]):
            (self.overlap / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.refresh_overlap_hashes()
            with self.assertRaises(ValueError):
                manifest.validate_overlap(self.overlap, self.prep, self.records, self.manifests)
        self.summary["pairs_examined"] = 5
        self.refresh_overlap_hashes()
        with self.assertRaisesRegex(ValueError, "incomplet"):
            self.freeze()
        self.assertFalse((self.root / manifest.NAME).exists())

    def test_reader_rejects_changed_labels_background_or_provenance(self):
        split = self.freeze()
        content = split.manifest_path.read_text()
        for modify in (lambda d: d["primary_targets"][0].update(label="wrong"),
                       lambda d: d["background_annex"].update(included_in_primary=True),
                       lambda d: d["preparation"]["csv_sha256"].update({"targets.csv": "absent"}),
                       lambda d: d["overlap"].update(threshold=.99)):
            data = json.loads(content)
            modify(data)
            split.manifest_path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                manifest.load_external_manifest(split.manifest_path)


if __name__ == "__main__":
    unittest.main()
