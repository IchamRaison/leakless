"""Tests synthétiques CPU : aucun WAV réel, aucun fit ou poids GPU."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import diagnose_train as diagnostic


def fixture_rows():
    return [{"clip_id": f"c{g:02d}-{i}", "group_id": f"g{g:02d}",
             "fold": "train", "label": "leak" if g >= 6 else "no_leak"}
            for g in range(12) for i in range(1 + g % 3)]


class TrainDiagnosticChecks(unittest.TestCase):
    def test_folds_are_fixed_group_stratified_and_train_only(self):
        rows = fixture_rows()
        folds = diagnostic.grouped_folds(rows)
        self.assertEqual(folds, diagnostic.grouped_folds(list(reversed(rows))))
        by_id = {r["clip_id"]: r for r in rows}
        heldout = []
        for fold in folds:
            fit = {by_id[cid]["group_id"] for cid in fold["train_ids"]}
            ho = {by_id[cid]["group_id"] for cid in fold["heldout_ids"]}
            self.assertFalse(fit & ho)
            self.assertEqual(len(ho), 4)
            self.assertEqual(len({by_id[cid]["group_id"] for cid in fold["heldout_ids"]
                                  if by_id[cid]["label"] == "leak"}), 2)
            heldout.extend(fold["heldout_ids"])
        self.assertEqual(sorted(heldout), sorted(by_id))
        for bad in ([{**rows[0], "fold": "val"}, *rows[1:]],
                    [{**rows[0], "fold": "test"}, *rows[1:]], [*rows, rows[0]],
                    [{**rows[0], "group_id": "g06"}, *rows[1:]], rows[:2]):
            with self.assertRaises(ValueError):
                diagnostic.grouped_folds(bad)

    def test_parity_pass_binds_cache_version_sources_and_all_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cache = path / "train.npz"
            cache.write_bytes(b"canonical train fixture")
            source = path / "model.py"
            source.write_bytes(b"source fixture")
            report = {"schema": "pipe-parity-v2", "all_checks_pass": True,
                "checks": dict.fromkeys(diagnostic.PARITY_CHECKS, True),
                "provenance": {"score_atol": 1e-6, "preprocessing_version": "canonical-fixture",
                    "manifest_sha256": diagnostic.split_loader.FROZEN_SPLIT_SHA256,
                    "cache_sha256": {"train": diagnostic.sha256_file(cache), "val": "b" * 64},
                    "timef_manifest_sha256": "c" * 64,
                    "source_sha256": {"model.py": diagnostic.sha256_file(source)}}}
            preparation = {"preprocessing_version": "canonical-fixture", "records": 806,
                "fold_counts": {"train": 598, "val": 208}, "canonical_vs_timef_exact": True,
                "test_audio_or_cache_opened": False,
                "manifest_sha256": diagnostic.split_loader.FROZEN_SPLIT_SHA256,
                "cache_sha256": report["provenance"]["cache_sha256"], "timef_manifest_sha256": "c" * 64,
                "source_sha256": {name: diagnostic.sha256_file(ROOT / "src/pipe/tslm" / name)
                                  for name in ("prepare.py", "preprocessing.py")}}
            (path / "preparation.json").write_text(json.dumps(preparation))
            receipt = path / "report.json"
            def verify(value):
                receipt.write_text(json.dumps(value))
                return diagnostic.verify_parity(receipt, path, "canonical-fixture",
                                                source_paths={"model.py": source})
            self.assertEqual(verify(report), report["provenance"])
            (path / "preparation.json").write_text(json.dumps({**preparation, "records": 209}))
            with self.assertRaises(ValueError):
                verify(report)
            (path / "preparation.json").write_text(json.dumps(preparation))
            for bad in ({**report, "all_checks_pass": False}, {**report, "all_checks_pass": 1},
                        {**report, "schema": "old"},
                        {**report, "checks": {**report["checks"], "fresh_process_verified": False}},
                        {**report, "provenance": {**report["provenance"], "score_atol": 1e-5}},
                        {**report, "provenance": {**report["provenance"], "preprocessing_version": "v1"}}):
                with self.assertRaises(ValueError):
                    verify(bad)
            cache.write_bytes(b"substituted")
            with self.assertRaises(ValueError):
                verify(report)
            cache.write_bytes(b"canonical train fixture")
            source.write_bytes(b"substituted")
            with self.assertRaises(ValueError):
                verify(report)
            self.assertFalse((path / "val.npz").exists())

    def test_probe_calls_six_fits_with_same_disjoint_partitions(self):
        rows = fixture_rows()
        folds = diagnostic.grouped_folds(rows)
        matrix = np.column_stack([np.arange(len(rows)), np.ones(len(rows))])
        calls = []
        def fake_fit(fit, labels, heldout):
            fit_ids, heldout_ids = set(fit[:, 0].astype(int)), set(heldout[:, 0].astype(int))
            self.assertFalse(fit_ids & heldout_ids)
            calls.append((fit_ids, heldout_ids))
            return np.array([0.8 if rows[i]["label"] == "leak" else 0.2
                             for i in heldout[:, 0].astype(int)])
        with patch.object(diagnostic, "fit_fixed_logreg", side_effect=fake_fit):
            report = diagnostic.run_probes(rows, {"TimeNet256": matrix, "C1_fixed": matrix}, folds)
        self.assertEqual(len(calls), 6)
        self.assertEqual(calls[:3], calls[3:])
        self.assertEqual(report["TimeNet256"]["mean_group_roc_auc"], 1.0)
        self.assertNotIn("pooled_roc_auc", report["TimeNet256"])
        self.assertTrue(all(f["threshold"] == 0.5 for f in report["C1_fixed"]["folds"]))
        bad = [{**folds[0], "heldout_ids": folds[0]["heldout_ids"] + [folds[0]["train_ids"][0]]}, *folds[1:]]
        with self.assertRaises(ValueError):
            diagnostic.run_probes(rows, {"bad": matrix}, bad)

    def test_token_boundaries_padding_and_eos_are_exact(self):
        ids = np.array([[1, 2, 9, 10, 0], [3, 4, 5, 9, 10]])
        attention = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
        classes, rest, counts = diagnostic.token_masks(ids, attention, [[1, 2], [3, 4, 5]], [10])
        self.assertEqual(int(classes.sum()), 5)
        self.assertEqual(int(rest.sum()), 4)
        np.testing.assert_array_equal(classes | rest, attention.astype(bool))
        self.assertFalse((classes & rest).any())
        self.assertEqual(counts[0], {"class_tokens": 2, "description_tokens": 1,
                                    "eos_tokens": 1, "padding_tokens": 1})
        for bad_ids, bad_attention, class_ids, eos in (
                (ids, attention, [[1], [3, 4, 5]], [0]),
                (ids, attention, [[1, 3], [3, 4, 5]], [10]),
                (ids, np.array([[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]), [[1, 2], [3, 4, 5]], [10]),
                (ids, attention, [[1, 2], [3, 4, 5]], [])):
            with self.assertRaises(ValueError):
                diagnostic.token_masks(bad_ids, bad_attention, class_ids, eos)

    def test_capture_keeps_real_output_and_restores_forward_after_exception(self):
        result = object()
        def forward(**kwargs):
            return result
        llm = SimpleNamespace(forward=forward)
        with diagnostic.capture_loss_forward(llm) as calls:
            self.assertIs(llm.forward(labels="real mask"), result)
            self.assertEqual(calls[0][0], {"labels": "real mask"})
            self.assertIs(calls[0][1], result)
        self.assertIs(llm.forward, forward)
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with diagnostic.capture_loss_forward(llm):
                raise RuntimeError("failure")
        self.assertIs(llm.forward, forward)

    def test_batch_audit_preserves_counts_and_declares_reference_budget(self):
        rows = fixture_rows()
        result = diagnostic.batch_composition(rows)
        self.assertEqual(result, diagnostic.batch_composition(rows))
        for epoch in range(1, 9):
            batches = [r for r in result["batches"] if r["epoch"] == epoch]
            self.assertEqual(sum(r["n_clips"] for r in batches), len(rows))
            self.assertEqual(sum(r["n_leak"] for r in batches), sum(r["label"] == "leak" for r in rows))


if __name__ == "__main__":
    unittest.main()
