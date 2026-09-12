"""Tests CPU du gate, sans modèle ni métrique sur des données réelles."""
import copy
import importlib.util
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
import check_v2_parity as gate


def fixture_report(pid=1):
    scores = {name: 0.4 for name in gate.ADAPTERS}
    scores.update({f"batch_{size}_{order}": 0.4 for size in gate.BATCH_SIZES
                   for order in ("forward", "reversed")})
    provenance = {key: "same" for key in (
        "preprocessing_version", "historical_preprocessing_version", "manifest_sha256",
        "cache_sha256", "historical_cache_sha256", "checkpoint_checksums_sha256", "temporal_sha256",
        "timef_manifest_sha256", "source_sha256", "runtime", "scoring_spec", "state_sha256")}
    provenance.update(process={"hostname": "same-host", "pid": pid}, clip_ids=["val"],
                      seed=gate.SEED, score_atol=gate.SCORE_ATOL, threshold_diagnostic_only=None)
    return {"schema": gate.SCHEMA, "all_checks_pass": False,
            "checks": {key: key != "fresh_process_verified" for key in gate.CHECKS},
            "provenance": provenance, "records": [{"clip_id": "val", "exact": True}],
            "scores": {"val": scores}}


class V2ParityGateChecks(unittest.TestCase):
    def test_canonical_transform_matches_timef_and_refuses_legacy_bundle(self):
        def normalise(x):
            x = np.asarray(x, dtype=np.float64)
            centered = x - x.mean()
            return centered / np.sqrt(np.mean(centered ** 2))
        spec = importlib.util.spec_from_file_location("v2_gate_preprocessing", ROOT / "src/pipe/tslm/preprocessing.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"leakless_acoustic.connector": SimpleNamespace(_normalise=normalise)}):
            spec.loader.exec_module(module)
        waveform = np.random.default_rng(19).integers(-14000, 14000, 8000).astype(np.float64)
        timef = normalise(waveform).astype(np.float32)
        canonical = module.preprocess_audio(waveform, 8000, version=module.CANONICAL_VERSION)
        routes = {"canonical": canonical, "timef": module.band_series(timef), "cache": canonical.copy()}
        self.assertTrue(gate.compare_routes(routes)["exact"])
        routes["legacy"] = module.preprocess_audio(waveform, 8000)
        self.assertFalse(gate.compare_routes(routes)["exact"])
        self.assertFalse(gate.exact_arrays(canonical, canonical.astype(np.float64)))
        gate.require_canonical({"preprocessing_version": module.CANONICAL_VERSION}, module.CANONICAL_VERSION)
        for metadata in ({}, {"preprocessing_version": module.VERSION}):
            with self.assertRaisesRegex(ValueError, "legacy"):
                gate.require_canonical(metadata, module.CANONICAL_VERSION)

    def test_real_batch_sizes_and_reverse_preserve_ids_without_metadata(self):
        ids = [f"clip{i}" for i in range(9)]
        series = {cid: np.full((4, 64), i, dtype=np.float32) for i, cid in enumerate(ids)}
        calls = []
        def score_batch(arrays):
            self.assertTrue(all(isinstance(a, np.ndarray) for a in arrays))
            calls.append([float(a[0, 0]) for a in arrays])
            return [float(a[0, 0]) / 10 for a in arrays]
        results = gate.score_batches(ids, series, score_batch)
        self.assertEqual(len(results), 6)
        self.assertEqual({len(chunk) for chunk in calls}, {1, 2, 4})
        self.assertIn([8, 7, 6, 5], calls)
        for result in results.values():
            self.assertEqual(result, {cid: i / 10 for i, cid in enumerate(ids)})
        with self.assertRaises(ValueError):
            gate.score_batches(ids, series, lambda arrays: [0.5])
        with self.assertRaises(ValueError):
            gate.score_batches([*ids, ids[0]], series, score_batch)
        with self.assertRaises(ValueError):
            gate.score_batches(ids, series, lambda arrays: [float("nan")] * len(arrays))

    def test_fixed_tolerance_pairwise_and_threshold_not_chosen(self):
        report = fixture_report()
        scores = report["scores"]
        scores["val"] = {name: 0.5 for name in scores["val"]}
        scores["val"]["wav_bytes"] = 0.5 - 0.4e-6
        scores["val"]["batch_4_reversed"] = 0.5 + 0.4e-6
        summary = gate.summarize_scores(scores, threshold=0.5)
        self.assertTrue(summary["batch_scores_within_atol"])
        self.assertEqual(summary["threshold_diagnostic"]["near_threshold_ids"], ["val"])
        self.assertFalse(summary["threshold_diagnostic"]["selected_here"])
        self.assertEqual(len(summary["threshold_diagnostic"]["decision_flips"]), 1)
        scores["val"]["batch_4_reversed"] = 0.5 + 0.8e-6
        self.assertFalse(gate.summarize_scores(scores)["batch_scores_within_atol"])
        del scores["val"]["waveform"]
        with self.assertRaises(ValueError):
            gate.summarize_scores(scores)

    def test_fresh_reference_identity_coverage_and_all_path_spread(self):
        first, second = fixture_report(1), fixture_report(2)
        self.assertTrue(gate.verify_reference(second, first)["all_scores_within_atol"])
        with self.assertRaisesRegex(ValueError, "processus"):
            gate.verify_reference(first, first)
        for key in ("source_sha256", "runtime", "state_sha256", "cache_sha256"):
            changed = copy.deepcopy(second)
            changed["provenance"][key] = "changed"
            with self.assertRaisesRegex(ValueError, key):
                gate.verify_reference(changed, first)
        changed = copy.deepcopy(first)
        changed["checks"]["batch_scores_within_atol"] = False
        with self.assertRaises(ValueError):
            gate.verify_reference(second, changed)
        changed = copy.deepcopy(second)
        changed["records"].append(changed["records"][0])
        with self.assertRaises(ValueError):
            gate.verify_reference(changed, first)
        changed = copy.deepcopy(second)
        changed["records"][0]["series"] = "changed"
        with self.assertRaises(ValueError):
            gate.verify_reference(changed, first)
        second["scores"]["val"]["series"] = 0.400002
        self.assertFalse(gate.verify_reference(second, first)["all_scores_within_atol"])

    def test_main_first_receipt_is_not_gate_and_second_process_can_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = ["--checkpoint", "unused", "--prepared", "unused", "--historical-prepared", "unused",
                      "--timef-version", "unused", "--data-root", "unused"]
            first = fixture_report(1)
            with patch.object(gate, "prepare_inputs", return_value=({}, first["records"], first["provenance"])), \
                    patch.object(gate, "run_scores", return_value=(first["scores"], True)):
                self.assertEqual(gate.main([*common, "--output", str(root / "first")]), 0)
            receipt = json.loads((root / "first/report.json").read_text())
            self.assertFalse(receipt["all_checks_pass"])
            self.assertEqual(receipt["status"], "awaiting_fresh_process")
            second = fixture_report(2)
            with patch.object(gate, "prepare_inputs", return_value=({}, second["records"], second["provenance"])), \
                    patch.object(gate, "run_scores", return_value=(second["scores"], True)):
                self.assertEqual(gate.main([*common, "--output", str(root / "second"),
                                            "--reference", str(root / "first/report.json")]), 0)
            receipt = json.loads((root / "second/report.json").read_text())
            self.assertTrue(receipt["all_checks_pass"])
            with self.assertRaises(FileExistsError):
                gate.main([*common, "--output", str(root / "first")])

    def test_input_divergence_stops_before_loading_model_and_keeps_failure(self):
        fixture = fixture_report()
        fixture["records"][0]["exact"] = False
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(gate, "prepare_inputs", return_value=({}, fixture["records"], fixture["provenance"])), \
                patch.object(gate, "run_scores") as scorer:
            output = Path(tmp) / "failed"
            code = gate.main(["--checkpoint", "unused", "--prepared", "unused", "--historical-prepared", "unused",
                              "--timef-version", "unused", "--data-root", "unused", "--output", str(output)])
            self.assertEqual(code, 1)
            scorer.assert_not_called()
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["all_checks_pass"])
            self.assertEqual(report["error"]["stage"], "prepare_inputs")


if __name__ == "__main__":
    unittest.main()
