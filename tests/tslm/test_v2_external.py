"""Garde-fous du producteur externe sur signaux synthétiques, sans fit/GPU."""
import copy
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
import export_v2_external as exporter
import check_v2_parity as gate


class ExternalExportChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        # Valeurs plus fines que PCM16 : toute requantification serait visible.
        self.waveform = np.sin(np.arange(16000, dtype=np.float64) / 47) / 100000 + 1e-10
        self.arrays = {"arrays/fixture.npy": self.waveform}
        self.rows = [{"clip_id": f"clip-{i}", "array_path": "arrays/fixture.npy",
            "start_sample": str(i * 8000), "n_samples": "8000", "sample_rate": "8000"} for i in range(2)]

    def run_windows(self, callback, transform="T0", text=False, suffix=""):
        path = self.directory / f"raw{transform}{suffix}.jsonl"
        scores, summary = exporter.score_windows(self.rows, self.arrays, transform, callback, path, text_audit=text)
        return scores, summary, [json.loads(line) for line in path.read_text().splitlines()]

    def test_all_transforms_exact_float64_only_numeric_callback_and_no_mutation(self):
        original = self.waveform.copy()
        for transform in exporter.TRANSFORMS:
            received = []
            def callback(waveform, rate):
                self.assertEqual(rate, 8000)
                self.assertEqual(waveform.dtype, np.dtype("<f8"))
                received.append(waveform.copy())
                return 0.12345678901234567
            scores, summary, records = self.run_windows(callback, transform)
            self.assertEqual(set(scores), {"clip-0", "clip-1"})
            self.assertEqual(summary["successful"], 2)
            self.assertEqual(summary["errors"], 0)
            for i, value in enumerate(received):
                expected = exporter.TRANSFORMS[transform]["fn"](original[i*8000:(i+1)*8000],
                    exporter.clip_rng(transform, f"clip-{i}"))
                np.testing.assert_array_equal(value, expected)
                self.assertFalse(records[i]["requantized"])
            np.testing.assert_array_equal(original, self.waveform)
        self.assertEqual(exporter.BLOCK_SAMPLES, 250)

    def test_text_payload_is_preserved_and_score_reused_without_second_call(self):
        raw = "no_leak; Greatest mean spectral energy: 1000-2000 Hz.\n"
        payload = {"probability_leak": 0.712345678901, "prediction": "leak", "threshold": .61,
            "description": "Greatest mean spectral energy: 0-1000 Hz.",
            "fallback_used": True, "fallback_reasons": ["generated_band_disagrees_with_dsp"],
            "audit": {"raw_text_exact": raw, "raw_generation_returns": [[raw]], "raw_format_valid": True,
                      "raw_band_matches_dsp": False, "raw_class_matches_decision": False}}
        calls = []
        def callback(waveform, rate):
            calls.append((waveform.shape, rate))
            return copy.deepcopy(payload)
        scores, summary, records = self.run_windows(callback, text=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(scores["clip-0"], payload["probability_leak"])
        self.assertEqual(records[0]["payload"], payload)
        self.assertEqual(summary["fallback_used"], 2)
        self.assertEqual(summary["raw_format_valid"], 2)
        self.assertEqual(summary["raw_band_matches_dsp"], 0)
        self.assertEqual(summary["raw_class_matches_decision"], 0)

    def test_nonfinite_and_model_error_are_preserved_without_false_score(self):
        attempts = []
        def callback(waveform, rate):
            attempts.append(1)
            if len(attempts) == 1:
                return float("nan")
            error = RuntimeError("GPU indisponible")
            error.coherent_audit = {"raw_generation_returns": [], "raw_error": "OOM"}
            raise error
        scores, summary, records = self.run_windows(callback)
        self.assertEqual(scores, {})
        self.assertEqual(summary["errors"], 2)
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(records[0]["invalid_payload_repr"], "nan")
        self.assertIsNone(records[0]["payload"])
        self.assertEqual(records[1]["error"]["coherent_audit"]["raw_error"], "OOM")
        self.assertNotIn("prediction", records[1])

    def test_incompatible_dtype_and_duplicate_ids_fail_without_calling_model(self):
        def callback(*args):
            self.fail("Le modèle ne doit pas être appelé")
        self.arrays["arrays/fixture.npy"] = self.waveform.astype(np.float32)
        scores, summary, records = self.run_windows(callback)
        self.assertEqual(scores, {})
        self.assertEqual(summary["errors"], 2)
        self.rows[1]["clip_id"] = "clip-0"
        with self.assertRaisesRegex(ValueError, "répété"):
            self.run_windows(callback, suffix="duplicate")

    def gate_fixture(self):
        ids = ["fixture-val", "fixture-train"]
        identity = {"checkpoint_checksums_sha256": "a" * 64, "temporal_sha256": "b" * 64,
                    "preprocessing_version": "canonical-fixture"}
        metadata = {"scoring_spec": {"version": "C-fixture"}, "amplitude_evidence": True,
                    "amplitude_evidence_version": "amp-fixture"}
        provenance = {**identity, **metadata, "clip_ids": ids, "seed": exporter.STRESS_SEED,
            "score_atol": 1e-6, "manifest_sha256": exporter.split_loader.FROZEN_SPLIT_SHA256,
            "test_audio_or_cache_opened": False, "quality_metrics_calculated": False,
            "threshold_diagnostic_only": .600000123456789, "historical_preprocessing_version": "legacy",
            "cache_sha256": {}, "historical_cache_sha256": {}, "timef_manifest_sha256": "d" * 64,
            "runtime": {}, "state_sha256": {"encoder": "e" * 64}, "amplitude_tokens": {},
            "process": {"hostname": "fixture", "pid": 2},
            "source_sha256": {Path(name).name: value for name, value in exporter.source_hashes().items()}}
        names = list(gate.ADAPTERS) + [f"batch_{size}_{order}" for size in gate.BATCH_SIZES
                                      for order in ("forward", "reversed")]
        report = {"schema": gate.SCHEMA, "all_checks_pass": True, "status": "passed",
            "checks": {key: True for key in gate.CHECKS}, "provenance": provenance,
            "records": [{"clip_id": cid, "exact": True} for cid in ids],
            "scores": {cid: {name: .7 for name in names} for cid in ids}, "reference_sha256": "f" * 64}
        reference = copy.deepcopy(report)
        reference["provenance"]["process"]["pid"] = 1
        report["score_summary"] = gate.summarize_scores(report["scores"], provenance["threshold_diagnostic_only"])
        report["fresh_process"] = gate.verify_reference(report, reference)
        return report, reference, identity, metadata, provenance["threshold_diagnostic_only"], ids

    def check_gate(self, fixture):
        report, reference, identity, metadata, threshold, ids = fixture
        return exporter.validate_final_gate(report, reference, "f" * 64, identity, metadata, threshold, ids)

    def test_gate_requires_final_identity_exact_threshold_and_fresh_proof(self):
        fixture = self.gate_fixture()
        self.check_gate(fixture)
        for field, value in (("checkpoint_checksums_sha256", "0" * 64),
                             ("threshold_diagnostic_only", .5), ("test_audio_or_cache_opened", True)):
            changed = copy.deepcopy(fixture)
            changed[0]["provenance"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check_gate(changed)
        changed = copy.deepcopy(fixture)
        changed[0]["checks"]["fresh_process_verified"] = False
        with self.assertRaises(ValueError):
            self.check_gate(changed)

    def test_gate_rejects_decision_flip_even_below_numeric_tolerance(self):
        fixture = self.gate_fixture()
        report, reference, _, _, threshold, ids = fixture
        for candidate in (report, reference):
            candidate["scores"][ids[0]] = {name: threshold-1e-8 for name in candidate["scores"][ids[0]]}
        report["scores"][ids[0]]["series"] = threshold+1e-8
        report["score_summary"] = gate.summarize_scores(report["scores"], threshold)
        report["fresh_process"] = gate.verify_reference(report, reference)
        self.assertTrue(report["score_summary"]["batch_scores_within_atol"])
        with self.assertRaisesRegex(ValueError, "seuil plein"):
            self.check_gate(fixture)
        # Flip uniquement entre processus : chaque passage a ses propres décisions cohérentes.
        report["scores"][ids[0]] = {name: threshold+1e-8 for name in report["scores"][ids[0]]}
        report["score_summary"] = gate.summarize_scores(report["scores"], threshold)
        report["fresh_process"] = gate.verify_reference(report, reference)
        self.assertEqual(report["score_summary"]["threshold_diagnostic"]["decision_flips"], [])
        with self.assertRaisesRegex(ValueError, "seuil plein"):
            self.check_gate(fixture)

    def test_threshold_full_precision_and_finite_outside_bounds_never_clipped(self):
        path = self.directory / "c1-predictions.csv"
        path.write_text("clip_id,probability_leak\nfixture,.7\n")
        receipt = {"schema": "pipe-v2-c1-threshold-v1", "checkpoint_sha256": "a" * 64,
            "fit_fold": "val", "threshold": -1e-9, "threshold_repr": repr(-1e-9), "rule": "fixture-rule",
            "split_sha256": exporter.split_loader.FROZEN_SPLIT_SHA256,
            "validation_predictions_sha256": exporter.sha256_file(path),
            "threshold_method_sha256": exporter.sha256_file(ROOT / "scripts/eval/harness/metrics.py")}
        complete = {"thresholds": {"c1": -1e-9}}
        self.assertEqual(exporter.validate_threshold(receipt, "c1", self.directory, complete, "fixture-rule"), -1e-9)
        for key, value in (("threshold_repr", "0.000000"), ("fit_fold", "external"),
                           ("validation_predictions_sha256", "0" * 64)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                exporter.validate_threshold({**receipt, key: value}, "c1", self.directory, complete, "fixture-rule")

    def test_output_and_manifest_guard_precede_model_loading(self):
        args = SimpleNamespace(output=self.directory)
        with patch.object(exporter, "campaign_context") as campaign:
            with self.assertRaises(FileExistsError):
                exporter.export(args)
            campaign.assert_not_called()
        path = self.directory / "external_aghashahi_v1.json"
        path.write_text("{}")
        with patch.object(exporter.overlap, "load_external") as load:
            with self.assertRaisesRegex(ValueError, "gel autorisé"):
                exporter.external_inputs(self.directory, path)
            load.assert_not_called()

    def test_source_or_weights_mutation_is_detected(self):
        path = self.directory / "weights"
        path.write_bytes(b"synthetic weights")
        files = {path: exporter.sha256_file(path)}
        exporter.verify_files(files)
        path.write_bytes(b"modified weights")
        with self.assertRaisesRegex(ValueError, "modifié"):
            exporter.verify_files(files)

    def publication_fixture(self):
        from harness.split_loader import Clip, Split
        directory = self.directory / "campaign"
        validation = directory / "validation"
        validation.mkdir(parents=True)
        threshold = .6123456789012345
        identity = {"checkpoint_checksums_sha256": "a" * 64, "temporal_sha256": "b" * 64,
            "config_hash": "c" * 64, "model_version": "synthetic-final", "preprocessing_version": "canonical-fixture",
            "scoring_spec_sha256": "d" * 64, "source_sha256": {"fixture.py": "e" * 64}}
        receipt = {"schema_version": "pipe-threshold-evidence-v1", "fit_fold": "val",
            "threshold": threshold, "threshold_repr": repr(threshold), "model_identity": identity,
            "rule": "argmax_validation_cluster_macro_f1;median_cluster_score;smallest_threshold_on_exact_tie",
            "split_sha256": exporter.split_loader.FROZEN_SPLIT_SHA256,
            "validation_predictions_sha256": "f" * 64,
            "threshold_method_sha256": exporter.sha256_file(ROOT / "scripts/eval/harness/metrics.py")}
        (validation / "threshold-evidence.json").write_text(json.dumps(receipt))
        manifest = self.directory / "external_aghashahi_v1.json"
        manifest.write_text(json.dumps({"preparation": {"fixture": True}}))
        manifest_sha = exporter.sha256_file(manifest)
        clips = tuple(Clip(r["clip_id"], i, "leak" if i else "no_leak", f"g{i}", "external")
                      for i, r in enumerate(self.rows))
        split = Split(clips, manifest_sha, manifest, {})
        gate_path = self.directory / "gate.json"
        gate_path.write_text("{}")
        states = {name: "a" * 64 for name in ("encoder", "projector", "llm")}
        ctx = {"directory": directory, "bundle": directory / "final/tslm/bundle",
            "registration": {"code_revision": "a" * 40}, "preregistration_sha256": "b" * 64,
            "model_identity": identity, "metadata": {"scoring_spec": {"version": "fixture"}},
            "receipts": {"tslm": receipt}, "thresholds": {"tslm": threshold},
            "gate": {"runtime": {}, "state_sha256": states}}
        args = SimpleNamespace(output=self.directory / "publication", model="tslm", audit_text=False,
            transform="T0", run_id="synthetic-final-external", code_revision="a" * 40,
            campaign=directory, external=self.directory, manifest=manifest,
            final_gate=gate_path, final_gate_reference=gate_path)
        rows = {"external": self.rows, "background": [{**self.rows[0], "clip_id": "background-only"}]}
        return args, ctx, split, rows, states

    def test_publication_real_contract_two_files_and_background_without_binary_labels(self):
        from pipe.tslm import coherent
        import run_v2_campaign as campaign
        args, ctx, split, rows, states = self.publication_fixture()
        calls, loads = [], []
        class Backend:
            def __init__(self, *args):
                loads.append(args)
            def state_hashes(self):
                return dict(states)
            def predict_waveform(self, waveform, rate):
                calls.append((waveform.copy(), rate))
                return {"probability_leak": .7, "prediction": "leak", "threshold": ctx["thresholds"]["tslm"],
                    "fallback_used": False, "audit": {"raw_text_exact": "fixture exact\n"}}
        with patch.object(exporter, "MANIFEST_SHA256", split.sha256), \
                patch.object(exporter, "campaign_context", return_value=ctx), \
                patch.object(exporter, "external_inputs", return_value=(split, rows, self.arrays, {})), \
                patch.object(campaign, "verify_runtime"), patch.object(coherent, "CoherentPredictor", Backend):
            result = exporter.export(args)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(loads), 1)
        self.assertEqual(len(calls), 3)
        self.assertEqual({p.name for p in (args.output / "run").iterdir()}, {"metadata.json", "predictions.csv"})
        lines = (args.output / "run/predictions.csv").read_text().splitlines()
        self.assertEqual(lines[0], "clip_id,probability_leak")
        self.assertEqual(len(lines), 3)
        self.assertNotIn("background-only", "\n".join(lines))
        background = (args.output / "audit/background-predictions.csv").read_text()
        self.assertEqual(background.splitlines()[0], "clip_id,probability_leak")
        self.assertIn("background-only", background)
        self.assertNotIn("no_leak", background)
        metadata = json.loads((args.output / "run/metadata.json").read_text())
        self.assertEqual(metadata["model_identity"], ctx["model_identity"])
        self.assertEqual(metadata["threshold_provenance_sha256"], exporter.sha256_file(
            ctx["directory"] / "validation/threshold-evidence.json"))
        self.assertNotIn("threshold", metadata)
        self.assertFalse(result["metrics_calculated"])

    def test_any_failed_window_prevents_publication_but_preserves_raw_audit(self):
        from pipe.tslm import coherent
        import run_v2_campaign as campaign
        args, ctx, split, rows, states = self.publication_fixture()
        class Backend:
            def __init__(self, *args):
                pass
            def state_hashes(self):
                return dict(states)
            def predict_waveform(self, waveform, rate):
                raise RuntimeError("Panne synthétique explicite")
        with patch.object(exporter, "MANIFEST_SHA256", split.sha256), \
                patch.object(exporter, "campaign_context", return_value=ctx), \
                patch.object(exporter, "external_inputs", return_value=(split, rows, self.arrays, {})), \
                patch.object(campaign, "verify_runtime"), patch.object(coherent, "CoherentPredictor", Backend):
            with self.assertRaisesRegex(ValueError, "Export incomplet"):
                exporter.export(args)
        self.assertFalse((args.output / "run").exists())
        self.assertTrue((args.output / "audit/failure.json").is_file())
        records = [json.loads(line) for line in (args.output / "audit/external.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(row["error"]["message"] == "Panne synthétique explicite" for row in records))


if __name__ == "__main__":
    unittest.main()
