"""Contrôles CPU : contrat d'export, gel et RNG interprocessus, sans modèle."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import export_run  # noqa: E402
from check_v1_reload import close_score  # noqa: E402


class ExportChecks(unittest.TestCase):
    def test_predictions_exact_schema_mapping_precision_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            scores = {"clip_b": 0.12345678901234567, "clip_a": 0.999999999999991}
            export_run.write_predictions(path, ["clip_b", "clip_a"], scores)
            with path.open() as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ["clip_id", "probability_leak"])
                rows = list(reader)
            self.assertEqual([r["clip_id"] for r in rows], ["clip_a", "clip_b"])
            self.assertEqual({r["clip_id"]: float(r["probability_leak"]) for r in rows}, scores)
            original = path.read_bytes()
            for bad in ({"clip_a": float("nan"), "clip_b": .2},
                        {"clip_a": float("inf"), "clip_b": .2},
                        {"clip_a": -0.1, "clip_b": .2}, {"clip_a": 1.01, "clip_b": .2},
                        {"clip_a": .2}, {**scores, "unknown": .5}):
                with self.assertRaises(ValueError):
                    export_run.write_predictions(path, ["clip_a", "clip_b"], bad)
                self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                export_run.write_predictions(path, ["clip_a", "clip_a"], {"clip_a": .2})

    def test_existing_run_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "metadata.json").write_text("existing")
            with self.assertRaises(FileExistsError):
                export_run.export(argparse.Namespace(output=path))
            self.assertEqual((path / "metadata.json").read_text(), "existing")

    def test_export_requires_full_code_revision_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            for revision in ("", "abc123", "g" * 40):
                args = argparse.Namespace(output=Path(directory) / "new-run", run_id="fixture",
                                          code_revision=revision)
                with self.assertRaises(ValueError):
                    export_run.export(args)
                self.assertFalse(args.output.exists())

    def test_changed_audit_mapping_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "split_v2_audit.csv"
            original = (ROOT / "manifests/split_v2_audit.csv").read_bytes()
            audit.write_bytes(original + b"\n")
            self.assertNotEqual(export_run.sha256_file(audit), export_run.FROZEN_AUDIT_SHA256)
            args = argparse.Namespace(output=root / "new-run", run_id="fixture",
                                      code_revision="a" * 40, manifests=root)
            predictor = Mock()
            with patch.dict(sys.modules, {"pipe.tslm.predict": SimpleNamespace(Predictor=predictor)}):
                with self.assertRaisesRegex(ValueError, "split_v2_audit.csv divergent"):
                    export_run.export(args)
            predictor.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_frozen_provenance_requires_real_count_matching_reload_and_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"version": "fixture", "aggregation": "sum"}
            checksums = {key: "fixture" for key in (
                "scoring_spec.json", "reload-expected.json", "reload-example.wav",
                "training-report.json", "metadata.json", "temporal.pt")}
            (root / "checksums.json").write_text(json.dumps(checksums))
            (root / "scoring_spec.json").write_text(json.dumps(spec))
            report = {"fresh_process_reload": True, "score_match": True,
                      "deterministic_output_match": True, "input_paths_match": True,
                      "checkpoint_checksums_sha256": export_run.sha256_file(root / "checksums.json")}
            report_path = root / "reload.json"
            report_path.write_text(json.dumps(report))
            predictor = SimpleNamespace(
                metadata={"n_configs_compared": 3, "training_commit": "fixture-commit",
                          "config_hash": "fixture-config", "test_labels_not_used_for_tuning": True},
                model=SimpleNamespace(scoring_spec=lambda: spec))
            got = export_run.frozen_provenance(root, report_path, predictor)
            self.assertEqual(got["n_configs_compared"], 3)
            for bad_count in (None, True, 0, 4, "3"):
                predictor.metadata["n_configs_compared"] = bad_count
                with self.assertRaises(ValueError):
                    export_run.frozen_provenance(root, report_path, predictor)
            predictor.metadata["n_configs_compared"] = 3
            for declaration in (None, False, "true"):
                predictor.metadata["test_labels_not_used_for_tuning"] = declaration
                with self.assertRaises(ValueError):
                    export_run.frozen_provenance(root, report_path, predictor)
            predictor.metadata["test_labels_not_used_for_tuning"] = True
            report["checkpoint_checksums_sha256"] = "different-checkpoint"
            report_path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                export_run.frozen_provenance(root, report_path, predictor)
            report["checkpoint_checksums_sha256"] = export_run.sha256_file(root / "checksums.json")
            report_path.write_text(json.dumps(report))
            predictor.model.scoring_spec = lambda: {"version": "different"}
            with self.assertRaises(ValueError):
                export_run.frozen_provenance(root, report_path, predictor)

    def test_reload_tolerance_is_small_absolute_and_rejects_invalid_scores(self):
        self.assertTrue(close_score(.5, .5000001))
        for actual, expected in ((.5, .5001), (float("nan"), .5), (float("inf"), .5),
                                 (1.01, 1.01), (-.01, -.01)):
            self.assertFalse(close_score(actual, expected))

    def test_complete_export_mock_maps_ids_and_stresses_float_audio(self):
        clips = [SimpleNamespace(clip_id=f"c{i:03}", fold="val" if i < 208 else "test")
                 for i in reversed(range(402))]
        split = SimpleNamespace(clips=clips, sha256=export_run.split_loader.FROZEN_SPLIT_SHA256,
                                path_of=lambda cid: "odd.wav" if int(cid[1:]) % 2 else "even.wav")
        waveform_calls = []

        def score_waveform(waveform, rate):
            self.assertEqual(waveform.shape, (8000,))
            self.assertEqual(waveform.dtype, np.float64)
            self.assertEqual(rate, 8000)
            waveform_calls.append(waveform)
            return float(waveform.mean() / 10000)

        predictor = SimpleNamespace(score=lambda raw: float(raw) / 1000, score_waveform=score_waveform)
        fake_modules = {
            "pipe.tslm.predict": SimpleNamespace(Predictor=lambda *a, **kw: predictor),
            "pipe.tslm.preprocessing": SimpleNamespace(
                decode_wav=lambda raw: np.arange(8000, dtype=np.float64) + float(raw))}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "odd.wav").write_bytes(b"300")
            (root / "even.wav").write_bytes(b"200")
            (root / "split_v2_audit.csv").write_text("fixture audit mapping\n")
            audit_sha256 = export_run.sha256_file(root / "split_v2_audit.csv")
            for transform in ("T0", "T2"):
                args = argparse.Namespace(output=root / transform, run_id=f"fixture-{transform}",
                                          checkpoint=root, reload_report=root / "reload.json",
                                          manifests=root, data_root=root, transform=transform, device="cpu",
                                          code_revision="a" * 40)
                with patch.dict(sys.modules, fake_modules), \
                        patch.object(export_run, "FROZEN_AUDIT_SHA256", audit_sha256), \
                        patch.object(export_run.split_loader, "load_split", return_value=split), \
                        patch.object(export_run, "frozen_provenance", return_value={"n_configs_compared": 3}), \
                        patch.object(export_run.subprocess, "run") as controller, patch("builtins.print"):
                    export_run.export(args)
                self.assertIn("--template", controller.call_args_list[0].args[0])
                self.assertIn("--inspect", controller.call_args_list[1].args[0])
                self.assertEqual({p.name for p in args.output.iterdir()}, {"metadata.json", "predictions.csv"})
                with (args.output / "predictions.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 402)
                self.assertEqual([r["clip_id"] for r in rows], sorted(c.clip_id for c in clips))
                expected = .2 if transform == "T0" else .41995
                self.assertAlmostEqual(float(rows[0]["probability_leak"]), expected)
                metadata = json.loads((args.output / "metadata.json").read_text())
                self.assertEqual(metadata["transform"], transform)
                self.assertEqual(metadata["export_commit"], "a" * 40)
                self.assertEqual(metadata["split_audit_filename"], "split_v2_audit.csv")
                self.assertEqual(metadata["split_audit_sha256"], audit_sha256)
                self.assertIn("scripts/timenet/leakless_acoustic/connector.py", metadata["source_sha256"])
            self.assertEqual(len(waveform_calls), 402)

    def test_official_stress_golden_seed_and_cross_process_identity(self):
        self.assertEqual(export_run.sha256_file(ROOT / "scripts/temporal/stress.py"),
                         export_run.STRESS_SHA256)
        snippet = (
            "import hashlib, json, sys; import numpy as np; "
            f"sys.path.insert(0, {str(ROOT / 'scripts/temporal')!r}); "
            "from stress import derive_seed, TRANSFORMS, clip_rng; "
            "assert derive_seed('T2','c003cd25f5a5f') == 10036266156883065158; "
            "x=np.sin(np.arange(8000)/50.0); "
            "print(json.dumps({n: hashlib.sha256(TRANSFORMS[n]['fn'](x, "
            "clip_rng(n, 'toy_clip')).tobytes()).hexdigest() for n in ('T2','T3')}))")
        outputs = []
        for hashseed in ("1", "777"):
            env = dict(os.environ, PYTHONHASHSEED=hashseed)
            outputs.append(subprocess.check_output([sys.executable, "-c", snippet], env=env, text=True))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
