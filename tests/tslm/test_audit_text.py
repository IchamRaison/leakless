"""Fixtures synthétiques CPU : aucune donnée/poids réels, aucune métrique finale."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import audit_text

PATTERN = re.compile(r"(leak|no_leak);\s*(Greatest mean spectral energy: (?:0-1000|1000-2000|2000-3000|3000-4000) Hz\.)")
TEXT = "  leak; Greatest mean spectral energy: 0-1000 Hz.\n"


class TextAuditChecks(unittest.TestCase):
    def test_frozen_audio_manifest_and_per_clip_bytes_are_verified(self):
        expected = hashlib.md5(b"original").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "split_v2_audit.csv"
            path.write_text(f"clip_id,md5\nfixture,{expected}\n")
            with patch.object(audit_text, "FROZEN_AUDIT_SHA256", audit_text.sha256_file(path)):
                values = audit_text.load_expected_audio_md5(root)
                self.assertEqual(values, {"fixture": expected})
                audit_text.verify_audio_bytes(b"original", "fixture", values)
                with self.assertRaises(ValueError):
                    audit_text.verify_audio_bytes(b"substituted", "fixture", values)
                path.write_text(f"clip_id,md5\nwrong,{expected}\n")
                with self.assertRaises(ValueError):
                    audit_text.load_expected_audio_md5(root)

    def test_capture_is_transparent_including_raw_whitespace_and_exceptions(self):
        returned = [TEXT]
        original = Mock(return_value=returned)
        model = SimpleNamespace(generate=original)
        with audit_text.capture_generation(model) as calls:
            self.assertIs(model.generate("input", max_new_tokens=48), returned)
            original.assert_called_once_with("input", max_new_tokens=48)
            self.assertEqual(calls[0]["returned"], [TEXT])
            original.side_effect = RuntimeError("synthetic failure")
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                model.generate("input")
            self.assertEqual(calls[1]["error"]["message"], "synthetic failure")
        self.assertIs(model.generate, original)

    def test_all_clip_denominators_include_wrong_invalid_and_failed_outputs(self):
        rows = []
        cases = [(TEXT, {"abstained": False}, None),
                 (TEXT.replace("0-1000", "1000-2000"), {"abstained": False}, None),
                 ("unparseable", {"abstained": True}, None),
                 (None, None, {"type": "RuntimeError"})]
        for i, (text, payload, error) in enumerate(cases):
            calls = [{"returned": [text], "elapsed_ms": 2., "error": None}] if text else []
            row = {"clip_id": str(i), "fold": "val", "api_payload": payload, "error": error,
                   "api_elapsed_ms": 3., "generation_calls": calls}
            row.update(audit_text.grade_text(payload, calls, "0-1000", .2, .5, PATTERN))
            rows.append(row)
        test_row = {**rows[0], "clip_id": "test", "fold": "test"}
        summary = audit_text.summarize([*rows, test_row], {"val": 4, "test": 1})
        self.assertEqual(summary["val"]["n_correct_band"], 1)
        self.assertEqual(summary["val"]["correct_band_rate_all_clips"], .25)
        self.assertEqual(summary["val"]["n_errors"], 1)
        self.assertEqual(summary["val"]["n_abstentions"], 1)
        self.assertEqual(summary["val"]["n_class_comparisons_unavailable"], 2)
        self.assertEqual(summary["val"]["n_class_disagreements"], 2)
        self.assertEqual(summary["test"]["correct_band_rate_all_clips"], 1.)
        self.assertEqual(rows[0]["generated_text_exact"], TEXT)
        self.assertIsNone(rows[3]["generated_class"])
        self.assertIsNone(rows[3]["class_decision_disagreement"])
        with self.assertRaises(ValueError):
            audit_text.summarize([*rows, test_row], {"val": 5, "test": 1})

    def test_threshold_is_full_precision_verified_on_val_not_test(self):
        full = .543210123456789
        run = SimpleNamespace(run_id="fixture", metadata={"training_commit": "training", "config_hash": "config"})
        split = SimpleNamespace(sha256="split")
        report = {"split": {"sha256": "split"}, "runs": {"fixture": {
            "run_id": "fixture", "training_commit": "training", "config_hash": "config",
            "threshold_recomputed_on_val": round(full, 6), "folds": {"val": {"threshold": full}}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps(report))
            with patch.object(audit_text, "vectors", return_value=(np.array([0, 1]), np.array([.2, .8]),
                                                                     np.array(["a", "b"]), ["x", "y"])) as vectors, \
                    patch.object(audit_text.metrics, "pick_threshold", return_value=full) as pick:
                self.assertEqual(audit_text.validation_threshold(split, run, path), full)
                vectors.assert_called_once_with(split, run, "val")
                self.assertEqual(pick.call_args.kwargs, {"fold": "val"})
                with self.assertRaises(ValueError):
                    audit_text.validation_threshold(split, run, path, round(full, 6))
                report["runs"]["fixture"]["folds"]["val"]["threshold"] = round(full, 6)
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    audit_text.validation_threshold(split, run, path)

    def test_existing_output_fails_before_importing_or_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                audit_text.audit(argparse.Namespace(output=Path(directory)))

    def test_full_mock_audit_records_402_attempts_and_excludes_warmup(self):
        clips = [SimpleNamespace(clip_id=f"c{i:03}", fold="val" if i < 208 else "test") for i in range(402)]
        split = SimpleNamespace(sha256="split", clips=clips,
            fold=lambda name: [c for c in reversed(clips) if c.fold == name],
            by_id=lambda: {c.clip_id: c for c in clips},
            path_of=lambda cid: "substituted.wav" if cid == "c002" else "sample.wav")
        metadata = {"transform": "T0", "checkpoint_checksums_sha256": "fixture", "temporal_sha256": "fixture",
                    "scoring_spec": {"version": "fixture"}, "config_hash": "fixture", "training_commit": "fixture",
                    "source_sha256": {name: "fixture" for name in (
                        "src/pipe/tslm/model.py", "src/pipe/tslm/predict.py", "src/pipe/tslm/preprocessing.py",
                        "scripts/timenet/leakless_acoustic/connector.py")}}
        run = SimpleNamespace(run_id="fixture", metadata=metadata, probabilities={c.clip_id: .2 for c in clips})
        model = SimpleNamespace(generate=Mock(return_value=[TEXT]), scoring_spec=lambda: metadata["scoring_spec"])
        seen_inputs = []

        def predict(raw):
            self.assertIsInstance(raw, bytes)
            seen_inputs.append(raw)
            if len(seen_inputs) == 3:  # Une panne réelle de l'API synthétique, jamais omise.
                raise RuntimeError("synthetic inference failure")
            model.generate("numeric-only fixture")
            return SimpleNamespace(model_dump=lambda **kw: {"prediction": "leak", "abstained": False,
                                                            "description": TEXT.strip().split("; ", 1)[1]})

        predictor = SimpleNamespace(model=model, metadata=metadata, predict=predict)
        modules = {"pipe.tslm.predict": SimpleNamespace(Predictor=Mock(return_value=predictor), OUTPUT_PATTERN=PATTERN),
                   "pipe.tslm.preprocessing": SimpleNamespace(decode_wav=lambda raw: np.ones(8000),
                       preprocess_audio=lambda raw, rate: np.zeros((4, 64)), measured_band=lambda series: "0-1000")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "bundle"
            checkpoint.mkdir()
            (checkpoint / "reload-expected.json").write_text(json.dumps({"clip_id": "c000"}))
            (checkpoint / "reload-example.wav").write_bytes(b"sample")
            (root / "sample.wav").write_bytes(b"sample")
            (root / "substituted.wav").write_bytes(b"modified audio")
            run_dir = root / "run"
            run_dir.mkdir()
            with (run_dir / "predictions.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["clip_id", "probability_leak"])
                writer.writerows((c.clip_id, .2) for c in clips)
            args = argparse.Namespace(output=root / "audit", code_revision="a" * 40, manifests=root,
                run=run_dir, checkpoint=checkpoint, data_root=root, threshold_provenance=root / "metrics.json",
                threshold=None, device="cpu")
            with patch.dict(sys.modules, modules), patch.object(audit_text, "sha256_file", return_value="fixture"), \
                    patch.object(audit_text, "load_expected_audio_md5", return_value={
                        c.clip_id: hashlib.md5(b"sample").hexdigest() for c in clips}), \
                    patch.object(audit_text.split_loader, "load_split", return_value=split), \
                    patch.object(audit_text.contract, "load_run", return_value=run), \
                    patch.object(audit_text, "validation_threshold", return_value=.5), patch("builtins.print"):
                audit_text.audit(args)
            rows = [json.loads(line) for line in (args.output / "raw.jsonl").read_text().splitlines()]
            summary = json.loads((args.output / "summary.json").read_text())
            provenance = json.loads((args.output / "provenance.json").read_text())
            self.assertEqual(len(seen_inputs), 402)  # 1 warmup + 401 WAV intègres ; 1 WAV refusé.
            self.assertEqual(len(rows), 402)
            self.assertEqual([row["clip_id"] for row in rows], [c.clip_id for c in clips])
            self.assertEqual(summary["folds"]["val"]["n_errors"], 2)
            self.assertEqual(summary["folds"]["val"]["n_correct_band"], 206)
            self.assertEqual(summary["folds"]["test"]["n_attempted"], 194)
            self.assertEqual(rows[1]["error"]["message"], "synthetic inference failure")
            self.assertIsNone(rows[1]["generated_class"])
            self.assertEqual(rows[2]["error"]["stage"], "audio_integrity")
            self.assertFalse(rows[2]["audio_md5_verified"])
            self.assertIsNone(rows[2]["class_decision_disagreement"])
            self.assertEqual(rows[0]["generated_text_exact"], TEXT)
            self.assertFalse(summary["warmup_included"])
            self.assertEqual(provenance["status"], "complete")
            self.assertEqual({p.name for p in args.output.iterdir()}, {"raw.jsonl", "summary.json", "provenance.json"})


if __name__ == "__main__":
    unittest.main()
