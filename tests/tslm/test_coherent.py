"""Restitution/provenance sur faux backend CPU ; aucun poids ni signal réel."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pipe.tslm import coherent


class FakeError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class CoherentChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.bundle = root / "bundle"
        self.bundle.mkdir()
        self.metadata = {"model_version": "candidate-fixture", "preprocessing_version": "canonical-fixture-v2",
                         "config_hash": "a" * 64}
        self.spec = {"version": "scoring-fixture"}
        (self.bundle / "metadata.json").write_text(json.dumps(self.metadata))
        (self.bundle / "scoring_spec.json").write_text(json.dumps(self.spec))
        (self.bundle / "temporal.pt").write_bytes(b"synthetic weights")
        checksums = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.bundle.iterdir()}
        (self.bundle / "checksums.json").write_text(json.dumps(checksums))
        self.source_patch = patch.object(coherent, "_source_fingerprints", return_value={"fixture.py": "b" * 64})
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        self.evidence_path, self.decision_path = root / "validation.json", root / "decision.json"
        self.receipt = {"schema_version": "pipe-threshold-evidence-v1", "fit_fold": "val",
                        "threshold": .6123456789012345, "threshold_repr": repr(.6123456789012345),
                        "model_identity": coherent.checkpoint_identity(self.bundle), "rule": "fixed validation rule",
                        "split_sha256": "c" * 64, "validation_predictions_sha256": "d" * 64,
                        "threshold_method_sha256": "e" * 64}
        self.evidence_path.write_text(json.dumps(self.receipt))
        self.artifact = coherent.decision_artifact(self.bundle, self.evidence_path, "decision-fixture-1")
        self.decision_path.write_text(json.dumps(self.artifact))
        pattern = re.compile(r"(leak|no_leak);\s*(Greatest mean spectral energy: (?:0-1000|1000-2000|2000-3000|3000-4000) Hz\.)")
        self.generate = Mock(return_value=["leak; Greatest mean spectral energy: 0-1000 Hz.\n"])
        self.model = SimpleNamespace(generate=self.generate, scoring_spec=lambda: self.spec)

        def predict(raw):
            generated = self.model.generate({"numeric_fixture": True})[0].strip()
            match = pattern.fullmatch(generated)
            payload = {"prediction": match[1] if match else None, "abstained": match is None,
                       "description": match[2] if match else generated}
            return SimpleNamespace(model_dump=lambda **kwargs: payload)

        self.backend = SimpleNamespace(metadata=self.metadata, model=self.model,
                                       score=Mock(return_value=.7), predict=predict)
        self.preprocess = Mock(return_value="canonical series")
        self.measure = Mock(return_value="0-1000")
        self.factory = Mock(return_value=self.backend)
        self.modules = {
            "pipe.tslm.predict": SimpleNamespace(Predictor=self.factory, PredictionError=FakeError,
                                                 OUTPUT_PATTERN=pattern, preprocess_for_model=self.preprocess),
            "pipe.tslm.preprocessing": SimpleNamespace(decode_wav=Mock(return_value="decoded waveform"),
                                                        measured_band=self.measure)}
        module_patch = patch.dict(sys.modules, self.modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)

    def service(self):
        return coherent.CoherentPredictor(self.bundle, self.decision_path, self.evidence_path, device="cpu")

    def test_checked_text_and_canonical_dispatch_and_exact_boundary(self):
        service = self.service()
        self.backend.score.return_value = self.artifact["threshold"]
        result = service.predict(b"fixture WAV")
        self.assertEqual(result["prediction"], "leak")  # >= et seuil plein, aucun 0.5 implicite.
        self.assertEqual(result["score_type"], "raw")
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["description_source"], "llm_checked_against_dsp")
        self.assertEqual(result["audit"]["raw_text_exact"], self.generate.return_value[0])
        self.preprocess.assert_called_once_with("decoded waveform", 8000, self.metadata)
        self.measure.assert_called_once_with("canonical series")
        self.assertIs(self.model.generate, self.generate)

    def test_raw_class_disagreement_never_competes_with_score(self):
        self.generate.return_value = ["no_leak; Greatest mean spectral energy: 0-1000 Hz.\n"]
        result = self.service().predict(b"fixture WAV")
        self.assertEqual(result["prediction"], "leak")
        self.assertTrue(result["fallback_used"])
        self.assertIn("generated_class_disagrees_with_score", result["fallback_reasons"])
        self.assertEqual(result["audit"]["raw_api_payload"]["prediction"], "no_leak")
        self.assertEqual(result["description"], "Greatest mean spectral energy: 0-1000 Hz.")

    def test_wrong_band_and_invalid_text_fall_back_without_claiming_raw_success(self):
        for text, reason in (("leak; Greatest mean spectral energy: 1000-2000 Hz.", "generated_band_disagrees_with_dsp"),
                             ("Fuite certaine à Paris.", "invalid_generated_format")):
            self.generate.return_value = [text]
            result = self.service().predict(b"fixture WAV")
            self.assertEqual(result["dominant_band_hz"], "0-1000")
            self.assertEqual(result["description"], "Greatest mean spectral energy: 0-1000 Hz.")
            self.assertEqual(result["description_source"], "dsp_template_fallback")
            self.assertIn(reason, result["fallback_reasons"])
            self.assertEqual(result["audit"]["raw_text_exact"], text)

    def test_identified_text_decode_failure_preserves_error_with_valid_score(self):
        self.generate.side_effect = UnicodeError("synthetic text decode failure")
        result = self.service().predict(b"fixture WAV")
        self.assertEqual(result["prediction"], "leak")
        self.assertTrue(result["fallback_used"])
        self.assertIsNone(result["audit"]["raw_text_exact"])
        self.assertEqual(result["audit"]["raw_error"]["type"], "UnicodeError")
        self.assertIs(self.model.generate, self.generate)

    def test_score_and_backend_errors_never_return_no_leak(self):
        service = self.service()
        for value in (float("nan"), float("inf"), -.1, 1.1, True, "0.1"):
            self.backend.score.return_value = value
            with self.assertRaises(FakeError) as caught:
                service.predict(b"fixture WAV")
            self.assertEqual(caught.exception.code, "invalid_score")
            self.assertFalse(service._lock.locked())
        self.generate.assert_not_called()
        self.backend.score.return_value = .1
        for code in ("unsupported_audio", "silent_audio", "model_unavailable", "gpu_out_of_memory"):
            self.backend.score.side_effect = FakeError(code, "synthetic score failure")
            with self.assertRaises(FakeError) as caught:
                service.predict(b"invalid")
            self.assertEqual(caught.exception.code, code)
        self.backend.score.side_effect = None
        for failure in (FakeError("gpu_out_of_memory", "synthetic GPU failure"), RuntimeError("CUDA error: fixture")):
            self.generate.side_effect = failure
            with self.assertRaises(FakeError) as caught:
                service.predict(b"fixture WAV")
            self.assertEqual(caught.exception.coherent_audit["raw_error"]["message"], str(failure))
            self.assertIs(self.model.generate, self.generate)
            self.assertFalse(service._lock.locked())

    def test_stale_threshold_model_and_evidence_rejected_before_backend_load(self):
        for key in ("temporal_sha256", "config_hash", "scoring_spec_sha256", "preprocessing_version", "source_sha256"):
            receipt = copy.deepcopy(self.receipt)
            receipt["model_identity"][key] = "wrong"
            self.evidence_path.write_text(json.dumps(receipt))
            with self.assertRaises(FakeError) as caught:
                self.service()
            self.assertEqual(caught.exception.code, "model_unavailable")
        self.evidence_path.write_text(json.dumps(self.receipt))
        artifact = {**self.artifact, "threshold": .5}
        self.decision_path.write_text(json.dumps(artifact))
        with self.assertRaises(FakeError):
            self.service()
        self.factory.assert_not_called()

    def test_receipt_rejects_test_fold_nonfinite_bool_missing_and_duplicate_fields(self):
        for changes in ({"fit_fold": "test"}, {"threshold": True}, {"threshold": float("nan")},
                        {"threshold": "0.5"}, {"validation_predictions_sha256": "missing"}):
            self.evidence_path.write_text(json.dumps({**self.receipt, **changes}))
            with self.assertRaises(ValueError):
                coherent.decision_artifact(self.bundle, self.evidence_path, "fixture")
        self.evidence_path.write_text('{"threshold": 0.1, "threshold": 0.9}')
        with self.assertRaisesRegex(ValueError, "répété"):
            coherent.decision_artifact(self.bundle, self.evidence_path, "fixture")
        self.evidence_path.write_text(json.dumps({key: value for key, value in self.receipt.items() if key != "threshold"}))
        with self.assertRaises(ValueError):
            coherent.decision_artifact(self.bundle, self.evidence_path, "fixture")


if __name__ == "__main__":
    unittest.main()
