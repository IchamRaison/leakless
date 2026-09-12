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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pipe.tslm import coherent


class FakeError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class FakeOutOfMemory(RuntimeError):
    pass


class CoherentChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.bundle = root / "bundle"
        self.bundle.mkdir()
        self.metadata = {"model_version": "candidate-fixture", "preprocessing_version": "canonical-fixture-v2",
                         "config_hash": "a" * 64, "max_new_tokens": 48}
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
                                       score=Mock(return_value=.7), score_waveform=Mock(return_value=.7), predict=predict)
        self.preprocess = Mock(return_value="canonical series")
        self.measure = Mock(return_value="0-1000")
        self.collate = Mock(return_value=[{"numeric_fixture": True}])
        self.model_input = Mock(return_value={"numeric_fixture": True})
        self.amplitude = Mock(return_value=np.arange(9, dtype=np.float64))
        def make_input(series, metadata, amplitude_features=None):
            if metadata.get("amplitude_evidence", False):
                return self.model_input(series, amplitude_features=amplitude_features)
            assert amplitude_features is None
            return self.model_input(series)
        self.factory = Mock(return_value=self.backend)
        self.modules = {
            "pipe.tslm.predict": SimpleNamespace(Predictor=self.factory, PredictionError=FakeError,
                OUTPUT_PATTERN=pattern, preprocess_for_model=self.preprocess,
                extend_time_series_to_match_patch_size_and_aggregate=self.collate,
                model_input_for_model=make_input),
            "pipe.tslm.preprocessing": SimpleNamespace(decode_wav=Mock(return_value="decoded waveform"),
                measured_band=self.measure, amplitude_features=self.amplitude),
            "torch": SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=FakeOutOfMemory))}
        module_patch = patch.dict(sys.modules, self.modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)

    def service(self):
        return coherent.CoherentPredictor(self.bundle, self.decision_path, self.evidence_path, device="cpu")

    def test_memory_state_audit_uses_same_backend_and_releases_shared_lock(self):
        service = self.service()
        for name in ("encoder", "projector", "llm"):
            setattr(self.model, name, {"state": name})
        hasher = Mock(side_effect=lambda module: module["state"])
        with patch.dict(sys.modules, {"pipe.tslm.train": SimpleNamespace(tensor_state_hash=hasher)}):
            before = service.state_hashes()
            self.model.encoder["state"] = "changed"
            self.assertNotEqual(before, service.state_hashes())
            service._lock.acquire()
            with self.assertRaises(FakeError) as raised:
                service.state_hashes()
            self.assertEqual(raised.exception.code, "model_busy")
            service._lock.release()
            hasher.side_effect = RuntimeError("hash failed")
            with self.assertRaisesRegex(RuntimeError, "hash failed"):
                service.state_hashes()
            self.assertFalse(service._lock.locked())
        self.factory.assert_called_once()

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

    def test_waveform_retains_values_without_pcm16_and_shares_wav_decision(self):
        waveform = np.linspace(-1_700_000_000.125, 1_700_000_000.125, 8000, dtype=np.float64)
        original = waveform.copy()
        self.modules["pipe.tslm.preprocessing"].decode_wav.return_value = waveform
        service = self.service()
        wav = service.predict(b"synthetic original WAV")
        self.preprocess.reset_mock()
        self.modules["pipe.tslm.preprocessing"].decode_wav.reset_mock()
        result = service.predict_waveform(waveform, 8000)
        for key in ("prediction", "probability_leak", "threshold", "description", "dominant_band_hz",
                    "description_source", "fallback_used", "decision_artifact_sha256"):
            self.assertEqual(result[key], wav[key])
        self.assertIsNone(result["audit"]["raw_api_payload"])
        self.assertEqual(result["audit"]["raw_text_exact"], wav["audit"]["raw_text_exact"])
        self.assertFalse(result["audit"]["input_identity"]["requantized"])
        snapshot, rate = self.backend.score_waveform.call_args.args
        np.testing.assert_array_equal(snapshot, original)
        np.testing.assert_array_equal(waveform, original)
        self.assertEqual(snapshot.dtype, np.float64)
        self.assertIsNot(snapshot, waveform)
        self.assertEqual(rate, 8000)
        self.assertIs(self.preprocess.call_args.args[0], snapshot)
        self.assertEqual(self.preprocess.call_args.args[1:], (8000, self.metadata))
        self.modules["pipe.tslm.preprocessing"].decode_wav.assert_not_called()
        self.collate.assert_called_once_with([{"numeric_fixture": True}], normalize=False)
        self.model_input.assert_called_once_with("canonical series")
        self.generate.assert_called_with([{"numeric_fixture": True}], max_new_tokens=48, max_time=15.0)
        self.assertEqual(result["input_sha256"], coherent._waveform_identity(original, 8000))

    def test_waveform_identity_describes_values_and_rate_not_npy_or_byte_order(self):
        little = np.array([1.25, -25000.5, 90000.125], dtype="<f8")
        big = little.astype(">f8")
        digest = coherent._waveform_identity(little, 8000)
        self.assertEqual(digest, coherent._waveform_identity(big, 8000))
        self.assertNotEqual(digest, coherent._waveform_identity(little, 16000))
        changed = little.copy()
        changed[0] = np.nextafter(changed[0], np.inf)
        self.assertNotEqual(digest, coherent._waveform_identity(changed, 8000))

    def test_waveform_generation_uses_the_same_opt_in_amplitude_policy(self):
        service = self.service()
        self.backend.metadata["amplitude_evidence"] = True  # Double privé, pas mutation d'un vrai bundle.
        waveform = np.linspace(-.7, .9, 8000)
        result = service.predict_waveform(waveform)
        snapshot, rate = self.backend.score_waveform.call_args.args
        self.assertIs(self.amplitude.call_args.args[0], snapshot)
        self.assertEqual(self.amplitude.call_args.args[1], rate)
        self.assertIs(self.model_input.call_args.kwargs["amplitude_features"], self.amplitude.return_value)
        self.assertEqual(result["prediction"], "leak")
        self.assertEqual(result["description_source"], "llm_checked_against_dsp")

    def test_waveform_errors_and_text_fallbacks_preserve_audit_and_release_lock(self):
        service = self.service()
        waveform = np.arange(8000, dtype=np.float64)
        for text, reason in (("no_leak; Greatest mean spectral energy: 0-1000 Hz.", "generated_class_disagrees_with_score"),
                             ("leak; Greatest mean spectral energy: 3000-4000 Hz.", "generated_band_disagrees_with_dsp"),
                             ("invalid text", "invalid_generated_format")):
            self.generate.return_value = [text]
            result = service.predict_waveform(waveform)
            self.assertEqual(result["prediction"], "leak")
            self.assertIn(reason, result["fallback_reasons"])
            self.assertEqual(result["audit"]["raw_text_exact"], text)
            self.assertIs(self.model.generate, self.generate)
        self.generate.side_effect = UnicodeError("waveform text decode error")
        result = service.predict_waveform(waveform)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["audit"]["raw_error"]["type"], "UnicodeError")
        self.generate.side_effect = FakeOutOfMemory("out of memory fixture")
        with self.assertRaises(FakeError) as caught:
            service.predict_waveform(waveform)
        self.assertEqual(caught.exception.code, "gpu_out_of_memory")
        self.assertEqual(caught.exception.coherent_audit["raw_error"]["code"], "gpu_out_of_memory")
        self.assertFalse(service._lock.locked())
        self.assertIs(self.model.generate, self.generate)
        self.generate.side_effect = None
        for invalid_return in ([], None, [None]):
            self.generate.return_value = invalid_return
            with self.assertRaises(FakeError) as caught:
                service.predict_waveform(waveform)
            self.assertEqual(caught.exception.code, "model_unavailable")
            self.assertIsNotNone(caught.exception.coherent_audit["raw_error"])
            self.assertIs(self.model.generate, self.generate)
            self.assertFalse(service._lock.locked())
        for score in (float("nan"), -1, True):
            self.backend.score_waveform.return_value = score
            with self.assertRaises(FakeError) as caught:
                service.predict_waveform(waveform)
            self.assertEqual(caught.exception.code, "invalid_score")
        self.backend.score_waveform.return_value = .7
        for code in ("silent_audio", "unsupported_audio", "model_unavailable"):
            self.backend.score_waveform.side_effect = FakeError(code, "invalid waveform fixture")
            with self.assertRaises(FakeError) as caught:
                service.predict_waveform(waveform)
            self.assertEqual(caught.exception.code, code)
        self.backend.score_waveform.side_effect = None
        for bad in (np.array([float("nan")]), np.array([1j]), np.array(["text"]), np.zeros((2, 2))):
            with self.assertRaises(FakeError) as caught:
                service.predict_waveform(bad)
            self.assertEqual(caught.exception.code, "unsupported_audio")
            self.assertFalse(service._lock.locked())


if __name__ == "__main__":
    unittest.main()
