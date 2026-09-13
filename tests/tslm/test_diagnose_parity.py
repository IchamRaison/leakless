"""Tests CPU de localisation numérique ; aucun modèle téléchargé ni score qualité."""
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import diagnose_parity as diagnostic


class ParityDiagnosticChecks(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch disponible dans le runtime ML")
    def test_torch_hooks_preserve_outputs_and_capture_causal_positions(self):
        import torch

        class Decoder(torch.nn.Module):
            def forward(self, *, inputs_embeds, attention_mask):
                self.result = SimpleNamespace(logits=torch.arange(2 * 7 * 16, dtype=torch.float32).reshape(2, 7, 16))
                return self.result

        model = SimpleNamespace(encoder=torch.nn.Identity(), projector=torch.nn.Identity(), llm=Decoder(),
                                tokenizer=SimpleNamespace(pad_token_id=0),
                                scoring_spec=lambda: {"class_token_ids": [[1, 2], [3, 4, 5]],
                                                      "class_token_counts": [2, 3]})
        x = torch.ones((1, 4, 6))
        with diagnostic.capture_stages(model) as (arrays, dtypes):
            self.assertIs(model.encoder(x), x)
            self.assertIs(model.projector(x), x)
            returned = model.llm(inputs_embeds=torch.ones((2, 7, 6)), attention_mask=torch.ones((2, 7)))
            self.assertIs(returned, model.llm.result)
        np.testing.assert_array_equal(arrays["class_logits"], returned.logits[:, 3:6].numpy())
        self.assertEqual(arrays["class_token_logprobs"][0, 2], 0)
        self.assertEqual(dtypes["encoder_0_output"], "torch.float32")
        for module in (model.encoder, model.projector, model.llm):
            self.assertFalse(module._forward_hooks)
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with diagnostic.capture_stages(model):
                raise RuntimeError("fixture")
        self.assertFalse(model.llm._forward_hooks)

    def test_float_roundtrip_is_not_declared_exact_by_allclose(self):
        x = np.array([1.00000006, -0.123456789], dtype=np.float64)
        result = diagnostic.array_difference(x, x.astype(np.float32))
        self.assertFalse(result["exact_values"])
        self.assertLess(result["max_abs"], 1e-6)
        self.assertTrue(result["same_shape"])
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            diagnostic.array_info(np.array([float("nan")]))
        traces = diagnostic.compare_stages({"input": x, "output": x},
                                            {"input": x.copy(), "output": x.astype(np.float32)})
        self.assertEqual(traces["first_exact_divergence"], "output")

    def test_fixed_cases_and_all_validation_never_select_test(self):
        clips = [SimpleNamespace(clip_id=cid, fold="val") for cid in diagnostic.VALIDATION_CASES]
        clips += [SimpleNamespace(clip_id=f"val{i}", fold="val") for i in range(205)]
        clips += [SimpleNamespace(clip_id=diagnostic.TRAIN_CASE, fold="train"),
                  SimpleNamespace(clip_id="forbidden-test", fold="test")]
        split = SimpleNamespace(by_id=lambda: {c.clip_id: c for c in clips},
                                fold=lambda name: [c for c in clips if c.fold == name])
        self.assertEqual(len(diagnostic.selected_ids(split, False)), 4)
        self.assertEqual(len(diagnostic.selected_ids(split, True)), 209)
        self.assertNotIn("forbidden-test", diagnostic.selected_ids(split, True))
        clips[0].fold = "test"
        with self.assertRaises(ValueError):
            diagnostic.selected_ids(split, False)

    def test_real_band_transform_compares_f32_hypothesis_without_changing_direct(self):
        # Seul l'import TimeNet est remplacé ; la vraie transformation et le vrai
        # décodeur PCM du projet sont exécutés sur un WAV synthétique sans label.
        def normalize(x):
            x = np.asarray(x, dtype=np.float64)
            centered = x - x.mean()
            return centered / np.sqrt(np.mean(centered ** 2))
        connector = SimpleNamespace(_normalise=normalize)
        spec = importlib.util.spec_from_file_location("parity_test_preprocessing", ROOT / "src/pipe/tslm/preprocessing.py")
        preprocessing = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"leakless_acoustic.connector": connector}):
            spec.loader.exec_module(preprocessing)
        values = np.random.default_rng(19).integers(-14000, 14000, size=8000).astype("<i2")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(values.tobytes())
        timef = normalize(values).astype(np.float32)
        cached = preprocessing.band_series(timef)
        original = cached.copy()
        with patch.dict(sys.modules, {"leakless_acoustic.connector": connector,
                                     "pipe.tslm.preprocessing": preprocessing}):
            routes, waveforms, comparisons = diagnostic.make_routes(output.getvalue(), cached, timef)
        np.testing.assert_array_equal(cached, original)
        np.testing.assert_array_equal(routes["roundtrip_f32"], routes["timef"])
        self.assertTrue(comparisons["cache_vs_roundtrip_f32"]["exact_values"])
        self.assertFalse(comparisons["cache_vs_direct"]["exact_values"])
        self.assertTrue(waveforms["rounded_f32_vs_timef"]["exact_values"])

    def test_reference_requires_same_inputs_and_distinct_processes(self):
        provenance = {"checkpoint_checksums_sha256": "checkpoint", "temporal_sha256": "temporal",
                      "clip_ids": ["v"], "scoring_spec": {"sum": True}, "cache_sha256": {"val": "cache"},
                      "hostname": "same-machine", "pid": 1}
        row = {"clip_id": "v", "input_sha256": "raw", "series": {"direct": "numeric-hash"},
               "scores": {"direct": [0.5]}}
        previous = {"provenance": provenance, "records": [row]}
        current = {"provenance": {**provenance, "pid": 2},
                   "records": [{**row, "scores": {"direct": [0.5000001]}}]}
        self.assertTrue(diagnostic.compare_reference(current, previous)["all_scores_within_atol"])
        current["records"][0]["scores"]["direct"] = [0.51]
        self.assertFalse(diagnostic.compare_reference(current, previous)["all_scores_within_atol"])
        current["records"][0]["series"] = {"direct": "different-hash"}
        with self.assertRaises(ValueError):
            diagnostic.compare_reference(current, previous)
        with self.assertRaises(ValueError):
            diagnostic.compare_reference(previous, previous)
        for values in ([float("nan")], [1.01], []):
            with self.assertRaises(ValueError):
                diagnostic.score_spread(values)


if __name__ == "__main__":
    unittest.main()
