"""Contrat amplitude et packing causal sur fixtures, sans poids Qwen ni GPU."""
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
sys.path.insert(0, str(ROOT / "src"))
from harness.features import C1_NAMES, c1_envelope


def normalize(values):
    x = np.asarray(values).astype(np.float64)
    x -= x.mean()
    rms = float(np.sqrt(np.mean(x*x)))
    return x if rms == 0 else x/rms


def preprocessing_fixture():
    spec = importlib.util.spec_from_file_location("amplitude_preprocessing_fixture", ROOT / "src/pipe/tslm/preprocessing.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"leakless_acoustic.connector": SimpleNamespace(_normalise=normalize)}):
        spec.loader.exec_module(module)
    return module


def prediction_fixture(preprocessing):
    spec = importlib.util.spec_from_file_location("amplitude_prediction_fixture", ROOT / "src/pipe/tslm/predict.py")
    module = importlib.util.module_from_spec(spec)
    modules = {"torch": SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=MemoryError)),
        "opentslm.time_series_datasets.util": SimpleNamespace(
            extend_time_series_to_match_patch_size_and_aggregate=lambda batch, normalize: batch),
        "pipe.tslm.model": SimpleNamespace(AcousticQwenSP=Mock()),
        "pipe.tslm.preprocessing": preprocessing,
        "pipe.contracts": SimpleNamespace(Observation=lambda **kw: kw, Prediction=lambda **kw: SimpleNamespace(**kw))}
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class AmplitudeCPUChecks(unittest.TestCase):
    def setUp(self):
        self.p = preprocessing_fixture()
        self.prediction = prediction_fixture(self.p)
        self.waveform = np.random.default_rng(20260912).normal(size=8000) * 80000.5
        self.series = self.p.preprocess_audio(self.waveform, 8000, version=self.p.CANONICAL_VERSION)
        self.values = normalize(self.waveform).astype(np.float32)
        self.amplitude = self.p.amplitude_from_normalized(self.values)
        self.c_metadata = {"amplitude_evidence": True, "amplitude_evidence_version": self.p.AMPLITUDE_EVIDENCE_VERSION,
            "single_clip_acoustic_encoding": True, "preprocessing_version": self.p.CANONICAL_VERSION,
            "model_version": "fixture-C", "max_new_tokens": 48}

    def test_c1_is_reused_exactly_on_float32_timef_values(self):
        np.testing.assert_array_equal(self.amplitude, c1_envelope(self.values.astype(np.float64)))
        np.testing.assert_array_equal(self.amplitude, self.p.amplitude_features(self.waveform, 8000))
        self.assertEqual(self.amplitude.shape, (9,))
        self.assertEqual(self.amplitude.dtype, np.float64)
        for values in (self.values.astype(np.float64), self.values[:-1], np.full(8000, np.nan, dtype=np.float32)):
            with self.assertRaises(ValueError):
                self.p.amplitude_from_normalized(values)
        with self.assertRaises(ValueError):
            self.p.amplitude_features(self.waveform, 16000)

    def test_text_is_six_significant_digits_and_keeps_four_numeric_channels(self):
        features = np.array([1.23456789, -0.123456789, 1e-9, 4, 5, 6, 7, 8, 9], dtype=np.float64)
        rendered = self.p.amplitude_text(features)
        expected = self.p.AMPLITUDE_TEXT_PREFIX + "; ".join(f"{name}={value:.6g}" for name, value in zip(C1_NAMES, features))
        self.assertEqual(rendered, expected)
        example = self.p.model_input(self.series, amplitude_features=features)
        self.assertEqual(set(example), {"pre_prompt", "post_prompt", "time_series", "time_series_text"})
        self.assertEqual(example["pre_prompt"], self.p.PRE_PROMPT + rendered)
        self.assertEqual(example["post_prompt"], self.p.POST_PROMPT)
        self.assertEqual(len(example["time_series_text"]), 4)
        np.testing.assert_array_equal(example["time_series"], self.series)
        self.assertIsNot(example["time_series"], self.series)
        self.assertEqual(self.p.model_input(self.series)["pre_prompt"], self.p.PRE_PROMPT)
        for bad in (np.ones(8), np.ones((9, 1)), np.full(9, np.nan), np.ones(9, dtype=bool), ["1"] * 9):
            with self.assertRaises(ValueError):
                self.p.amplitude_text(bad)

    def test_metadata_rejects_missing_cache_features_and_additions_to_a(self):
        helper = self.prediction.model_input_for_model
        with self.assertRaises(ValueError):
            helper(self.series, self.c_metadata)
        with self.assertRaises(ValueError):
            helper(self.series, {}, amplitude_features=self.amplitude)
        self.assertEqual(helper(self.series, {})["pre_prompt"], self.p.PRE_PROMPT)
        for changes in ({"amplitude_evidence": 1}, {"amplitude_evidence_version": "wrong"},
                        {"single_clip_acoustic_encoding": False}, {"preprocessing_version": self.p.VERSION}):
            with self.assertRaises(ValueError):
                helper(self.series, {**self.c_metadata, **changes}, amplitude_features=self.amplitude)
        with self.assertRaises(ValueError):
            helper(self.series, {"amplitude_evidence": False, "amplitude_evidence_version": self.p.AMPLITUDE_EVIDENCE_VERSION})

    def test_waveform_cache_and_text_prediction_use_same_nine_features(self):
        predictor = self.prediction.Predictor.__new__(self.prediction.Predictor)
        predictor.metadata = self.c_metadata
        predictor._lock = threading.Lock()
        predictor.model = Mock()
        predictor.model.score_probability_leak.return_value = [.25]
        predictor.model.generate.return_value = ["no_leak; Greatest mean spectral energy: 0-1000 Hz."]
        self.assertEqual(predictor.score_waveform(self.waveform), .25)
        waveform_example = predictor.model.score_probability_leak.call_args.args[0][0]
        self.assertEqual(predictor.score_series(self.series, amplitude_features=self.amplitude), .25)
        cached_example = predictor.model.score_probability_leak.call_args.args[0][0]
        self.assertEqual(cached_example["pre_prompt"], waveform_example["pre_prompt"])
        np.testing.assert_array_equal(cached_example["time_series"], waveform_example["time_series"])
        with patch.object(self.prediction, "decode_wav", return_value=self.waveform):
            self.prediction.predict_audio(predictor.model, b"synthetic WAV", self.c_metadata)
        generated_example = predictor.model.generate.call_args.args[0][0]
        self.assertEqual(generated_example["pre_prompt"], cached_example["pre_prompt"])
        np.testing.assert_array_equal(generated_example["time_series"], cached_example["time_series"])
        with self.assertRaises(self.prediction.PredictionError) as caught:
            predictor.score_series(self.series)
        self.assertEqual(caught.exception.code, "unsupported_audio")
        self.assertFalse(predictor._lock.locked())


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("opentslm"),
                     "Tests PyTorch CPU à exécuter dans le runtime ML de référence")
class AmplitudeTorchChecks(unittest.TestCase):
    def setUp(self):
        import torch
        from opentslm.model.llm.TimeSeriesLLM import TimeSeriesLLM
        from pipe.tslm.model import AcousticQwenSP
        self.torch = torch
        model = AcousticQwenSP.__new__(AcousticQwenSP)
        TimeSeriesLLM.__init__(model, "cpu")
        self.weight = torch.tensor(2., requires_grad=True)
        model.pad_and_apply_batch = lambda batch: (
            self.weight * torch.tensor([[[1.], [2.], [99.]], [[3.], [4.], [5.]]]),
            torch.tensor([[1, 1, 0], [1, 1, 1]]))

        class Tokenizer:
            pad_token_id = 0
            def __call__(self, answers, **kwargs):
                self.answers = answers
                return SimpleNamespace(input_ids=torch.tensor([[6, 7, 0], [8, 9, 7]]),
                                       attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]))

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding.from_pretrained(torch.arange(10).float()[:, None])
                self.kwargs = None
            def get_input_embeddings(self):
                return self.embedding
            def forward(self, **kwargs):
                self.kwargs = kwargs
                # Vraie CE causale ; gradients vers le dernier token de prompt.
                scale = torch.linspace(-.2, .2, 10)
                logits = kwargs["inputs_embeds"] * scale
                labels = kwargs["labels"][:, 1:]
                loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 10), labels.reshape(-1), ignore_index=-100)
                return SimpleNamespace(loss=loss, logits=logits)
        model.tokenizer, model.llm = Tokenizer(), Decoder()
        model.get_eos_token = lambda: "<eos>"
        self.model = model
        self.batch = [{"answer": "leak"}, {"answer": "no_leak"}]

    def test_opt_in_packing_first_class_after_true_prompt_and_only_final_padding(self):
        self.model.single_clip_acoustic_encoding = True
        loss = self.model.compute_loss(self.batch)
        observed = self.model.llm.kwargs
        self.assertEqual(observed["labels"].tolist(), [[-100, -100, 6, 7, -100, -100], [-100, -100, -100, 8, 9, 7]])
        self.assertEqual(observed["attention_mask"].tolist(), [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
        self.assertEqual(observed["inputs_embeds"][0, :, 0].tolist(), [2, 4, 6, 7, 0, 0])
        # Le label 6 de position 2 est prédit par la dernière position réelle 1.
        self.assertEqual(observed["labels"][0, 2].item(), 6)
        self.assertEqual(observed["inputs_embeds"][0, 1, 0].item(), 4)
        loss.backward()
        self.assertIsNotNone(self.weight.grad)
        self.assertTrue(self.torch.isfinite(self.weight.grad))
        self.assertNotEqual(self.weight.grad.item(), 0)

    def test_legacy_branch_and_opt_in_default_are_unchanged(self):
        self.assertFalse(self.model.single_clip_acoustic_encoding)
        self.assertFalse(self.model.amplitude_evidence)
        self.model.compute_loss(self.batch)
        observed = self.model.llm.kwargs
        self.assertEqual(observed["labels"].tolist(), [[-100, -100, -100, 6, 7, -100], [-100, -100, -100, 8, 9, 7]])
        self.assertEqual(observed["attention_mask"][0].tolist(), [1, 1, 0, 1, 1, 0])
        self.assertEqual(observed["inputs_embeds"][0, 2, 0].item(), 198)

    def test_c_scoring_declares_amplitude_but_retains_complete_class_tokens(self):
        from pipe.tslm.model import AcousticQwenSP, AMPLITUDE_SCORING_VERSION
        from pipe.tslm.preprocessing import amplitude_spec, model_input
        tokens = {"leak;": [1, 2], "no_leak;": [3, 4, 2]}
        self.model.tokenizer = SimpleNamespace(
            encode=lambda text, **kw: tokens[text], decode=lambda ids: next(k for k, v in tokens.items() if v == ids))
        self.model.single_clip_acoustic_encoding = True
        before = self.model.scoring_spec()
        self.model.amplitude_evidence = True
        after = self.model.scoring_spec()
        self.assertEqual(after["version"], AMPLITUDE_SCORING_VERSION)
        self.assertEqual(after["amplitude_evidence"], amplitude_spec())
        self.assertEqual({k: v for k, v in before.items() if k != "version"},
                         {k: v for k, v in after.items() if k not in ("version", "amplitude_evidence")})
        with self.assertRaises(ValueError):
            self.model._validate_inference_batch([model_input(np.zeros((4, 64), dtype=np.float32))])
        with self.assertRaises(ValueError):
            AcousticQwenSP("must-not-open", device="cpu", amplitude_evidence=True)
        with self.assertRaises(ValueError):
            AcousticQwenSP("must-not-open", device="cpu", amplitude_evidence=1)


if __name__ == "__main__":
    unittest.main()
