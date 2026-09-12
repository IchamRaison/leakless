"""Scores causaux analytiques, sans poids Qwen ni accès à un GPU/dataset."""
import math
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from opentslm.model.llm.TimeSeriesLLM import TimeSeriesLLM
from pipe.tslm.model import AcousticQwenSP, CLASS_CONTINUATIONS
from pipe.tslm.predict import PredictionError, Predictor
from pipe.tslm.preprocessing import model_input, preprocess_audio


class ToyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"leak;": [1, 3], "no_leak;": [2, 4, 3]}[text]

    def decode(self, ids):
        return {(1, 3): "leak;", (2, 4, 3): "no_leak;"}[tuple(ids)]


class ToyLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.arange(5).float()[:, None])
        self.probabilities = torch.tensor([
            [.05, .5, .25, .1, .1],       # Après le prompt : P(leak)=.5, P(no)=.25.
            [.15, .15, .15, .4, .15],    # Après leak : P(;)=.4.
            [.05, .05, .05, .05, .8],    # Après no : P(_leak)=.8.
            [.2, .2, .2, .2, .2],       # Après ; : description, jamais scorée.
            [.125, .125, .125, .5, .125],  # Après _leak : P(;)=.5.
        ])
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict):
        assert not use_cache and return_dict
        self.calls.append((inputs_embeds.clone(), attention_mask.clone()))
        # Une sentinelle 99 dans les prompts paddés doit avoir été retirée.
        return SimpleNamespace(logits=self.probabilities.log()[inputs_embeds[..., 0].long()])


class ToyModel(AcousticQwenSP):
    def __init__(self):
        TimeSeriesLLM.__init__(self, "cpu")
        self.tokenizer = ToyTokenizer()
        self.llm = ToyLLM()
        self.eval()

    def pad_and_apply_batch(self, batch):
        # Longueurs réelles différentes, mais même dernier token du prompt.
        return (torch.tensor([[[1.], [0.], [99.]], [[0.], [99.], [99.]]])[:len(batch)],
                torch.tensor([[1, 1, 0], [1, 0, 0]])[:len(batch)])


class ScoringChecks(unittest.TestCase):
    def test_causal_sum_unequal_lengths_padding_and_softmax(self):
        model = ToyModel()
        batch = [model_input(np.zeros((4, 64), dtype=np.float32)) for _ in range(2)]
        # P(leak;)=.5*.4=.2 ; P(no_leak;)=.25*.8*.5=.1, donc 2/3.
        got = model.score_class_logprobs(batch)
        expected = torch.tensor([[math.log(.2), math.log(.1)]] * 2, dtype=torch.float64)
        torch.testing.assert_close(got, expected)
        torch.testing.assert_close(torch.tensor(model.score_probability_leak(batch)),
                                   torch.tensor([2 / 3, 2 / 3]))
        spec = model.scoring_spec()
        self.assertEqual(spec["class_continuations"], list(CLASS_CONTINUATIONS))
        self.assertEqual(spec["class_token_counts"], [2, 3])
        self.assertFalse(spec["length_normalization"])
        self.assertFalse(spec["includes_eos"])
        self.assertFalse(spec["includes_description"])
        for index, (embeddings, mask) in enumerate(model.llm.calls):
            prefix_length = 2 if index % 2 == 0 else 1
            torch.testing.assert_close(embeddings[0, :prefix_length], embeddings[1, :prefix_length])
            self.assertEqual(mask[0].tolist(), [1] * (prefix_length + 2) + [0])
            self.assertEqual(mask[1].tolist(), [1] * (prefix_length + 3))
        # Modifier uniquement la distribution après ';' ne doit rien changer.
        model.llm.probabilities[3] = torch.tensor([.96, .01, .01, .01, .01])
        torch.testing.assert_close(model.score_class_logprobs(batch), expected)
        # L'exponentielle naïve de ces deux log-probas sous-déborderait à zéro.
        with patch.object(model, "score_class_logprobs",
                          return_value=torch.tensor([[-1000., -1001.]], dtype=torch.float64)):
            self.assertAlmostEqual(model.score_probability_leak(batch[:1])[0], 1 / (1 + math.exp(-1)))

    def test_rejects_metadata_training_mode_and_nonfinite_scores(self):
        model = ToyModel()
        example = model_input(np.zeros((4, 64), dtype=np.float32))
        for field in ("answer", "label", "clip_id", "fold", "pressure"):
            with self.assertRaises(ValueError):
                model.score_probability_leak([{**example, field: "forbidden"}])
        with self.assertRaises(ValueError):
            model.score_probability_leak([])
        model.train()
        with self.assertRaises(ValueError):
            model.score_probability_leak([example])
        model.eval()
        model.llm.probabilities[0, 1] = float("nan")
        with self.assertRaises(ValueError):
            model.score_probability_leak([example])

    def test_numeric_predictor_keeps_float_audio_and_preprocessing(self):
        predictor = Predictor.__new__(Predictor)
        predictor._lock = threading.Lock()
        predictor.model = Mock()
        predictor.model.score_probability_leak.return_value = [.123456789]
        # Valeurs hors PCM16 : surtout aucune quantification/clipping pour T3.
        waveform = 80000.5 * np.sin(2 * np.pi * 1337 * np.arange(8000) / 8000)
        expected = preprocess_audio(waveform, 8000)
        with patch("pipe.tslm.predict.decode_wav", side_effect=AssertionError("Pas de WAV")):
            self.assertEqual(predictor.score_waveform(waveform), .123456789)
        forwarded = predictor.model.score_probability_leak.call_args.args[0][0]
        self.assertEqual(set(forwarded), {"pre_prompt", "post_prompt", "time_series", "time_series_text"})
        np.testing.assert_array_equal(forwarded["time_series"].numpy(), expected)
        predictor.model.generate.assert_not_called()
        for value in (float("nan"), -0.1, 1.1):
            predictor.model.score_probability_leak.return_value = [value]
            with self.assertRaises(PredictionError):
                predictor.score_series(expected)
            self.assertFalse(predictor._lock.locked())
        for waveform in (np.zeros(8000), np.full(8000, np.nan), np.zeros(7999)):
            with self.assertRaises(PredictionError):
                predictor.score_waveform(waveform)


if __name__ == "__main__":
    unittest.main()
