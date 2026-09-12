"""Contrôles CPU exécutables : python -m unittest discover -s tests/tslm."""
import io
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import Mock

import numpy as np

from pipe.tslm.prepare import safe_member_path
from pipe.tslm.preprocessing import decode_wav, measured_band, model_input, preprocess_audio, target_text


class PipelineChecks(unittest.TestCase):
    def test_predict_separates_generated_text_and_dsp(self):
        from pipe.tslm.predict import PredictionError, predict_audio
        from pipe.tslm.model import AcousticQwenSP
        stream = io.BytesIO()
        values = (2000 * np.sin(2 * np.pi * 1500 * np.arange(8000) / 8000)).astype("<i2")
        with wave.open(stream, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(values.tobytes())
        model = Mock()
        model.generate.return_value = ["leak; Greatest mean spectral energy: 0-1000 Hz."]
        metadata = {"model_version": "test", "max_new_tokens": 48}
        prediction = predict_audio(model, stream.getvalue(), metadata)
        self.assertEqual(prediction.prediction, "leak")
        self.assertIn("0-1000", prediction.description)
        self.assertEqual(prediction.observations[0].value, "1000-2000")
        forwarded = model.generate.call_args.args[0][0]
        self.assertEqual(set(forwarded), {"pre_prompt", "post_prompt", "time_series", "time_series_text"})
        model.generate.return_value = ["La fuite est à Paris."]
        invalid = predict_audio(model, stream.getvalue(), metadata)
        self.assertIsNone(invalid.prediction)
        self.assertTrue(invalid.abstained)
        self.assertEqual(invalid.abstention_reason, "invalid_output")
        with self.assertRaises(PredictionError):
            predict_audio(model, b"bad", metadata)
        # Le garde-fou est dans le vrai generate, avant tout calcul ou chargement de poids.
        unloaded = AcousticQwenSP.__new__(AcousticQwenSP)
        with self.assertRaises(ValueError):
            unloaded.generate([{**forwarded, "answer": "leak"}])

    def test_debug_selection_is_train_only_and_group_distinct(self):
        from pipe.tslm.train import debug_rows
        rows = [{"clip_id": f"c{i}", "group_id": f"g{i // 2}", "fold": fold, "label": label}
                for i, (fold, label) in enumerate([
                    ("test", "leak"), ("val", "no_leak"), ("train", "leak"),
                    ("train", "leak"), ("train", "no_leak"), ("train", "no_leak")])]
        selected = debug_rows(rows, 2)
        self.assertEqual({r["fold"] for r in selected}, {"train"})
        self.assertEqual(len({r["group_id"] for r in selected}), 2)
        with self.assertRaises(ValueError):
            debug_rows(rows, 4)

    def test_numeric_flow_and_target_separation(self):
        t = np.arange(8000) / 8000
        x = np.sin(2 * np.pi * 1500 * t)
        series = preprocess_audio(x, 8000)
        self.assertEqual(series.shape, (4, 64))
        self.assertEqual(series.dtype, np.float32)
        self.assertEqual(measured_band(series), "1000-2000")
        np.testing.assert_allclose(preprocess_audio(12 * x + 8, 8000), series, atol=1e-6)
        np.testing.assert_array_equal(series[:, 61:], 0)
        example = model_input(series)
        self.assertEqual(set(example), {"pre_prompt", "post_prompt", "time_series", "time_series_text"})
        before = example["pre_prompt"] + example["post_prompt"]
        for label in ("leak", "no_leak"):
            self.assertEqual(target_text(label, series), f"{label}; Greatest mean spectral energy: 1000-2000 Hz.")
        self.assertEqual(before, example["pre_prompt"] + example["post_prompt"])
        self.assertNotIn("answer", example)
        self.assertFalse(np.shares_memory(series, example["time_series"]))

    def test_invalid_inputs_and_wav_roundtrip(self):
        stream = io.BytesIO()
        values = np.arange(8000, dtype="<i2")
        with wave.open(stream, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(values.tobytes())
        np.testing.assert_array_equal(decode_wav(stream.getvalue()), values)
        for raw in (b"not a WAV", stream.getvalue()[:-2]):
            with self.assertRaises(ValueError):
                decode_wav(raw)
        for values, rate in ((np.zeros(7999), 8000), (np.zeros(8000), 16000),
                             (np.full(8000, np.nan), 8000)):
            with self.assertRaises(ValueError):
                preprocess_audio(values, rate)

    def test_archive_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(safe_member_path("folder/clip.wav", root), root / "folder/clip.wav")
            for name in ("../outside.wav", "/outside.wav", "C:\\outside.wav", "folder/../../x"):
                with self.assertRaises(ValueError):
                    safe_member_path(name, root)
            (root / "escape").symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                safe_member_path("escape/outside.wav", root)


if __name__ == "__main__":
    unittest.main()
