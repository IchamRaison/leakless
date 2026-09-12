"""Contrôles CPU exécutables : python -m unittest discover -s tests/tslm."""
import io
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from pipe.tslm.prepare import safe_member_path
from pipe.tslm.preprocessing import decode_wav, measured_band, model_input, preprocess_audio, target_text


class PipelineChecks(unittest.TestCase):
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
