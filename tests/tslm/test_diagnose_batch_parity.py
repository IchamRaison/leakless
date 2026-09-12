"""Intervention et alignements synthétiques CPU, sans PyTorch ni données réelles."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import diagnose_batch_parity as diagnostic


def numpy_pad(sequences, *, batch_first, padding_value):
    assert batch_first
    result = np.full((len(sequences), max(len(s) for s in sequences), *sequences[0].shape[1:]),
                     padding_value, dtype=sequences[0].dtype)
    for i, sequence in enumerate(sequences):
        result[i, :len(sequence)] = sequence
    return result


class BatchParityChecks(unittest.TestCase):
    def test_contexts_preserve_real_neighbours_order_and_truncated_batches(self):
        ids = ["a", "b", "c", "d", "e"]
        found = {r["name"]: r for r in diagnostic.contexts(ids, "d")}
        self.assertEqual(found["batch_2_forward"]["clip_ids"], ["c", "d"])
        self.assertEqual(found["batch_2_forward"]["target_position"], 1)
        self.assertEqual(found["batch_4_forward"]["clip_ids"], ["a", "b", "c", "d"])
        self.assertEqual(found["batch_4_reversed"]["clip_ids"], ["e", "d", "c", "b"])
        last = {r["name"]: r for r in diagnostic.contexts(ids, "e")}
        self.assertEqual(last["batch_4_forward"]["clip_ids"], ["e"])
        self.assertEqual(last["batch_4_forward"]["requested_batch_size"], 4)
        self.assertEqual(last["batch_1_forward"]["clip_ids"], ["e"])
        for bad_ids, target in ((["a", "a"], "a"), (["a", "b"], "missing")):
            with self.assertRaises(ValueError):
                diagnostic.contexts(bad_ids, target)

    def test_temporary_padding_preserves_prompt_values_and_restores_method(self):
        class Model:
            def __init__(self):
                self.calls = []
            def pad_and_apply_batch(self, batch):
                self.calls.append(batch)
                if batch[0] == "fail":
                    raise RuntimeError("original failure")
                if len(batch) != 1:
                    raise RuntimeError("fixture requires singleton")
                data = np.asarray(batch[0], dtype=np.float32).reshape(1, -1, 1)
                # Un padding masqué contient un nombre non nul : il ne doit pas devenir prompt.
                values = np.concatenate((data, np.full((1, 1, 1), 99, dtype=np.float32)), axis=1)
                mask = np.array([[1] * data.shape[1] + [0]], dtype=np.int64)
                return values, mask
        model = Model()
        self.assertNotIn("pad_and_apply_batch", vars(model))
        with diagnostic.per_clip_padding(model, numpy_pad):
            data, mask = model.pad_and_apply_batch([[12, 47], [85]])
            np.testing.assert_array_equal(data, [[[12], [47]], [[85], [0]]])
            np.testing.assert_array_equal(mask, [[1, 1], [1, 0]])
            self.assertEqual(data.dtype, np.float32)
            self.assertEqual(mask.dtype, np.int64)
            self.assertEqual(model.calls, [[[12, 47]], [[85]]])
        self.assertNotIn("pad_and_apply_batch", vars(model))
        with self.assertRaisesRegex(RuntimeError, "original failure"):
            with diagnostic.per_clip_padding(model, numpy_pad):
                model.pad_and_apply_batch(["fail"])
        self.assertNotIn("pad_and_apply_batch", vars(model))
        with self.assertRaises(ValueError):
            with diagnostic.per_clip_padding(model, numpy_pad):
                model.pad_and_apply_batch([])
        self.assertNotIn("pad_and_apply_batch", vars(model))

    def test_existing_instance_override_is_restored_exactly(self):
        class Model:
            pass
        model = Model()
        original = lambda batch: (np.ones((1, 1, 2)), np.ones((1, 1)))
        model.pad_and_apply_batch = original
        with diagnostic.per_clip_padding(model, numpy_pad):
            model.pad_and_apply_batch([{}, {}])
        self.assertIs(model.pad_and_apply_batch, original)

    def test_component_slice_aligns_bulk_and_individual_calls(self):
        data = np.arange(3 * 4 * 5).reshape(12, 5).astype(np.float32)
        bulk = [(data, data * 2)]
        independent = [(data[i:i+4], data[i:i+4] * 2) for i in range(0, 12, 4)]
        for position in range(3):
            for side in (0, 1):
                expected = data[position*4:(position+1)*4] * (side + 1)
                np.testing.assert_array_equal(diagnostic.sample_component(bulk, side, position, 3, (4, 5)), expected)
                np.testing.assert_array_equal(diagnostic.sample_component(independent, side, position, 3, (4, 5)), expected)
        with self.assertRaises(ValueError):
            diagnostic.sample_component(bulk, 0, 0, 3, (4, 6))
        result = diagnostic.stage_differences({"input": data, "output": data},
                                               {"input": data.copy(), "output": data + 1})
        self.assertEqual(result["first_exact_divergence"], "output")


if __name__ == "__main__":
    unittest.main()
