"""Invariants CPU de campagne : python -m unittest discover -s tests/tslm -p test_campaign.py."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipe.tslm.campaign import (development_rows, diagnostic_rows, epoch_batches,
                                load_cache, select_candidate, validation_groups)


class CampaignChecks(unittest.TestCase):
    def test_validation_selection_uses_groups_and_earliest_exact_tie(self):
        rows = [{"clip_id": str(i), "group_id": group, "fold": "val", "label": label}
                for i, (group, label) in enumerate([
                    ("a", "leak"), ("a", "leak"), ("b", "no_leak")])]
        labels, scores = validation_groups(rows, {"0": 0.9, "1": 0.3, "2": 0.2})
        self.assertEqual(labels, [1, 0])
        np.testing.assert_allclose(scores, [0.6, 0.2])
        reports = [{"epoch": epoch, "validation_group_roc_auc": auc}
                   for epoch, auc in ((8, 0.9), (2, 0.8), (4, 0.9))]
        self.assertEqual(select_candidate(reports)["epoch"], 4)
        for bad in ([{**rows[0], "fold": "test"}, *rows[1:]],
                    [rows[0], {**rows[1], "label": "no_leak"}, rows[2]]):
            with self.assertRaises(ValueError):
                validation_groups(bad, {"0": 0.9, "1": 0.3, "2": 0.2})

    def test_split_and_cache_cannot_silently_include_test(self):
        rows = [{"clip_id": fold, "group_id": fold, "fold": fold, "label": "leak"}
                for fold in ("train", "val", "test")]
        self.assertEqual(set(development_rows(rows)), {"train", "val"})
        with self.assertRaises(ValueError):
            development_rows([rows[0], {**rows[1], "group_id": "train"}])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "train.npz", ids=np.array(["train"]), series=np.zeros((1, 4, 64)),
                     preprocessing_version="v")
            self.assertEqual(set(load_cache(root, "train", [rows[0]], "v")), {"train"})
            with self.assertRaises(ValueError):
                load_cache(root, "test", [rows[2]], "v")
            np.savez(root / "train.npz", ids=np.array(["test"]), series=np.zeros((1, 4, 64)),
                     preprocessing_version="v")
            with self.assertRaises(ValueError):
                load_cache(root, "train", [rows[0]], "v")

    def test_full_train_shuffle_and_fixed_diagnostic_groups(self):
        batches = epoch_batches(598, 8, 20260912, 1)
        self.assertEqual(len(batches), 75)
        self.assertEqual(len(batches[-1]), 6)
        flat = [i for batch in batches for i in batch]
        self.assertEqual(sorted(flat), list(range(598)))
        self.assertEqual(batches, epoch_batches(598, 8, 20260912, 1))
        self.assertNotEqual(batches, epoch_batches(598, 8, 20260912, 2))
        rows = [{"clip_id": f"c{i:02d}", "group_id": f"g{i // 2}", "fold": "val",
                 "label": "leak" if i < 8 else "no_leak"} for i in range(16)]
        chosen = diagnostic_rows(list(reversed(rows)))
        self.assertEqual(len(chosen), 8)
        self.assertEqual(len({r["group_id"] for r in chosen}), 8)
        self.assertEqual(chosen, diagnostic_rows(rows))


if __name__ == "__main__":
    unittest.main()
