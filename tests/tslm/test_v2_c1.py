"""Helpers C1 V2 : données synthétiques seulement, sans WAV ni GPU."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import v2_c1  # noqa: E402

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
GRID = (.01, .1, 1., 10.)  # Oracle du test ; production importe la constante officielle.


def fixture():
    rows = [{"clip_id": f"opaque-{g}-{i}", "group_id": f"g-{g}", "fold": "train",
             "label": "leak" if g >= 6 else "no_leak"}
            for g in range(12) for i in range(2)]
    matrix = np.zeros((len(rows), len(v2_c1.features.C1_NAMES)))
    matrix[:, 0] = np.arange(len(rows))
    matrix[:, 1] = [.8 if row["label"] == "leak" else .2 for row in rows]
    folds = []
    for fold_id in range(3):
        heldout = {f"g-{g}" for g in range(12) if g % 3 == fold_id}
        folds.append({"fold_id": fold_id,
            "train_ids": [r["clip_id"] for r in rows if r["group_id"] not in heldout],
            "heldout_ids": [r["clip_id"] for r in rows if r["group_id"] in heldout],
            "heldout_groups": sorted(heldout)})
    return rows, matrix, folds


class C1StructureTests(unittest.TestCase):
    def test_exactly_four_candidates_twelve_disjoint_fits_fixed_threshold_and_tie_break(self):
        rows, matrix, folds = fixture()
        calls = []
        def fake_fit(X, y, C):
            calls.append((X.copy(), y.copy(), C))
            return None, None
        with patch.object(v2_c1, "_official_grid", return_value=GRID), \
                patch.object(v2_c1, "fit_final", side_effect=fake_fit), \
                patch.object(v2_c1, "predict_probability", side_effect=lambda s, c, X: X[:, 1]), \
                patch.object(v2_c1.metrics, "pick_threshold", side_effect=AssertionError("Aucun seuil ajusté")):
            result = v2_c1.compare(rows, matrix, folds)
        self.assertEqual(result["selected_C"], .01)
        self.assertEqual(result["n_configs_compared"], 4)
        self.assertEqual(result["n_fits"], 12)
        self.assertEqual(len(calls), 12)
        positions = {r["clip_id"]: i for i, r in enumerate(rows)}
        for candidate_index, C in enumerate(GRID):
            candidate = result["candidates"][candidate_index]
            self.assertEqual(candidate["C"], C)
            self.assertEqual(candidate["mean_group_roc_auc"], 1.)
            for i, fold in enumerate(folds):
                fitted, labels, used_C = calls[candidate_index * 3 + i]
                expected = [positions[cid] for cid in fold["train_ids"]]
                np.testing.assert_array_equal(fitted, matrix[expected])
                np.testing.assert_array_equal(labels, [int(rows[j]["label"] == "leak") for j in expected])
                self.assertEqual(used_C, C)
                self.assertEqual(candidate["folds"][i]["threshold"], .5)
                self.assertEqual(candidate["folds"][i]["clip_level"]["tp"], 4)

    def test_selection_uses_full_precision_not_display_rounding(self):
        rows, matrix, folds = fixture()
        exact = []
        for group_auc in (.50000001, .50000002, .4, .3):
            exact.extend([group_auc, .5] * 3)
        with patch.object(v2_c1, "_official_grid", return_value=GRID), \
                patch.object(v2_c1, "fit_final", return_value=(None, None)), \
                patch.object(v2_c1, "predict_probability", side_effect=lambda s, c, X: X[:, 1]), \
                patch.object(v2_c1.metrics, "evaluate", side_effect=lambda *args: {"threshold": .5}), \
                patch.object(v2_c1.metrics, "roc_auc", side_effect=exact):
            result = v2_c1.compare(rows, matrix, folds)
        self.assertEqual(result["selected_C"], .1)
        self.assertAlmostEqual(result["candidates"][1]["mean_group_roc_auc"], .50000002)
        self.assertGreater(result["candidates"][1]["mean_group_roc_auc"],
                           result["candidates"][0]["mean_group_roc_auc"])

    def test_all_invalid_partitions_are_rejected_before_any_fit(self):
        rows, matrix, folds = fixture()
        cases = []
        bad = copy.deepcopy(folds)
        bad[2]["heldout_ids"].append(bad[2]["train_ids"][0])
        cases.append((rows, matrix, bad))
        bad = copy.deepcopy(folds)
        bad[2]["train_ids"].append(bad[2]["train_ids"][0])
        cases.append((rows, matrix, bad))
        bad = copy.deepcopy(folds)
        bad[0]["heldout_groups"] = []
        cases.append((rows, matrix, bad))
        bad = copy.deepcopy(folds)
        bad[0]["heldout_ids"][0] = "unknown"
        cases.append((rows, matrix, bad))
        cases.append((rows, matrix, [folds[0], folds[0], folds[2]]))
        for fold_name in ("val", "test", "external"):
            cases.append(([{**rows[0], "fold": fold_name}, *rows[1:]], matrix, folds))
        cases.append(([rows[0], rows[0], *rows[2:]], matrix, folds))
        cases.append(([{**rows[0], "group_id": "g-6"}, *rows[1:]], matrix, folds))
        bad_matrix = matrix.copy()
        bad_matrix[0, 0] = np.nan
        cases.extend(((rows, bad_matrix, folds), (rows, matrix[:, :2], folds),
                      (rows, matrix[:-1], folds), (rows, matrix, folds[:2])))
        for index, args in enumerate(cases):
            with self.subTest(case=index), patch.object(v2_c1, "fit_final") as fit:
                with self.assertRaises(ValueError):
                    v2_c1.compare(*args)
                fit.assert_not_called()

    def test_untrusted_or_corrupt_pickle_is_not_deserialized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pkl"
            path.write_bytes(b"not even a pickle")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            for kwargs in ({"expected_sha256": digest},
                           {"expected_sha256": digest, "trusted": 1},
                           {"expected_sha256": "bad", "trusted": True},
                           {"expected_sha256": "f" * 64, "trusted": True}):
                with self.subTest(kwargs=kwargs), patch.object(v2_c1.pickle, "loads") as loads:
                    with self.assertRaises(ValueError):
                        v2_c1.load_checkpoint(path, **kwargs)
                    loads.assert_not_called()


@unittest.skipUnless(HAS_SKLEARN, "sklearn absent localement ; exécuter dans .venv-repro CPU")
class C1SklearnTests(unittest.TestCase):
    def test_official_grid_real_fits_and_faithful_checkpoint_round_trip(self):
        from run_controls import C_GRID
        self.assertEqual(v2_c1._official_grid(), C_GRID)
        rows, X, folds = fixture()
        result = v2_c1.compare(rows, X, folds)
        self.assertEqual(result["selected_C"], .01)
        y = np.array([int(row["label"] == "leak") for row in rows])
        scaler, classifier = v2_c1.fit_final(X, y, result["selected_C"])
        np.testing.assert_array_equal(scaler.mean_, X.mean(axis=0))
        self.assertEqual(classifier.solver, "lbfgs")
        self.assertEqual(classifier.max_iter, 5000)
        self.assertEqual(classifier.random_state, 20260912)
        before = v2_c1.predict_probability(scaler, classifier, X)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c1.pkl"
            digest = v2_c1.save_checkpoint(path, scaler, classifier)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            loaded_scaler, loaded_classifier = v2_c1.load_checkpoint(
                path, expected_sha256=digest, trusted=True)
            np.testing.assert_array_equal(before, v2_c1.predict_probability(loaded_scaler, loaded_classifier, X))
            np.testing.assert_array_equal(scaler.scale_, loaded_scaler.scale_)
            np.testing.assert_array_equal(classifier.coef_, loaded_classifier.coef_)
            with self.assertRaises(FileExistsError):
                v2_c1.save_checkpoint(path, scaler, classifier)
            with patch.object(v2_c1, "_runtime", return_value={"different": "version"}):
                with self.assertRaisesRegex(ValueError, "versions"):
                    v2_c1.load_checkpoint(path, expected_sha256=digest, trusted=True)

    def test_scaler_is_fit_on_each_inner_train_only(self):
        rows, X, folds = fixture()
        original_fit = v2_c1.fit_final
        means = []
        def capture(fitted_X, y, C):
            scaler, classifier = original_fit(fitted_X, y, C)
            np.testing.assert_array_equal(scaler.mean_, fitted_X.mean(axis=0))
            means.append(scaler.mean_.copy())
            return scaler, classifier
        with patch.object(v2_c1, "fit_final", side_effect=capture):
            v2_c1.compare(rows, X, folds)
        self.assertEqual(len(means), 12)
        self.assertFalse(np.array_equal(means[0], X.mean(axis=0)))

    def test_convergence_warning_is_a_failure_and_grid_cannot_expand(self):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        rows, X, _ = fixture()
        y = np.array([int(row["label"] == "leak") for row in rows])
        def warn(*args, **kwargs):
            warnings.warn("synthetic non-convergence", ConvergenceWarning)
        with patch.object(LogisticRegression, "fit", side_effect=warn):
            with self.assertRaises(ConvergenceWarning):
                v2_c1.fit_final(X, y, 1.)
        for C in (.5, True, float("nan")):
            with self.assertRaises(ValueError):
                v2_c1.fit_final(X, y, C)


if __name__ == "__main__":
    unittest.main()
