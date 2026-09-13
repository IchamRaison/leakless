"""Contraste D2 : données synthétiques seulement, aucun WAV ni modèle Qwen."""
from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import diagnose_nonlinear as d2

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
FAKE_PARAMS = {**d2.PARAMS, "max_depth": None, "warm_start": False}


def fixture():
    rows = [{"clip_id": f"opaque-{group:02}-{i}", "group_id": f"group-{group:02}",
             "fold": "train", "label": "leak" if group >= 6 else "no_leak"}
            for group in range(12) for i in range(2)]
    matrix = np.zeros((len(rows), 256), dtype=np.float32)
    matrix[:, 0] = np.arange(len(rows))
    matrix[:, 1] = [.8 if r["label"] == "leak" else .2 for r in rows]
    folds = []
    for fold_id in range(3):
        heldout = {f"group-{group:02}" for group in range(12) if group % 3 == fold_id}
        folds.append({"fold_id": fold_id,
            "train_ids": [r["clip_id"] for r in rows if r["group_id"] not in heldout],
            "heldout_ids": [r["clip_id"] for r in rows if r["group_id"] in heldout],
            "heldout_groups": sorted(heldout)})
    probes = {}
    for name, n_features in (("TimeNet256", 256), ("C1_fixed", 9)):
        reports, predictions = [], []
        for fold in folds:
            selected = [r for r in rows if r["clip_id"] in fold["heldout_ids"]]
            scores = {r["clip_id"]: .7 if r["label"] == "leak" else .3 for r in selected}
            report = d2.campaign.metric_report(selected, scores, .5)
            report.update(fold_id=fold["fold_id"], n_fit_clips=len(fold["train_ids"]))
            reports.append(report)
            predictions.extend({"clip_id": cid, "inner_fold": fold["fold_id"],
                                "probability_leak": p} for cid, p in scores.items())
        probes[name] = {"n_features": n_features, "folds": reports,
            "mean_group_roc_auc": float(np.mean([r["group_roc_auc_full"] for r in reports])),
            "mean_clip_roc_auc": float(np.mean([r["clip_roc_auc_full"] for r in reports])),
            "predictions_by_fold": predictions}
    return rows, matrix, folds, probes


def fake_classifier(calls):
    class Classifier:
        classes_ = np.array([0, 1])
        n_iter_ = 200

        def get_params(self, deep=True):
            return copy.deepcopy(FAKE_PARAMS)

        def fit(self, X, y):
            calls.append((X.copy(), y.copy(), self))
            return self

        def predict_proba(self, X):
            return np.column_stack((1-X[:, 1], X[:, 1]))
    return Classifier()


def thread_stub():
    return patch.dict(sys.modules, {"threadpoolctl": SimpleNamespace(
        threadpool_limits=lambda **kwargs: nullcontext())})


def prepared(path, rows, matrix, extra=False):
    path.mkdir()
    # Les deux conteneurs diffèrent ; les seules séries conservées restent exactes.
    kwargs = {"unconsumed_amplitude": np.zeros((len(rows), 9))} if extra else {}
    np.savez(path / "train.npz", ids=[r["clip_id"] for r in rows],
             series=matrix.reshape(-1, 4, 64), preprocessing_version=d2.CANONICAL_VERSION, **kwargs)
    d2.write_json(path / "preparation.json", {"synthetic": True})


class D2Tests(unittest.TestCase):
    def test_inputs_bind_reference_cache_and_require_parity_before_computation(self):
        rows, matrix, folds, probes = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {name: root / name for name in d2.PATH_NAMES}
            prepared(paths["prepared"], rows, matrix, extra=True)
            prepared(paths["previous_prepared"], rows, matrix)
            paths["reference"].mkdir()
            paths["manifests"].mkdir()
            for name in ("split_v2.csv", "split_v2_audit.csv"):
                (paths["manifests"] / name).write_text("synthetic")
            d2.write_json(paths["parity_report"], {"synthetic": True})
            fold_document = {"folds": folds, "preprocessing_version": d2.CANONICAL_VERSION,
                "cache_train_sha256": d2.sha256_file(paths["previous_prepared"] / "train.npz"),
                "parity_report_sha256": "historical-gate"}
            d2.write_json(paths["reference"] / "folds.json", fold_document)
            d2.write_json(paths["reference"] / "probes.json", probes)
            fold_sha = d2.sha256_file(paths["reference"] / "folds.json")
            probe_sha = d2.sha256_file(paths["reference"] / "probes.json")
            metadata = {"schema": "pipe-train-diagnostic-v1", "source_fold": "train",
                "folds_sha256": fold_sha, "threshold": .5, "n_total_fits": 6,
                "n_probe_configurations": 1, "n_fixed_c1_configurations": 1,
                "parity_report_sha256": "historical-gate", "recipe": {
                    "C": 1., "max_iter": 5000, "solver": "lbfgs", "class_weight": None,
                    "scaler": "StandardScaler_fit_inner_train_only", "seed": 20260912}}
            d2.write_json(paths["reference"] / "metadata.json", metadata)
            with patch.object(d2, "FOLDS_SHA256", fold_sha), patch.object(d2, "PROBES_SHA256", probe_sha), \
                    patch.object(d2.split_loader, "load_split", return_value=object()), \
                    patch.object(d2.diagnostic, "train_rows", return_value=rows), \
                    patch.object(d2.campaign, "validate_folds") as validate, \
                    patch.object(d2.diagnostic, "verify_parity", return_value={}) as parity, \
                    patch.object(d2, "SOURCE_NAMES", ()), patch.object(d2, "estimator") as model:
                loaded, actual, actual_folds, _, identity = d2.inputs(paths)
                self.assertEqual(loaded, rows)
                self.assertEqual(actual.tobytes(), matrix.tobytes())
                self.assertEqual(actual_folds, folds)
                validate.assert_called_once_with(rows, fold_document)
                parity.assert_called_once_with(paths["parity_report"], paths["prepared"], d2.CANONICAL_VERSION)
                self.assertEqual(identity["matrix_shape"], [len(rows), 256])
                self.assertEqual(identity["matrix_dtype"], "float32")
                model.assert_not_called()
                with patch.object(d2.diagnostic, "verify_parity", side_effect=ValueError("gate non PASS")), \
                        patch.object(d2, "matrix_from_caches") as extract, self.assertRaisesRegex(ValueError, "gate"):
                    d2.inputs(paths)
                extract.assert_not_called()
                with patch.object(d2, "FOLDS_SHA256", "other"), self.assertRaisesRegex(ValueError, "Folds"):
                    d2.inputs(paths)
                with patch.object(d2, "PROBES_SHA256", "other"), self.assertRaisesRegex(ValueError, "sondes"):
                    d2.inputs(paths)
                # Même si les séries restaient lisibles, un ancien cache non épinglé est refusé.
                with (paths["previous_prepared"] / "train.npz").open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "Provenance"):
                    d2.inputs(paths)

    def test_exact_caches_row_major_float32_padding_and_train_only(self):
        rows, matrix, _, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            current, previous = Path(tmp) / "current", Path(tmp) / "previous"
            prepared(current, rows, matrix, extra=True)
            prepared(previous, rows, matrix)
            self.assertNotEqual(d2.sha256_file(current / "train.npz"), d2.sha256_file(previous / "train.npz"))
            with patch.object(d2, "load_cache", wraps=d2.load_cache) as load:
                actual = d2.matrix_from_caches(rows, current, previous)
            self.assertEqual([call.args[1] for call in load.call_args_list], ["train", "train"])
            self.assertEqual(actual.dtype, np.float32)
            self.assertEqual(actual.tobytes(), matrix.tobytes())
            # Une valeur altérée, un cast ou un pad ajouté ne devient pas une nouvelle représentation.
            for mode in ("value", "dtype", "padding"):
                changed = matrix.copy()
                if mode == "value":
                    changed[0, 0] = .1
                elif mode == "dtype":
                    changed = changed.astype(np.float64)
                else:
                    changed[0, 63] = 1
                cached = {r["clip_id"]: x.reshape(4, 64) for r, x in zip(rows, changed)}
                original = {r["clip_id"]: x.reshape(4, 64) for r, x in zip(rows, matrix)}
                with patch.object(d2, "load_cache", side_effect=[cached, original]), self.assertRaises(ValueError):
                    d2.matrix_from_caches(rows, current, previous)

    def test_references_verify_coverage_folds_targets_and_aggregates_without_fit(self):
        rows, matrix, folds, probes = fixture()
        with patch.object(d2.diagnostic, "fit_fixed_logreg", side_effect=AssertionError("Aucun refit")):
            result = d2.reference_reports(rows, matrix, folds, probes)
        self.assertEqual(set(result), {"TimeNet256", "C1_fixed"})
        self.assertEqual(len(result["TimeNet256"]), 3)
        cases = []
        bad = copy.deepcopy(probes)
        bad["TimeNet256"]["predictions_by_fold"][0]["clip_id"] = "official-test"
        cases.append((rows, bad))
        bad = copy.deepcopy(probes)
        bad["TimeNet256"]["predictions_by_fold"][0]["inner_fold"] = 1
        cases.append((rows, bad))
        bad = copy.deepcopy(probes)
        bad["C1_fixed"]["folds"][0]["clip_roc_auc_full"] = .4
        cases.append((rows, bad))
        bad = copy.deepcopy(probes)
        bad["C1_fixed"]["mean_group_roc_auc"] = .4
        cases.append((rows, bad))
        inverted = [{**r, "label": "leak" if r["label"] == "no_leak" else "no_leak"} for r in rows]
        cases.append((inverted, probes))
        for changed_rows, changed_probes in cases:
            with self.assertRaises(ValueError):
                d2.reference_reports(changed_rows, matrix, folds, changed_probes)

    def test_three_fresh_fits_exact_inner_training_only_and_no_threshold_selection(self):
        rows, matrix, folds, _ = fixture()
        calls, limits = [], []
        def limited(**kwargs):
            limits.append(kwargs)
            return nullcontext()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(sys.modules, {"threadpoolctl": SimpleNamespace(threadpool_limits=limited)}), \
                patch.object(d2, "estimator", side_effect=lambda: fake_classifier(calls)), \
                patch.object(d2.campaign.metrics, "pick_threshold", side_effect=AssertionError("Seuil interdit")):
            result = d2.fit_three(rows, matrix, folds, Path(tmp), FAKE_PARAMS)
            self.assertEqual(len(calls), 3)
            self.assertEqual(len({id(call[2]) for call in calls}), 3)
            self.assertEqual(limits, [{"limits": 1}] * 3)
            positions = {r["clip_id"]: i for i, r in enumerate(rows)}
            for fold, (X, y, _), record in zip(folds, calls, result):
                indices = [positions[cid] for cid in fold["train_ids"]]
                self.assertEqual(X.tobytes(), matrix[indices].tobytes())
                np.testing.assert_array_equal(y, [int(rows[i]["label"] == "leak") for i in indices])
                self.assertEqual(record["n_iter"], 200)
                predictions = d2.campaign.read_json(Path(tmp) / f"fold-{fold['fold_id']}-predictions.json")
                for partition, ids in (("fit", fold["train_ids"]), ("heldout", fold["heldout_ids"])):
                    self.assertEqual(sorted(p["clip_id"] for p in predictions if p["partition"] == partition),
                                     sorted(ids))
                    self.assertEqual(record["partitions"][partition]["threshold"], .5)

    def test_bad_matrix_groups_official_folds_or_last_partition_rejected_before_fit(self):
        rows, matrix, folds, _ = fixture()
        cases = [(rows, matrix.astype(np.float64), folds), (rows, matrix[:, :-1], folds)]
        for official in ("val", "test", "external"):
            cases.append(([{**rows[0], "fold": official}, *rows[1:]], matrix, folds))
        bad_folds = copy.deepcopy(folds)
        bad_folds[2]["heldout_ids"].append(bad_folds[2]["train_ids"][0])
        cases.append((rows, matrix, bad_folds))
        changed_rows = copy.deepcopy(rows)
        changed_rows[0]["group_id"] = rows[2]["group_id"]
        cases.append((changed_rows, matrix, folds))
        with thread_stub(), tempfile.TemporaryDirectory() as tmp:
            for args in cases:
                with patch.object(d2, "estimator") as factory, self.assertRaises(ValueError):
                    d2.fit_three(*args, Path(tmp), FAKE_PARAMS)
                factory.assert_not_called()

    def test_full_params_and_exact_iteration_budget_are_checked(self):
        rows, matrix, folds, _ = fixture()
        with thread_stub(), tempfile.TemporaryDirectory() as tmp:
            calls = []
            model = fake_classifier(calls)
            with patch.object(d2, "estimator", return_value=model), self.assertRaisesRegex(ValueError, "Paramètres"):
                d2.fit_three(rows, matrix, folds, Path(tmp), {**FAKE_PARAMS, "max_depth": 9})
            self.assertFalse(calls)
            model.n_iter_ = 199
            with patch.object(d2, "estimator", return_value=model), self.assertRaisesRegex(ValueError, "itérations"):
                d2.fit_three(rows, matrix, folds, Path(tmp), FAKE_PARAMS)
            self.assertEqual(len(calls), 1)
            self.assertFalse(list(Path(tmp).iterdir()))

    def test_nll_extremes_are_honest_and_json_finite(self):
        rows = [{"clip_id": "negative", "group_id": "g0", "fold": "train", "label": "no_leak"},
                {"clip_id": "positive", "group_id": "g1", "fold": "train", "label": "leak"}]
        correct = d2.report(rows, {"negative": 0., "positive": 1.})["binary_nll"]
        self.assertEqual(correct["mean"], 0.)
        self.assertEqual(correct["n_infinite"], 0)
        wrong = d2.report(rows, {"negative": 1., "positive": 0.})["binary_nll"]
        self.assertEqual(wrong["n_infinite"], 2)
        self.assertIsNone(wrong["mean"])
        json.dumps(wrong, allow_nan=False)
        tiny = np.nextafter(0., 1.)
        finite = d2.report(rows, {"negative": 0., "positive": tiny})["binary_nll"]
        self.assertEqual(finite["mean"], -np.log(tiny) / 2)
        for score in (float("nan"), float("inf"), -.1, 1.1):
            with self.assertRaises(ValueError):
                d2.report(rows, {"negative": 0., "positive": score})

    def test_comparisons_are_heldout_strict_positive_full_precision_and_no_pooled_auc(self):
        rows, matrix, folds, probes = fixture()
        refs = d2.reference_reports(rows, matrix, folds, probes)
        records = []
        for i, delta in enumerate((1e-9, 2e-9, 3e-9)):
            heldout = copy.deepcopy(refs["TimeNet256"][i])
            heldout["group_roc_auc_full"] = .5 + delta
            refs["TimeNet256"][i]["group_roc_auc_full"] = .5
            records.append({"fold_id": i, "partitions": {"heldout": heldout, "fit": {"unused": True}}})
        result = d2.comparisons(records, refs)
        self.assertTrue(result["primary"]["coherent_improvement_on_these_folds"])
        self.assertFalse(result["primary"]["statistical_significance_or_independent_confirmation"])
        records[2]["partitions"]["heldout"]["group_roc_auc_full"] = .5
        self.assertFalse(d2.comparisons(records, refs)["primary"]["coherent_improvement_on_these_folds"])
        records[0]["partitions"]["heldout"]["binary_nll"]["mean"] = None
        self.assertFalse(d2.comparisons(records, refs)["TimeNet256"]["folds"][0]["binary_nll_delta_defined"])

    def test_preregister_no_fit_freezes_changes_and_refuses_existing_output(self):
        rows, matrix, folds, probes = fixture()
        refs = d2.reference_reports(rows, matrix, folds, probes)
        identity = {"cache": "a", "source": "b"}
        inputs = (rows, matrix, folds, refs, identity)
        runtime = {"get_params": FAKE_PARAMS, "version": "synthetic"}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "d2"
            args = SimpleNamespace(output=output, code_revision="a" * 40,
                                   **{name: Path(tmp) / name for name in d2.PATH_NAMES})
            with patch.object(d2, "inputs", return_value=inputs), \
                    patch.object(d2, "runtime_identity", return_value=runtime), \
                    patch.object(d2, "fit_three", side_effect=AssertionError("Aucun fit à la préinscription")):
                receipt = d2.preregister(args)
                self.assertEqual(receipt["fits"], 0)
                d2.context(output)
                with self.assertRaises(FileExistsError):
                    d2.preregister(args)
                with patch.object(d2, "runtime_identity", return_value={**runtime, "version": "changed"}), \
                        self.assertRaisesRegex(ValueError, "runtime"):
                    d2.context(output)
                with patch.object(d2, "inputs", return_value=(*inputs[:-1], {**identity, "cache": "changed"})), \
                        self.assertRaisesRegex(ValueError, "Sources"):
                    d2.context(output)
                registration = d2.campaign.read_json(output / "preregistration.json")
                registration["params"]["max_iter"] = 201
                (output / "preregistration.json").write_text(json.dumps(registration))
                with self.assertRaisesRegex(ValueError, "Préinscription"):
                    d2.context(output)

    def test_run_seals_reuses_complete_and_preserves_incomplete_or_tampered_results(self):
        rows, matrix, folds, probes = fixture()
        refs = d2.reference_reports(rows, matrix, folds, probes)
        registration = {"runtime": {"get_params": FAKE_PARAMS}}
        ctx = (registration, rows, matrix, folds, refs)
        with tempfile.TemporaryDirectory() as tmp, thread_stub():
            output = Path(tmp)
            d2.write_json(output / "preregistration.json", {"synthetic": True})
            calls = []
            with patch.object(d2, "context", return_value=ctx), \
                    patch.object(d2, "estimator", side_effect=lambda: fake_classifier(calls)):
                complete = d2.run(output)
                self.assertEqual(len(calls), 3)
                self.assertEqual(d2.run(output), complete)
                self.assertEqual(len(calls), 3)
                summary = d2.campaign.read_json(output / "run/summary.json")
                self.assertEqual(summary["mean_auc_by_partition"]["heldout"]["group_roc_auc_full"], 1.)
                self.assertFalse(summary["pooled_auc"])
                (output / "run/fold-0-metrics.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "Artefacts modifiés"):
                    d2.run(output)
                self.assertEqual(len(calls), 3)
        with tempfile.TemporaryDirectory() as tmp, patch.object(d2, "context", return_value=ctx), \
                patch.object(d2, "fit_three") as fit:
            output = Path(tmp)
            d2.write_json(output / "preregistration.json", {"synthetic": True})
            (output / "run").mkdir()
            with self.assertRaisesRegex(ValueError, "incomplète"):
                d2.run(output)
            fit.assert_not_called()
            self.assertTrue((output / "run").is_dir())


@unittest.skipUnless(HAS_SKLEARN, "sklearn absent localement ; test synthétique dans .venv-repro CPU")
class D2SklearnTests(unittest.TestCase):
    def test_real_hgb_three_fits_200_iterations_and_runtime_stable_without_qwen(self):
        rows, matrix, folds, _ = fixture()
        model_before = sys.modules.get("pipe.tslm.model")
        runtime = d2.runtime_identity()
        self.assertEqual({k: runtime["get_params"][k] for k in d2.PARAMS}, d2.PARAMS)
        with tempfile.TemporaryDirectory() as tmp:
            result = d2.fit_three(rows, matrix, folds, Path(tmp), runtime["get_params"])
        self.assertEqual(len(result), 3)
        self.assertEqual([r["n_iter"] for r in result], [200, 200, 200])
        self.assertEqual(d2.runtime_identity(), runtime)
        self.assertIs(sys.modules.get("pipe.tslm.model"), model_before)


if __name__ == "__main__":
    unittest.main()
