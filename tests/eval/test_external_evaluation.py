"""Évaluation externe opt-in : fixtures seules, aucun modèle ni dataset réel."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import evaluate_predictions as evaluator  # noqa: E402
from harness import contract, metrics  # noqa: E402
from harness.split_loader import Clip, FROZEN_SPLIT_SHA256, Split  # noqa: E402


class ExternalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        manifest = self.root / "external_synthetic_v1.json"
        manifest.write_text('{"scope": "synthetic"}\n')
        self.split = Split(tuple(Clip(cid, label, "leak" if label else "no_leak", cid, "external")
                                 for cid, label in (("a", 1), ("b", 0), ("c", 1), ("d", 0))),
                           hashlib.sha256(manifest.read_bytes()).hexdigest(), manifest, {})
        self.identity = {"checkpoint_checksums_sha256": "a" * 64, "temporal_sha256": "b" * 64,
            "config_hash": "c" * 64, "model_version": "synthetic-v2-C",
            "preprocessing_version": "synthetic-canonical", "scoring_spec_sha256": "d" * 64,
            "source_sha256": {"model.py": "e" * 64}}
        common = {"fit_fold": "val", "threshold": .6500000000000001,
            "threshold_repr": repr(.6500000000000001), "rule": evaluator.EXTERNAL_THRESHOLD_RULE,
            "split_sha256": FROZEN_SPLIT_SHA256, "validation_predictions_sha256": "f" * 64,
            "threshold_method_sha256": hashlib.sha256(Path(metrics.__file__).read_bytes()).hexdigest()}
        self.tslm = {**common, "schema_version": "pipe-threshold-evidence-v1", "model_identity": self.identity}
        self.c1 = {**common, "schema": "pipe-v2-c1-threshold-v1", "checkpoint_sha256": "9" * 64,
                   "threshold": .4, "threshold_repr": repr(.4)}
        self.tslm_path = self.write_receipt("tslm-threshold.json", self.tslm)
        self.c1_path = self.write_receipt("c1-threshold.json", self.c1)

    def write_receipt(self, name, receipt):
        path = self.root / name
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        return path

    def make_run(self, name="tslm-T0", *, kind="tslm", transform="T0", evidence=None, extra=None):
        evidence = evidence or (self.tslm_path if kind == "tslm" else self.c1_path)
        metadata = {"transform": transform, "threshold_provenance_sha256": evaluator._sha256(evidence)}
        if kind == "tslm":
            metadata.update(model_identity=copy.deepcopy(self.identity),
                            **{key: self.identity[key] for key in (
                                "checkpoint_checksums_sha256", "temporal_sha256", "config_hash")})
        else:
            metadata["checkpoint_sha256"] = self.c1["checkpoint_sha256"]
        metadata.update(extra or {})
        directory = self.root / name
        contract.write_run(directory, run_id=name, model_name=kind, checkpoint="not-loaded",
            training_commit="a" * 40, split=self.split, threshold_rule="fixed validation receipt",
            probabilities={"a": .8, "b": .2, "c": .7, "d": .3}, extra=metadata,
            external_manifest_name=self.split.manifest_path.name, external_manifest_sha256=self.split.sha256)
        return contract.load_run(directory, self.split, folds=("external",),
            external_manifest_name=self.split.manifest_path.name, external_manifest_sha256=self.split.sha256)

    def test_external_and_all_stress_use_exact_fixed_thresholds_without_pick_threshold(self):
        runs, proofs = {}, {}
        with patch.object(metrics, "pick_threshold", side_effect=AssertionError("Seuil interdit ici")), \
                patch.object(metrics, "bootstrap_ci", return_value={"synthetic": True}) as bootstrap, \
                patch.object(metrics, "paired_bootstrap_delta", return_value={"n_clusters": 4}) as paired:
            for transform in ("T0", "T1", "T2", "T3"):
                run = self.make_run("tslm-" + transform, transform=transform)
                proof = evaluator.load_external_threshold(run, self.tslm_path)
                runs[run.run_id], proofs[run.run_id] = run, proof
                result = evaluator.evaluate_run(self.split, run, external_threshold=proof)
                self.assertEqual(set(result["folds"]), {"external"})
                self.assertIsNone(result["threshold_recomputed_on_val"])
                self.assertEqual(result["folds"]["external"]["threshold"], self.tslm["threshold"])
                self.assertEqual(bootstrap.call_args.args[3], self.tslm["threshold"])
                self.assertEqual(result["fixed_threshold_provenance"]["run_files_sha256"],
                                 {name: evaluator._sha256(run.source / name)
                                  for name in ("metadata.json", "predictions.csv")})
            c1 = self.make_run("c1-T0", kind="c1")
            runs[c1.run_id] = c1
            proofs[c1.run_id] = evaluator.load_external_threshold(c1, self.c1_path)
            result = evaluator.compare(self.split, runs, "tslm-T0", "c1-T0", fold="external", thresholds=proofs)
            self.assertEqual(result["fold"], "external")
            self.assertEqual(paired.call_args.args[4:], (self.tslm["threshold"], self.c1["threshold"]))
            for transform in ("T1", "T2", "T3"):
                evaluator.compare(self.split, runs, "tslm-" + transform, "tslm-T0", fold="external", thresholds=proofs)
                self.assertEqual(paired.call_args.args[4:], (self.tslm["threshold"], self.tslm["threshold"]))

    def test_missing_proof_cannot_fall_back_to_validation_or_default_threshold(self):
        run = self.make_run()
        with patch.object(metrics, "pick_threshold", side_effect=AssertionError("Aucun choix externe")):
            with self.assertRaises(ValueError):
                evaluator.evaluate_run(self.split, run)
            with self.assertRaises(ValueError):
                evaluator.evaluate_run(self.split, run, external_threshold=.5)
            with self.assertRaises(ValueError):
                evaluator.compare(self.split, {run.run_id: run}, run.run_id, run.run_id)
            with self.assertRaises(ValueError):
                evaluator.compare(self.split, {run.run_id: run}, run.run_id, run.run_id, fold="external")

    def test_evidence_checkpoint_method_fold_and_precision_mismatches_are_rejected(self):
        changes = [{"fit_fold": "external"}, {"fit_fold": "test"}, {"threshold_repr": "0.65"},
                   {"threshold": float("nan")}, {"threshold": True}, {"rule": "new rule"},
                   {"split_sha256": self.split.sha256}, {"threshold_method_sha256": "1" * 64},
                   {"validation_predictions_sha256": "short"},
                   {"model_identity": {**self.identity, "checkpoint_checksums_sha256": "8" * 64}}]
        for i, change in enumerate(changes):
            receipt = self.write_receipt(f"bad-{i}.json", {**self.tslm, **change})
            run = self.make_run(f"bad-{i}", evidence=receipt)
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluator.load_external_threshold(run, receipt)
        for key in ("checkpoint_checksums_sha256", "temporal_sha256", "config_hash"):
            run = self.make_run(key, extra={key: "7" * 64})
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluator.load_external_threshold(run, self.tslm_path)
        c1 = self.make_run("bad-c1", kind="c1", extra={"checkpoint_sha256": "7" * 64})
        with self.assertRaises(ValueError):
            evaluator.load_external_threshold(c1, self.c1_path)
        unknown = self.make_run("bad-transform", transform="T9")
        with self.assertRaises(ValueError):
            evaluator.load_external_threshold(unknown, self.tslm_path)

    def test_hashes_bind_receipt_and_run_files_and_in_memory_predictions(self):
        run = self.make_run()
        proof = evaluator.load_external_threshold(run, self.tslm_path)
        self.tslm_path.write_text("changed receipt")
        with self.assertRaisesRegex(ValueError, "empreinte"):
            evaluator.load_external_threshold(run, self.tslm_path)
        run.probabilities["a"] = .01
        with self.assertRaisesRegex(ValueError, "Objet de run"):
            evaluator.evaluate_run(self.split, run, external_threshold=proof)
        run.probabilities["a"] = .8
        with (run.source / "predictions.csv").open("a") as stream:
            stream.write("extra,.3\n")
        with self.assertRaisesRegex(ValueError, "run inchangé"):
            evaluator.evaluate_run(self.split, run, external_threshold=proof)

    def test_same_checkpoint_cannot_choose_another_threshold_for_stress(self):
        original = self.make_run()
        changed_receipt = self.write_receipt("changed-threshold.json",
            {**self.tslm, "threshold": .2, "threshold_repr": repr(.2)})
        stress = self.make_run("tslm-T1", transform="T1", evidence=changed_receipt)
        runs = {r.run_id: r for r in (original, stress)}
        proofs = {original.run_id: evaluator.load_external_threshold(original, self.tslm_path),
                  stress.run_id: evaluator.load_external_threshold(stress, changed_receipt)}
        with patch.object(metrics, "paired_bootstrap_delta") as paired:
            with self.assertRaisesRegex(ValueError, "même reçu"):
                evaluator.compare(self.split, runs, original.run_id, stress.run_id,
                                  fold="external", thresholds=proofs)
            paired.assert_not_called()

    def test_finite_threshold_outside_score_range_is_not_clipped(self):
        # pick_threshold inclut min(score)-1e-9 / max(score)+1e-9 parmi les candidats.
        receipt = self.write_receipt("edge.json", {**self.tslm, "threshold": -1e-9,
                                                   "threshold_repr": repr(-1e-9)})
        run = self.make_run(evidence=receipt)
        proof = evaluator.load_external_threshold(run, receipt)
        with patch.object(metrics, "bootstrap_ci", return_value={}):
            report = evaluator.evaluate_run(self.split, run, external_threshold=proof)
        self.assertEqual(report["folds"]["external"]["threshold"], -1e-9)

    def test_cli_explicit_manifest_sha_receipts_and_no_overwrite(self):
        run = self.make_run()
        output = self.root / "report.json"
        argv = ["evaluate", "--runs", str(run.source), "--external-manifest", str(self.split.manifest_path),
                "--external-manifest-sha256", self.split.sha256,
                "--threshold-evidence", f"{run.run_id}={self.tslm_path}", "--out", str(output)]
        loader = SimpleNamespace(load_external_manifest=lambda path: self.split)
        with patch.dict(sys.modules, {"external_manifest": loader}), \
                patch.object(metrics, "pick_threshold", side_effect=AssertionError("Seuil externe interdit")), \
                patch.object(metrics, "bootstrap_ci", return_value={}), \
                patch.object(evaluator.split_loader, "load_split", side_effect=AssertionError("Pas de loader V1")), \
                patch("builtins.print"), patch.object(sys, "argv", argv):
            evaluator.main()
            report = json.loads(output.read_text())
            self.assertEqual(report["split"]["filename"], self.split.manifest_path.name)
            self.assertEqual(set(report["runs"][run.run_id]["folds"]), {"external"})
            with self.assertRaises(SystemExit):
                evaluator.main()
        wrong = [*argv]
        wrong[wrong.index("--external-manifest-sha256") + 1] = "0" * 64
        wrong[wrong.index("--out") + 1] = str(self.root / "wrong.json")
        with patch.dict(sys.modules, {"external_manifest": loader}), patch.object(sys, "argv", wrong):
            with self.assertRaisesRegex(ValueError, "SHA explicitement"):
                evaluator.main()
        self.assertFalse((self.root / "wrong.json").exists())

    def test_v1_functions_still_refit_on_val_and_keep_original_output_shape(self):
        split = Split(tuple(Clip(f"{fold}-{label}", label, str(label), f"g-{fold}-{label}", fold)
                            for fold in ("val", "test") for label in (0, 1)),
                      FROZEN_SPLIT_SHA256, self.root / "split_v2.csv", {})
        run = contract.PredictionRun("legacy", "legacy", {},
            {c.clip_id: .8 if c.label else .2 for c in split.clips}, self.root)
        with patch.object(metrics, "pick_threshold", return_value=.7123456789) as choose, \
                patch.object(metrics, "bootstrap_ci", return_value={}), \
                patch.object(metrics, "paired_bootstrap_delta", return_value={"n_clusters": 2}) as paired:
            result = evaluator.evaluate_run(split, run)
            self.assertEqual(choose.call_count, 1)
            self.assertEqual(set(result["folds"]), {"val", "test"})
            self.assertEqual(result["threshold_recomputed_on_val"], round(.7123456789, 6))
            self.assertNotIn("fixed_threshold_provenance", result)
            self.assertNotIn("transform", result)
            evaluator.compare(split, {"a": run, "b": run}, "a", "b")
            self.assertEqual(choose.call_count, 3)
            self.assertEqual(paired.call_args.args[4:], (.7123456789, .7123456789))
            np.testing.assert_array_equal(choose.call_args.args[0], [0, 1])


if __name__ == "__main__":
    unittest.main()
