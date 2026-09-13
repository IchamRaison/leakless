"""Rendu V2 depuis fixtures JSON ; aucun calcul d'évaluation ni accès audio."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import build_final_report as report


class V2ReportChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.campaign = self.root / "campaign"
        (self.campaign / "selection").mkdir(parents=True)
        registration = {"schema": "pipe-v2-campaign-v1", "code_revision": "a" * 40,
            "variants": ["A", "C"], "diagnostic_threshold": .5, "counts": {"fixture": True},
            "selection_rule": "fixture_mean_three_group_auc", "folds": [{"fold_id": i} for i in range(3)]}
        self.save(self.campaign / "preregistration.json", registration)
        self.prereg = report._sha(self.campaign / "preregistration.json")
        self.selection = {"selected_variant": "C", "selected_C1_C": .01,
            "preregistration_sha256": self.prereg, "selection_rule": "fixture_mean_three_group_auc",
            "counts": registration["counts"], "independent_final_evaluation": False,
            "pooled_out_of_fold_auc_calculated": False,
            "decision_note": "Examiner les erreurs avant tout refit",
            "candidates": [{"variant": name, "mean_group_roc_auc": .811119 if name == "A" else .822223,
                "mean_clip_roc_auc": .731117, "fold_metrics": [self.block(i) for i in range(3)]}
                for name in ("A", "C")],
            "c1_candidates": [{"C": c, "mean_group_roc_auc": .933339, "mean_clip_roc_auc": .833337,
                "folds": [{**self.block(i), "fold_id": i} for i in range(3)]} for c in (.01, .1)]}
        self.save(self.campaign / "selection/selection.json", self.selection)
        self.args = SimpleNamespace(v2_campaign=self.campaign, v2_evaluation=None, v2_audits=None,
                                    out=self.root / "report", code_revision="b" * 40)

    def save(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def block(self, index=0, threshold=.5):
        level = {"roc_auc": .712345+index/100, "pr_auc": .698765, "macro_f1": .634567, "brier": .234567,
                 "tp": 3, "tn": 4, "fp": 1+index, "fn": 2}
        return {"n_clips": 10+index, "n_clusters": 4, "n_clusters_leak": 2, "n_clusters_non_leak": 2,
                "threshold": threshold, "clip_level": level, "cluster_level": dict(level),
                "bootstrap_ci95": {"clip_roc_auc": {"ci95_low": .123456, "ci95_high": .987654}}}

    def render(self):
        forbidden = AssertionError("Le générateur ne doit pas charger les données ou recalculer")
        with patch.object(report, "evaluate_run", side_effect=forbidden), \
                patch.object(report, "compare", side_effect=forbidden), \
                patch.object(report, "vectors", side_effect=forbidden), \
                patch.object(report.split_loader, "load_split", side_effect=forbidden), \
                patch.object(report.metrics, "pick_threshold", side_effect=forbidden), \
                patch.object(report.metrics, "evaluate", side_effect=forbidden), \
                patch.object(report.metrics, "bootstrap_ci", side_effect=forbidden), patch("builtins.print"):
            provenance = report.build_v2(self.args)
        return (Path(self.args.out) / "FINAL_EVALUATION_V2.md").read_text(), provenance

    def test_cv_only_uses_stored_means_all_three_folds_and_one_c1_without_engine(self):
        text, provenance = self.render()
        self.assertIn("0.811119", text)
        self.assertIn("0.822223", text)
        self.assertIn("C1 (C=0.01)", text)
        self.assertNotIn("C1 (C=0.1)", text)
        for candidate in ("A", "C", "C1 (C=0.01)"):
            for fold in range(3):
                self.assertIn(f"{candidate}-fold{fold}", text)
        self.assertIn("au seuil 0,5 fixé pour le diagnostic CV", text)
        self.assertIn("| clip | 5 | 5 | 1 | 2 |", text)
        self.assertIn("aucun JSON d'évaluation externe fourni", text)
        self.assertIn("Non vérifié : aucun audit textuel fourni", text)
        self.assertIn("Événement, contexte et utilité du TSLM — non démontrés", text)
        self.assertFalse(provenance["metrics_recalculated"])
        self.assertFalse(provenance["audio_or_split_loaded"])
        self.assertEqual(len(provenance["source_sha256"]), 2)
        with self.assertRaises(FileExistsError):
            self.render()

    def evaluation(self):
        threshold = .6123456789012345
        def run(name, transform):
            return {"run_id": name, "model_name": "fixture TSLM", "training_commit": "a" * 40,
                "transform": transform, "threshold_recomputed_on_val": None,
                "fixed_threshold_provenance": {"fit_fold": "val", "threshold": threshold,
                    "schema": "pipe-threshold-evidence-v1", "rule": "fixture-validation-rule",
                    "threshold_repr": repr(threshold), "threshold_provenance_sha256": "d" * 64,
                    "validation_split_sha256": report.split_loader.FROZEN_SPLIT_SHA256,
                    "validation_predictions_sha256": "b" * 64, "threshold_method_sha256": "c" * 64,
                    "checkpoint_kind": "tslm", "checkpoint_sha256": "e" * 64,
                    "run_files_sha256": {"metadata.json": "f" * 64, "predictions.csv": "0" * 64}},
                "folds": {"external": self.block(threshold=threshold)}}
        return {"split": {"filename": report.V2_EXTERNAL_NAME, "sha256": report.V2_EXTERNAL_SHA256,
                "n_clips": 3600, "n_clusters": 60},
            "bootstrap": {"draws": 2000, "seed": 20260912, "unit": "cluster"},
            "runs": {"fixture-T0": run("fixture-T0", "T0"), "fixture-T1": run("fixture-T1", "T1")},
            "paired_comparisons": {"fixture-T0_vs_fixture-T1": {"cluster_roc_auc": {
                "delta_observe": .123, "ci95_low": -.045, "ci95_high": .234, "lecture": "inconclusive"}}}}

    def test_external_figures_intervals_full_threshold_and_stress_from_existing_json(self):
        self.args.v2_evaluation = self.root / "external.json"
        self.save(self.args.v2_evaluation, self.evaluation())
        text, provenance = self.render()
        self.assertIn("0.712", text)
        self.assertIn("[0.123, 0.988]", text)
        self.assertIn("+0.123", text)
        self.assertIn("[-0.045, +0.234]", text)
        self.assertIn("0.6123456789012345", text)
        self.assertIn("Les étiquettes ne sont pas présumées invariantes", text)
        self.assertTrue(provenance["external_evaluation_provided"])
        self.assertIn(str(self.args.v2_evaluation), provenance["source_sha256"])

    def audit(self):
        parent = self.root / "export"
        (parent / "run").mkdir(parents=True)
        (parent / "audit").mkdir()
        threshold = .6123456789012345
        metadata = {"run_id": "fixture-T0", "model_identity": {"fixture": True}, "transform": "T0",
            "threshold_provenance_sha256": "d" * 64, "preregistration_sha256": self.prereg,
            "split_filename": report.V2_EXTERNAL_NAME, "split_sha256": report.V2_EXTERNAL_SHA256}
        self.save(parent / "run/metadata.json", metadata)
        (parent / "run/predictions.csv").write_text("clip_id,probability_leak\nc0,0.7\nc1,0.8\nc2,0.2\n")
        summary = {**metadata, "schema": "pipe-v2-external-export-v1", "model": "tslm", "text_audit": True,
            "status": "complete", "weights_unchanged": True, "threshold": threshold, "threshold_repr": repr(threshold),
            "external_manifest_sha256": report.V2_EXTERNAL_SHA256,
            "run_sha256": {name: report._sha(parent / "run" / name) for name in ("metadata.json", "predictions.csv")},
            "partitions": {"external": {"expected": 3, "attempted": 3, "errors": 1,
                "raw_band_matches_dsp": 2, "raw_class_matches_decision": 2,
                "latency_ms": {"mean": 123.456}, "fallback_used": 1}, "background": {"expected": 60}}}
        self.save(parent / "audit/summary.json", summary)
        def payload(score):
            return {"probability_leak": score, "prediction": "leak" if score >= threshold else "no_leak",
                "threshold": threshold, "dominant_band_hz": "0-1000",
                "description": "Greatest mean spectral energy: 0-1000 Hz.", "fallback_used": False,
                "audit": {"raw_class_matches_decision": True, "raw_band_matches_dsp": True}}
        rows = [{"clip_id": "c0", "transform": "T0", "payload": payload(.7), "error": None},
                {"clip_id": "c1", "transform": "T0", "payload": None, "error": {"message": "fixture failure"}},
                {"clip_id": "c2", "transform": "T0", "payload": payload(.2), "error": None}]
        rows[0]["payload"].update(prediction="no_leak", description="Contradiction volontaire", fallback_used=True)
        with (parent / "audit/external.jsonl").open("w") as stream:
            stream.write("\n".join(json.dumps(row) for row in rows) + "\n")
        return parent

    def test_displayed_contradictions_independent_of_raw_flags_errors_and_missing_not_pass(self):
        self.args.v2_audits = [self.audit()]
        text, provenance = self.render()
        partitions = provenance["audits"][0]["partitions"]
        primary = partitions["external"]
        self.assertEqual(primary["expected"], 3)
        self.assertEqual(primary["attempted"], 3)
        self.assertEqual(primary["errors"], 1)
        self.assertEqual(primary["display_checked"], 2)
        self.assertEqual(primary["displayed_class_contradictions"], 1)
        self.assertEqual(primary["displayed_description_contradictions"], 1)
        self.assertEqual(primary["fallback_used"], 1)
        self.assertEqual(primary["status"], "non vérifié ou défaut constaté")
        self.assertEqual(partitions["background"]["status"], "non vérifié")
        self.assertNotIn("attempted", partitions["background"])
        self.assertIn("123.456", text)
        self.assertIn("pas une preuve acoustique indépendante", text)
        self.assertIn("Bruit annexe — sans cible binaire", text)
        self.assertEqual([e["clip_id"] for e in primary["examples_first_in_journal_order"]], ["c0", "c1"])
        self.assertIn(str(self.args.v2_audits[0] / "audit/external.jsonl"), provenance["source_sha256"])

    def test_mixed_run_identity_coverage_or_old_test_report_rejected(self):
        parent = self.audit()
        summary_path = parent / "audit/summary.json"
        summary = json.loads(summary_path.read_text())
        self.save(summary_path, {**summary, "run_id": "another-run"})
        self.args.v2_audits = [parent]
        with self.assertRaisesRegex(ValueError, "mélangés"):
            self.render()
        self.save(summary_path, summary)
        raw = parent / "audit/external.jsonl"
        original = raw.read_text()
        raw.write_text(original.replace('"clip_id": "c2"', '"clip_id": "unknown"'))
        with self.assertRaisesRegex(ValueError, "Couverture"):
            self.render()
        self.args.v2_audits = None
        self.args.v2_evaluation = self.root / "old.json"
        external = self.evaluation()
        external["split"]["filename"] = "split_v2.csv"
        self.save(self.args.v2_evaluation, external)
        with self.assertRaisesRegex(ValueError, "externe gelée"):
            self.render()

    def test_combined_selection_evaluation_and_audit_join_exact_producer_fields(self):
        parent = self.audit()
        self.args.v2_audits = [parent]
        evaluation = self.evaluation()
        proof = evaluation["runs"]["fixture-T0"]["fixed_threshold_provenance"]
        proof["run_files_sha256"] = {name: report._sha(parent / "run" / name)
                                     for name in ("metadata.json", "predictions.csv")}
        self.args.v2_evaluation = self.root / "combined.json"
        self.save(self.args.v2_evaluation, evaluation)
        text, provenance = self.render()
        self.assertIn("fixture-T0_vs_fixture-T1", text)
        self.assertEqual(provenance["audits"][0]["run_id"], "fixture-T0")
        self.assertEqual(provenance["audits"][0]["partitions"]["external"]["displayed_class_contradictions"], 1)
        self.args.out = self.root / "wrong-link"
        proof["run_files_sha256"]["predictions.csv"] = "0" * 64
        self.save(self.args.v2_evaluation, evaluation)
        with self.assertRaisesRegex(ValueError, "même run/seuil"):
            self.render()

    def test_missing_audit_directory_and_legacy_caption(self):
        self.args.v2_audits = [self.root / "missing"]
        text, provenance = self.render()
        self.assertIn("Non vérifié : Aucun reçu d'audit", text)
        self.assertEqual(provenance["audits"][0]["status"], "non vérifié")
        legacy = report.decision_table(self.block())
        self.assertIn("au seuil du run choisi sur validation", legacy)
        self.assertNotIn("diagnostic CV", legacy)


if __name__ == "__main__":
    unittest.main()
