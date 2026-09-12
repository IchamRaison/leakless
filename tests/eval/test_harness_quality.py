"""Tests synthétiques seulement : intégration du harness avant ouverture du test réel."""
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from build_final_report import build_markdown, decision_table  # noqa: E402
import build_final_report  # noqa: E402
from harness import metrics  # noqa: E402


class HarnessQualityChecks(unittest.TestCase):
    def test_unchanged_official_files(self):
        # Origine : 6dfdf63580bc15ce6a1cf0817b9da3569553aca1.
        expected = {
            "scripts/eval/harness/features.py": "db819eee9a7703b08c93b85dc8d5f70f23841d41d4a8ec3b5f2d7f0d4773040c",
            "scripts/eval/harness/__init__.py": "513c964ce3530abe82a6cc8549a1aa520852536dbbde237a8615f96319e6c93c",
        }
        for relative, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest, relative)

    def test_ap_ties_are_permutation_invariant(self):
        scores = np.array([.5, .5])
        for labels in ([1, 0], [0, 1]):
            self.assertEqual(metrics.pr_auc(np.array(labels), scores), .5)
        labels, scores = np.array([1, 0, 1, 0]), np.array([.9, .5, .5, .1])
        self.assertAlmostEqual(metrics.pr_auc(labels, scores), 5 / 6)
        order = np.array([0, 2, 1, 3])
        self.assertEqual(metrics.pr_auc(labels, scores), metrics.pr_auc(labels[order], scores[order]))
        self.assertTrue(math.isnan(metrics.pr_auc(np.array([1, 1]), np.array([.8, .2]))))
        self.assertTrue(math.isnan(metrics.pr_auc(np.array([0, 0]), np.array([.8, .2]))))

    def test_ap_matches_sklearn_on_synthetic_arrays_when_available(self):
        try:
            from sklearn.metrics import average_precision_score
        except ImportError:
            self.skipTest("Oracle sklearn facultatif ; le test exact NumPy des ex aequo reste actif")
        rng = np.random.default_rng(7)
        for _ in range(10):
            labels = np.r_[0, 1, rng.integers(0, 2, 18)]
            scores = rng.integers(0, 4, len(labels)) / 3
            self.assertAlmostEqual(metrics.pr_auc(labels, scores), average_precision_score(labels, scores))

    def test_existing_threshold_auc_and_verdict_rules(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([.1, .3, .7, .9])
        groups = np.array(["a", "a", "b", "b"])
        self.assertEqual(metrics.pick_threshold(labels, scores, groups), .8)
        with self.assertRaises(metrics.ThresholdFittingError):
            metrics.pick_threshold(labels, scores, groups, fold="test")
        self.assertEqual(metrics.roc_auc(np.array([1, 0]), np.array([.5, .5])), .5)
        self.assertEqual(metrics.pick_threshold(np.array([1, 0]), np.array([.5, .5]),
                                               np.array(["a", "b"])), .5 - 1e-9)
        self.assertEqual(metrics.verdict(.1, .3), "compatible with improvement")
        self.assertEqual(metrics.verdict(-.3, -.1), "compatible with degradation")
        self.assertEqual(metrics.verdict(-.1, .3), "inconclusive")

    def test_explicit_archive_revision_is_validated_and_does_not_call_git(self):
        bad = subprocess.run([sys.executable, str(ROOT / "scripts/eval/build_final_report.py"),
                              "--runs", "nonexistent", "--code-revision", "short"],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn("40 caractères hexadécimaux", bad.stderr)
        with tempfile.TemporaryDirectory() as directory:
            revision = "a" * 40
            split = SimpleNamespace(sha256="fixture", clips=[SimpleNamespace(group_id="toy")])
            argv = ["report", "--runs", "synthetic", "--tslm-run-id", "tslm-v1",
                    "--code-revision", revision, "--out", directory]
            with patch.object(sys, "argv", argv), \
                    patch.object(build_final_report.split_loader, "load_split", return_value=split), \
                    patch.object(build_final_report.contract, "load_run",
                                 return_value=SimpleNamespace(run_id="tslm-v1")), \
                    patch.object(build_final_report, "evaluate_run", return_value={}), \
                    patch.object(build_final_report, "build_markdown", return_value="synthetic") as render, \
                    patch("subprocess.run", side_effect=AssertionError("Git interdit pour l'archive")), \
                    patch("builtins.print"):
                build_final_report.main()
            self.assertEqual(render.call_args.args[-1], revision)

    def test_markdown_does_not_invent_control_superiority_or_inconclusiveness(self):
        def run(name, auc):
            def fold(negative):
                block = {"roc_auc": auc, "pr_auc": .5, "macro_f1": .5, "brier": .2,
                         "tp": 2, "fp": 1, "tn": 3, "fn": 2}
                return {"n_clips": 8, "n_clusters": 4, "n_clusters_leak": 4 - negative,
                        "n_clusters_non_leak": negative, "clip_level": block, "cluster_level": block,
                        "bootstrap_ci95": {"clip_roc_auc": {"ci95_low": .1, "ci95_high": .2}}}
            return {"model_name": name, "checkpoint": "synthetic", "training_commit": "fixture",
                    "timestamp": "synthetic", "folds": {"val": fold(2), "test": fold(3)}}
        result = {"split": {"filename": "synthetic.csv", "sha256": "fixture", "n_clips": 16,
                            "n_clusters": 8}, "aggregation_rule": "median",
                  "bootstrap": {"draws": 2000, "seed": 20260912, "unit": "cluster"},
                  "runs": {"c0": run("C0", .9), "c1": run("C1", .2),
                           "tslm-v1": run("TSLM", .95)}}
        comparisons = {"tslm-v1_vs_c1": {"cluster_roc_auc": {
            "delta_observe": .75, "ci95_low": .4, "ci95_high": .9,
            "lecture": "compatible with improvement"}}}
        text = build_markdown(result, comparisons, "tslm-v1", None, "fixture")
        self.assertIn("compatible with improvement", text)
        self.assertIn("**3 groupes *non-leak* en test, 2 en validation.**", text)
        self.assertIn("aucun rapport d'invariants fourni", text)
        for unsupported in ("C1 (forme d'enveloppe seule, audio normalisé) fait au moins aussi bien",
                            "Aucune comparaison n'est concluante à cette taille",
                            "Tous les intervalles sont larges et se recouvrent",
                            "Le niveau sonore absolu est un raccourci mesuré sur ce jeu"):
            self.assertNotIn(unsupported, text)

    def test_decision_table_uses_existing_counts_and_handles_zero_denominators(self):
        text = decision_table({"clip_level": {"tp": 3, "fp": 1, "tn": 4, "fn": 2},
                               "cluster_level": {"tp": 1, "fp": 1, "tn": 3, "fn": 0}})
        self.assertIn("| clip | 5 | 5 | 1 | 2 | 75.0 % | 66.7 % | 60.0 % | 20.0 % | 40.0 % |", text)
        self.assertIn("| groupe | 1 | 4 | 1 | 0 | 50.0 % | 66.7 % | 100.0 % | 25.0 % | 0.0 % |", text)
        empty = decision_table({"clip_level": {"tp": 0, "fp": 0, "tn": 2, "fn": 0},
                                "cluster_level": {"tp": 0, "fp": 0, "tn": 0, "fn": 0}})
        self.assertIn("| clip | 0 | 2 | 0 | 0 | — | — | — | 0.0 % | — |", empty)
        self.assertIn("| groupe | 0 | 0 | 0 | 0 | — | — | — | — | — |", empty)


if __name__ == "__main__":
    unittest.main()
