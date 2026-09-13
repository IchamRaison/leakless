"""Garde-fous du runner ; aucun chargement Qwen, GPU ou fit de données réelles."""
import copy
import importlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
campaign = importlib.import_module("run_v2_campaign")


class CampaignChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sha = "a" * 64
        self.ctx = {"output": self.root, "preregistration_sha256": self.sha,
            "split": object(), "registration": {"paths": {}, "counts": campaign.COUNTS,
                "folds": [{"fold_id": i, "train_ids": [f"train-{i}"],
                           "heldout_ids": [f"heldout-{i}"]} for i in range(3)]}}
        self.preprocessing = SimpleNamespace(CANONICAL_VERSION="canonical-v2",
            AMPLITUDE_EVIDENCE_VERSION="amplitude-v1", amplitude_features=Mock(),
            amplitude_text=lambda a: ",".join(map(str, a)), decode_wav=Mock(), preprocess_audio=Mock())

    def seal(self, relative, **result):
        directory = self.root / relative
        directory.mkdir(parents=True)
        campaign.write_json(directory / "result.json", result)
        return campaign.finish(directory, {"preregistration_sha256": self.sha, **result})

    def candidate(self, scores):
        return [{"fold_id": i, "metrics": {"group_roc_auc_full": score,
                 "clip_roc_auc_full": 1 - score, "group_roc_auc": round(score, 6)}}
                for i, score in enumerate(scores)]

    def all_comparisons(self, c1_sha=None):
        for variant in campaign.VARIANTS:
            for fold in self.ctx["registration"]["folds"]:
                self.seal(f"cv/{variant}/fold-{fold['fold_id']}", **fold,
                    training_ids=fold["train_ids"], variant=variant, validation_or_test_used=False,
                    metrics={"group_roc_auc_full": 0.7 if variant == "C" else 0.6,
                             "clip_roc_auc_full": 0.6})
        self.seal("cv/C1", selected_C=0.1, candidates=[], n_configs_compared=4, n_fits=12,
                  **({"preregistration_sha256": c1_sha} if c1_sha else {}))

    def test_finished_requires_complete_intact_artifacts_and_same_registration(self):
        self.assertIsNone(campaign.finished(self.root / "absent", preregistration_sha256=self.sha))
        (self.root / "partial").mkdir()
        with self.assertRaisesRegex(ValueError, "incomplète"):
            campaign.finished(self.root / "partial", preregistration_sha256=self.sha)
        expected = self.seal("complete", ok=True)
        self.assertEqual(campaign.finished(self.root / "complete", preregistration_sha256=self.sha), expected)
        with self.assertRaisesRegex(ValueError, "autre préinscription"):
            campaign.finished(self.root / "complete", preregistration_sha256="b" * 64)
        campaign.write_json(self.root / "complete" / "added.json", {"foreign": True})
        with self.assertRaisesRegex(ValueError, "Artefacts modifiés"):
            campaign.finished(self.root / "complete", preregistration_sha256=self.sha)

    def test_selection_uses_unrounded_mean_three_groups_not_clip_or_pooled(self):
        candidates = {"A": self.candidate([0.7, 0.7, 0.7]),
                      "C": self.candidate([0.7000001, 0.7, 0.7])}
        winner, summaries = campaign.choose_variant(candidates)
        self.assertEqual(winner, "C")
        self.assertGreater(summaries[1]["mean_group_roc_auc"], summaries[0]["mean_group_roc_auc"])
        candidates["C"] = copy.deepcopy(candidates["A"])
        self.assertEqual(campaign.choose_variant(candidates)[0], "A")
        candidates = {"A": self.candidate([1, 1, 0]), "C": self.candidate([0.7] * 3)}
        self.assertEqual(campaign.choose_variant(candidates)[0], "C")

    def test_selection_rejects_missing_fold_duplicate_and_nonfinite_auc(self):
        complete = {v: self.candidate([0.6] * 3) for v in campaign.VARIANTS}
        invalid = [dict(A=complete["A"]), {**complete, "C": complete["C"][:2]}]
        duplicated = copy.deepcopy(complete)
        duplicated["C"][2]["fold_id"] = 0
        invalid.append(duplicated)
        nonfinite = copy.deepcopy(complete)
        nonfinite["A"][1]["metrics"]["group_roc_auc_full"] = float("nan")
        invalid.append(nonfinite)
        for candidates in invalid:
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                campaign.choose_variant(candidates)

    def test_select_requires_all_six_plus_c1_and_never_opens_audio(self):
        with patch.object(campaign, "load_fold") as load:
            with self.assertRaisesRegex(ValueError, "six fits"):
                campaign.select(self.ctx)
            self.assertFalse((self.root / "selection").exists())
            self.all_comparisons()
            result = campaign.select(self.ctx)
            self.assertEqual(result["selected_variant"], "C")
            self.assertFalse(result["pooled_out_of_fold_auc_calculated"])
            self.assertEqual(campaign.select(self.ctx), result)
            load.assert_not_called()

    def test_select_rejects_c1_from_another_registration(self):
        self.all_comparisons(c1_sha="b" * 64)
        with self.assertRaisesRegex(ValueError, "autre préinscription"):
            campaign.select(self.ctx)
        self.assertFalse((self.root / "selection").exists())

    def test_reused_c1_and_final_stages_cannot_bypass_registration(self):
        self.seal("cv/C1", preregistration_sha256="b" * 64)
        with patch.object(campaign, "load_fold") as load:
            with self.assertRaisesRegex(ValueError, "autre préinscription"):
                campaign.compare_c1(self.ctx)
            with self.assertRaisesRegex(ValueError, "Sélection complète"):
                campaign.fit_final(self.ctx)
            load.assert_not_called()

    def test_context_rejects_mutated_recipe_or_source_before_any_fold_read(self):
        registration = {"schema": campaign.SCHEMA, "recipe": campaign.RECIPE,
            "variants": list(campaign.VARIANTS), "selection_rule": campaign.SELECTION_RULE,
            "threshold_rule": campaign.THRESHOLD_RULE, "counts": campaign.COUNTS,
            "prior_diagnostic_fits": campaign.PRIOR_DIAGNOSTIC_FITS,
            "diagnostic_threshold": 0.5, "official_validation_for_selection": False,
            "test_or_external_data_read": False, "paths": {}, "identity": {"source": "original"},
            "folds": []}
        path = self.root / "preregistration.json"
        campaign.write_json(path, registration)
        with patch.object(campaign, "prerequisites", return_value=(None, {"folds": []}, {},
                          {"source": "changed"})), patch.object(campaign, "load_fold") as load:
            with self.assertRaisesRegex(ValueError, "modifiés depuis"):
                campaign.context(self.root)
            load.assert_not_called()
        for key, value in (("recipe", {**campaign.RECIPE, "epochs": 5}),
                           ("variants", ["A", "B", "C"]),
                           ("official_validation_for_selection", True), ("counts", {})):
            with patch.object(campaign, "read_json", return_value={**registration, key: value}), \
                    patch.object(campaign, "prerequisites") as prerequisites:
                with self.assertRaisesRegex(ValueError, "Préinscription"):
                    campaign.context(self.root)
                prerequisites.assert_not_called()

    def test_preregistration_precedes_any_fit_and_reads_train_only(self):
        args = SimpleNamespace(output=self.root / "new", code_revision="c" * 40,
            **{name: self.root / name for name in campaign.PATH_ARGUMENTS})
        with patch.object(campaign, "prerequisites", return_value=("split", {"folds": []}, {}, {})), \
                patch.object(campaign, "load_fold") as load, patch.object(campaign, "initialize_training") as fit:
            result = campaign.preregister(args)
            load.assert_called_once()
            self.assertEqual(load.call_args.args[2], "train")
            fit.assert_not_called()
            registration = campaign.read_json(args.output / "preregistration.json")
            self.assertEqual(registration["counts"], campaign.COUNTS)
            self.assertEqual(registration["prior_diagnostic_fits"]["total_logistic_fits"], 6)
            self.assertEqual(registration["recipe"]["microbatch_size"], 1)
            self.assertEqual(result["preregistration_sha256"], campaign.sha256_file(args.output / "preregistration.json"))
            with self.assertRaises(FileExistsError):
                campaign.preregister(args)

    def test_invalid_gate_stops_before_preregistration_or_fit(self):
        args = SimpleNamespace(output=self.root / "new", code_revision="c" * 40,
            **{name: self.root / name for name in campaign.PATH_ARGUMENTS})
        with patch.object(campaign, "prerequisites", side_effect=ValueError("gate FAILED")), \
                patch.object(campaign, "load_fold") as load, patch.object(campaign, "initialize_training") as fit:
            with self.assertRaisesRegex(ValueError, "gate FAILED"):
                campaign.preregister(args)
            load.assert_not_called()
            fit.assert_not_called()
        self.assertFalse(args.output.exists())

    def test_gate_requires_exact_variant_and_complete_amplitude_tokens(self):
        constants = SimpleNamespace(AMPLITUDE_SCORING_VERSION="C", CANONICAL_SCORING_VERSION="A")
        ids = [f"c{i:03}" for i in range(209)]
        gate = {"clip_ids": ids, "amplitude_evidence": True,
                "scoring_spec": {"version": "C", "acoustic_batching": "one_clip_four_channels"},
                "test_audio_or_cache_opened": False, "quality_metrics_calculated": False,
                "amplitude_tokens": {cid: 1 for cid in ids}}
        with patch.dict(sys.modules, {"pipe.tslm.model": constants}):
            campaign.validate_gate(gate, "C", ids)
            for changed in ({"amplitude_tokens": {}}, {"clip_ids": list(reversed(ids))},
                            {"amplitude_evidence": False}, {"test_audio_or_cache_opened": True}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    campaign.validate_gate({**gate, **changed}, "C", ids)

    def test_provenance_is_variant_exact_and_does_not_claim_final_parity_pass(self):
        a, c = campaign.provenance_overrides("A"), campaign.provenance_overrides("C")
        self.assertEqual(a["audio"], c["audio"])
        self.assertIn("Aucun WAV de validation embarqué", a["audio"]["reload_example"])
        self.assertIn("n'atteste pas leur réussite", a["audio"]["reload_example"])
        self.assertTrue(c["transformations"].startswith(a["transformations"]))
        self.assertNotIn("descripteurs", a["transformations"])
        self.assertIn("Neuf descripteurs statiques C1", c["transformations"])
        self.assertIn("six chiffres significatifs", c["transformations"])
        with self.assertRaises(ValueError):
            campaign.provenance_overrides("B")

    def test_frozen_fold_document_and_group_partitions_checked_without_new_split(self):
        path = ROOT / "docs/evidence/tslm-v2/train-diagnostic-001/folds.json"
        self.assertEqual(campaign.sha256_file(path), campaign.FOLDS_SHA256)
        document = campaign.read_json(path)
        with patch.object(campaign.v2_c1, "_partition_indices") as partition:
            campaign.validate_folds([], document)
            self.assertEqual(partition.call_args.args[2], document["folds"])
            with self.assertRaisesRegex(ValueError, "Protocole"):
                campaign.validate_folds([], {**document, "seed": 12})

    def test_initializer_constructs_new_model_and_optimizer_each_time_with_frozen_qwen(self):
        parameters = lambda frozen=False: [SimpleNamespace(requires_grad=not frozen)]
        models, optimizers = [], []

        def make_model(base, **kwargs):
            self.assertEqual(base, "base-only")
            self.assertEqual(kwargs, {"device": "cuda", "single_clip_acoustic_encoding": True,
                                      "amplitude_evidence": len(models) == 1})
            model = SimpleNamespace(encoder=SimpleNamespace(parameters=lambda: parameters()),
                projector=SimpleNamespace(parameters=lambda: parameters()),
                llm=SimpleNamespace(parameters=lambda: parameters(frozen=True)))
            models.append(model)
            return model

        def make_optimizer(groups, weight_decay):
            self.assertEqual([g["lr"] for g in groups], [2e-4, 1e-4])
            self.assertEqual(weight_decay, 0.01)
            optimizer = SimpleNamespace(state={}, groups=groups)
            optimizers.append(optimizer)
            return optimizer

        torch = SimpleNamespace(manual_seed=Mock(), cuda=SimpleNamespace(is_available=lambda: False),
                                optim=SimpleNamespace(AdamW=make_optimizer))
        with patch.dict(sys.modules, {"torch": torch, "pipe.tslm.model": SimpleNamespace(AcousticQwenSP=make_model)}):
            first = campaign.initialize_training("base-only", "A")
            first[1].state["previous_fold"] = 1
            second = campaign.initialize_training("base-only", "C")
        self.assertIsNot(first[0], second[0])
        self.assertIsNot(first[1], second[1])
        self.assertEqual(second[1].state, {})
        self.assertEqual(torch.manual_seed.call_args_list[0].args, (20260912,))
        self.assertEqual(torch.manual_seed.call_args_list[1].args, (20260912,))

    def test_model_examples_exclude_ids_labels_metadata_except_supervised_answer(self):
        rows = [{"clip_id": "sensitive-id", "group_id": "group", "label": "leak",
                 "fold": "train", "pressure": "secret-metadata"}]
        series, amplitude = {"sensitive-id": np.zeros((4, 64))}, {"sensitive-id": np.arange(9)}
        calls = []

        def model_input(values, metadata, amplitude_features):
            calls.append(amplitude_features)
            return {"pre_prompt": "signal", "time_series": values}

        self.preprocessing.target_text = lambda label, values: label + "; description"
        with patch.dict(sys.modules, {"pipe.tslm.predict": SimpleNamespace(model_input_for_model=model_input),
                                       "pipe.tslm.preprocessing": self.preprocessing}):
            a = campaign.examples(rows, series, amplitude, {"amplitude_evidence": False})
            c = campaign.examples(rows, series, amplitude, {"amplitude_evidence": True}, training=True)
        self.assertEqual(set(a[0]), {"pre_prompt", "time_series"})
        self.assertEqual(set(c[0]), {"pre_prompt", "time_series", "answer"})
        self.assertIsNone(calls[0])
        np.testing.assert_array_equal(calls[1], amplitude["sensitive-id"])

    def test_load_fold_checks_waveform_cache_bands_amplitude_and_official_c1(self):
        rows = [{"clip_id": "clip", "group_id": "group", "fold": "train", "label": "leak"}]
        series, amplitude, original = np.zeros((4, 64), dtype=np.float32), np.arange(9.0), np.arange(8000.0)
        paths = {"prepared": str(self.root), "manifests": "unused", "data_root": str(self.root)}
        (self.root / "clip.wav").write_bytes(b"synthetic-test-fixture")
        split = SimpleNamespace(path_of=lambda cid: "clip.wav")
        self.preprocessing.preprocess_audio.return_value = series
        self.preprocessing.amplitude_features.return_value = amplitude
        self.preprocessing.decode_wav.return_value = original

        def cache(amplitude_value):
            np.savez(self.root / "train.npz", ids=["clip"], series=series[None],
                preprocessing_version="canonical-v2", amplitude_features=amplitude_value[None],
                amplitude_text=[self.preprocessing.amplitude_text(amplitude)],
                amplitude_evidence_version="amplitude-v1")

        with patch.dict(sys.modules, {"pipe.tslm.preprocessing": self.preprocessing}), \
                patch.object(campaign.diagnostic, "train_rows", return_value=rows), \
                patch.object(campaign.diagnostic, "load_expected_audio_md5", return_value={}), \
                patch.object(campaign.diagnostic, "verify_audio_bytes") as md5, \
                patch.object(campaign.features, "c1_envelope", return_value=amplitude + 1) as c1:
            cache(amplitude)
            result = campaign.load_fold(paths, split, "train")
            np.testing.assert_array_equal(result[2]["clip"], amplitude)
            np.testing.assert_array_equal(result[3], (amplitude + 1)[None])
            self.assertIs(c1.call_args.args[0], original)
            md5.assert_called_once()
            cache(amplitude + 1)
            with self.assertRaisesRegex(ValueError, "divergente"):
                campaign.load_fold(paths, split, "train")
            with self.assertRaisesRegex(ValueError, "test/externe"):
                campaign.load_fold(paths, split, "test")

    def test_validation_needs_both_final_fits_and_distinct_process_before_loading_model_or_audio(self):
        fake_predictor = Mock()
        with patch.dict(sys.modules, {"pipe.tslm.coherent": SimpleNamespace(checkpoint_identity=Mock(),
            decision_artifact=Mock()), "pipe.tslm.predict": SimpleNamespace(Predictor=fake_predictor)}), \
                patch.object(campaign, "load_fold") as load:
            with self.assertRaisesRegex(ValueError, "nouveau processus"):
                campaign.validate(self.ctx)
            self.seal("final/tslm", hostname=campaign.platform.node(), pid=campaign.os.getpid())
            self.seal("final/c1", checkpoint_sha256="b" * 64)
            with self.assertRaisesRegex(ValueError, "nouveau processus"):
                campaign.validate(self.ctx)
            fake_predictor.assert_not_called()
            load.assert_not_called()

    def test_cli_forbids_mutating_paths_or_partial_campaign_after_registration(self):
        with patch.object(campaign, "context") as context, patch("sys.stderr", new_callable=io.StringIO):
            for argv in (["fit-fold", "--base", "other", "--variant", "A", "--fold", "0"],
                         ["select", "--variant", "C"], ["fit-fold", "--variant", "A"],
                         ["preregister", "--variant", "C"]):
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    campaign.main([*argv, "--output", str(self.root)])
            context.assert_not_called()


if __name__ == "__main__":
    unittest.main()
