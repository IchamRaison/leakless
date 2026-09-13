"""Garde-fous D0 sur fixtures : aucun poids, WAV réel ou entraînement."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import diagnose_causal as causal


def fixtures():
    rows, parts = [], {}
    for part, count, nl, leak in (("fit", 500, 16, 52), ("heldout", 98, 8, 26)):
        groups = [(f"{part}-nl-{i}", "no_leak") for i in range(nl)]
        groups += [(f"{part}-leak-{i}", "leak") for i in range(leak)]
        members = []
        for i in range(count):
            group, label = groups[i % len(groups)]
            row = {"clip_id": f"{part}-{i:03d}", "group_id": group, "label": label, "fold": "train"}
            rows.append(row)
            members.append(row["clip_id"])
        parts[part] = members
    return sorted(rows, key=lambda r: r["clip_id"]), {
        "fold_id": 0, "train_ids": parts["fit"], "heldout_ids": parts["heldout"]}


def debug_rows(rows, count):
    # Double du sélecteur existant, pour ne pas importer PyTorch en tests CPU.
    selected = []
    for label in ("leak", "no_leak"):
        seen = set()
        for row in sorted(rows, key=lambda r: r["clip_id"]):
            if row["label"] == label and row["group_id"] not in seen:
                selected.append(row)
                seen.add(row["group_id"])
                if len(seen) == count // 2:
                    break
    return selected


def observation(probability):
    return {"scoring": {"probability_leak": probability, "binary_nll": .2},
            "loss": {"terms": {name: {"n_tokens": 2, "nll_sum": .4}
                for name in ("class", "first_discriminating_token", "description", "eos", "response")}},
            "optimizer_steps": 0, "llm_forward_counts": {"supervision": 1, "scoring": 1},
            "state_guard": {"unchanged": True}, "alignment": {"prompts_exact": True}}


class CausalRunnerChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows, self.fold = fixtures()
        self.train_patch = patch.dict(sys.modules, {"pipe.tslm.train": SimpleNamespace(debug_rows=debug_rows)})
        self.train_patch.start()
        self.addCleanup(self.train_patch.stop)
        self.scope = causal.cohorts(self.rows, self.fold)

    def test_cohorts_and_donors_are_deterministic_fit_only_and_class_controlled(self):
        self.assertEqual(causal.cohorts(list(reversed(self.rows)), self.fold), self.scope)
        by_id = {r["clip_id"]: r for r in self.rows}
        subset = self.scope["subset_ids"]
        self.assertEqual(len(subset), 32)
        self.assertTrue(set(subset) <= set(self.fold["train_ids"]))
        self.assertEqual(len({by_id[cid]["group_id"] for cid in subset}), 32)
        self.assertEqual(sum(by_id[cid]["label"] == "leak" for cid in subset), 16)
        for name, mapping in self.scope["donor_maps"].items():
            self.assertEqual(set(mapping), set(mapping.values()))
            self.assertEqual(set(mapping), set(subset))
            self.assertTrue(all(cid != donor for cid, donor in mapping.items()))
            for cid, donor in mapping.items():
                self.assertEqual(by_id[cid]["label"] == by_id[donor]["label"], name == "within_class_cycle")
        self.assertEqual(causal.PROTOCOL["observations"], {"A": 694, "C": 822})
        self.assertEqual(causal.PROTOCOL["total_llm_forwards"], 3032)

    def test_no_official_fold_or_cross_group_can_enter_cohort(self):
        for fold in ("val", "test", "external"):
            with self.assertRaisesRegex(ValueError, "598 train"):
                causal.cohorts([{**self.rows[0], "fold": fold}, *self.rows[1:]], self.fold)
        invalid = copy.deepcopy(self.fold)
        invalid["heldout_ids"][0] = invalid["train_ids"][0]
        with self.assertRaisesRegex(ValueError, "500/98"):
            causal.cohorts(self.rows, invalid)
        bad = copy.deepcopy(self.rows)
        bad[-1]["group_id"] = bad[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "Groupes partagés"):
            causal.cohorts(bad, self.fold)

    def test_preregister_reads_only_train_and_binds_scope_before_model_load(self):
        parent = self.root / "campaign"
        parent.mkdir()
        args = SimpleNamespace(campaign=parent, output=self.root / "d0", code_revision="a" * 40)
        ctx = {"registration": {"paths": {}, "folds": [self.fold], "identity": {"original": True}}, "split": object()}
        with patch.object(causal, "checked_campaign", return_value=(ctx, {}, {"six": "receipts"})), \
                patch.object(causal.campaign, "load_fold", return_value=(self.rows, {}, {}, None)) as load, \
                patch.object(causal, "source_hashes", return_value={"source": "sha"}), \
                patch.object(causal, "input_hashes", return_value={"train": "sha"}), \
                patch.object(causal, "initialize") as initialize:
            result = causal.preregister(args)
            self.assertTrue(result["preregistered"])
            load.assert_called_once_with({}, ctx["split"], "train")
            initialize.assert_not_called()
            stored = json.loads((args.output / "preregistration.json").read_text())
            self.assertFalse(stored["model_loaded"])
            self.assertEqual(stored["cohorts"], self.scope)
            self.assertEqual(stored["cv_receipts"], {"six": "receipts"})
            with self.assertRaises(FileExistsError):
                causal.preregister(args)
            args.output = parent / "cv/new"
            with self.assertRaisesRegex(ValueError, "campagne scellée"):
                causal.preregister(args)

    def test_context_rejects_changed_source_before_reading_campaign(self):
        out = self.root / "d0"
        out.mkdir()
        value = {"schema": causal.SCHEMA, "protocol": causal.PROTOCOL,
                 "campaign_preregistration_sha256": causal.CAMPAIGN_SHA256,
                 "model_loaded": False, "source_sha256": {"old": "sha"}}
        causal.campaign.write_json(out / "preregistration.json", value)
        with patch.object(causal, "source_hashes", return_value={"new": "sha"}), \
                patch.object(causal, "checked_campaign") as checked:
            with self.assertRaisesRegex(ValueError, "Préinscription"):
                causal.context(out)
            checked.assert_not_called()

    def test_partial_stops_and_complete_receipt_is_reused_without_loading_data(self):
        out = self.root / "d0"
        out.mkdir()
        ctx = {"output": out, "preregistration_sha256": "a" * 64}
        (out / "A").mkdir()
        with patch.object(causal.campaign, "load_fold") as load:
            with self.assertRaisesRegex(ValueError, "incomplète"):
                causal.observe(ctx, "A")
            load.assert_not_called()
        result = causal.campaign.finish(out / "A", {"preregistration_sha256": "a" * 64, "complete_fixture": True})
        with patch.object(causal, "initialize") as initialize, patch.object(causal.campaign, "load_fold") as load:
            self.assertEqual(causal.observe(ctx, "A"), result)
            initialize.assert_not_called()
            load.assert_not_called()

    def test_intervention_changes_only_selected_input_and_preserves_receiver_target(self):
        row, donor = self.rows[0], self.rows[1]
        cid, did = row["clip_id"], donor["clip_id"]
        series, amplitudes = {cid: "receiver-series", did: "donor-series"}, {cid: "receiver-text", did: "donor-text"}
        def examples(rows, s, a, metadata, training=False):
            item = {"series": s[cid], "amplitude": a[cid]}
            if training:
                item["answer"] = "receiver-original-target"
            return [item]
        observe = Mock(return_value=observation(.4))
        fake = {"causal_observations": SimpleNamespace(observe_clip=observe),
                "opentslm.time_series_datasets.util": SimpleNamespace(
                    extend_time_series_to_match_patch_size_and_aggregate=lambda batch, normalize: batch)}
        with patch.dict(sys.modules, fake), patch.object(causal.campaign, "examples", side_effect=examples):
            for intervention, expected in (("series", ("donor-series", "receiver-text")),
                                          ("amplitude", ("receiver-series", "donor-text")),
                                          ("both", ("donor-series", "donor-text"))):
                causal.observe_example(object(), row, series, amplitudes, {"amplitude_evidence": True},
                                       donor=donor, intervention=intervention)
                item = observe.call_args.args[1][0]
                self.assertEqual((item["series"], item["amplitude"]), expected)
                self.assertEqual(item["answer"], "receiver-original-target")
            with self.assertRaisesRegex(ValueError, "non préinscrite"):
                causal.observe_example(object(), row, series, amplitudes, {"amplitude_evidence": False},
                                       donor=donor, intervention="amplitude")

    def test_score_controls_reject_coverage_and_record_mismatches_without_clipping(self):
        self.assertTrue(causal.check_scores({"a": .2}, {"a": .2}, "toy")["passed"])
        result = causal.check_scores({"a": .3}, {"a": .2}, "toy")
        self.assertFalse(result["passed"])
        self.assertEqual(result["mismatch_ids"], ["a"])
        with self.assertRaisesRegex(ValueError, "Couverture"):
            causal.check_scores({"a": .2}, {"b": .2}, "toy")

    def runtime_fixture(self, variant="A"):
        out, parent = self.root / f"d0-{variant}", self.root / "campaign"
        out.mkdir()
        checkpoint = parent / f"cv/{variant}/fold-0"
        checkpoint.mkdir(parents=True)
        by_id = {r["clip_id"]: r for r in self.rows}
        self.prob = lambda row: .8 if row["label"] == "leak" else .2
        with (checkpoint / "predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("clip_id", "probability_leak"))
            writer.writerows((cid, self.prob(by_id[cid])) for cid in self.scope["heldout_ids"])
        ctx = {"output": out, "preregistration_sha256": "a" * 64,
            "campaign": {"registration": {"paths": {"base": "base"}, "folds": [self.fold]},
                         "split": object(), "gates": {variant: {"runtime": {}}}},
            "registration": {"campaign": str(parent), "cohorts": self.scope, "input_sha256": {}},
            "receipts": {variant: {0: {"state_sha256_before": {"state": "initial"},
                "state_sha256_after": {"state": "terminal"}, "scoring_spec": {"spec": True}}}}}
        return ctx

    def runtime_patches(self, ctx, states):
        from contextlib import ExitStack
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(causal.campaign, "load_fold", return_value=(self.rows, {}, {}, None)))
        stack.enter_context(patch.object(causal.campaign, "verify_runtime"))
        stack.enter_context(patch.object(causal.campaign, "variant_metadata", return_value={"amplitude_evidence": "C" in ctx["receipts"]}))
        model = SimpleNamespace(scoring_spec=lambda: {"spec": True})
        stack.enter_context(patch.object(causal, "initialize", return_value=model))
        stack.enter_context(patch.object(causal, "load_terminal"))
        stack.enter_context(patch.object(causal, "state_hashes", side_effect=states))
        stack.enter_context(patch.object(causal, "context", return_value=ctx))
        stack.enter_context(patch.object(causal, "partition_summary", return_value={"synthetic_summary": True}))
        stack.enter_context(patch("builtins.print"))
        return stack

    def test_full_mock_run_has_exact_budget_receipts_and_donor_controls(self):
        ctx = self.runtime_fixture()
        self.runtime_patches(ctx, [{"state": "initial"}] * 2 + [{"state": "terminal"}] * 2)
        def observe(model, row, *args, **kwargs):
            return observation(self.prob(kwargs.get("donor", row)))
        with patch.object(causal, "observe_example", side_effect=observe) as observe:
            result = causal.observe(ctx, "A")
        self.assertEqual(observe.call_count, 694)
        self.assertEqual(result["llm_forwards"], 1388)
        self.assertTrue(result["reload"]["passed"])
        self.assertTrue(all(v["donor_reproduction"]["passed"] for v in result["interventions"].values()))
        self.assertEqual(set(result["terminal_partitions"]), {"fit", "heldout"})
        self.assertFalse(result["pooled_598_performance_calculated"])
        self.assertEqual(causal.campaign.finished(ctx["output"] / "A", preregistration_sha256="a" * 64), result)

    def test_c_mock_run_has_three_interventions_per_mapping_and_822_observations(self):
        ctx = self.runtime_fixture("C")
        self.runtime_patches(ctx, [{"state": "initial"}] * 2 + [{"state": "terminal"}] * 2)
        def observe(model, row, *args, **kwargs):
            return observation(self.prob(kwargs.get("donor", row)))
        with patch.object(causal, "observe_example", side_effect=observe) as observe:
            result = causal.observe(ctx, "C")
        self.assertEqual(observe.call_count, 822)
        self.assertEqual(result["llm_forwards"], 1644)
        self.assertEqual(len(result["interventions"]), 6)
        for name, item in result["interventions"].items():
            self.assertEqual("donor_reproduction" in item, name.endswith("-both"))
            self.assertFalse(item["quality_metrics_calculated"])

    def test_changed_initial_state_stops_before_observation_and_preserves_partial(self):
        ctx = self.runtime_fixture()
        self.runtime_patches(ctx, [{"state": "wrong"}])
        with patch.object(causal, "observe_example") as observe:
            with self.assertRaisesRegex(ValueError, "État initial"):
                causal.observe(ctx, "A")
            observe.assert_not_called()
        self.assertTrue((ctx["output"] / "A/started.json").exists())
        self.assertFalse((ctx["output"] / "A/complete.json").exists())
        with self.assertRaisesRegex(ValueError, "incomplète"):
            causal.observe(ctx, "A")

    def test_reload_failure_stops_before_interventions_or_summary(self):
        ctx = self.runtime_fixture()
        self.runtime_patches(ctx, [{"state": "initial"}] * 2 + [{"state": "terminal"}])
        with patch.object(causal, "observe_example", return_value=observation(.5)) as observe:
            with self.assertRaisesRegex(ValueError, "réservés non reproduits"):
                causal.observe(ctx, "A")
            self.assertEqual(observe.call_count, 630)
        self.assertFalse(json.loads((ctx["output"] / "A/reload.json").read_text())["passed"])
        self.assertFalse((ctx["output"] / "A/complete.json").exists())

    def test_summary_uses_token_sums_and_fixed_fit_prior_without_selection(self):
        rows = self.rows[:2]
        observations = {r["clip_id"]: observation(.5) for r in rows}
        observations[rows[1]["clip_id"]]["loss"]["terms"]["class"] = {"n_tokens": 4, "nll_sum": 1.2}
        with patch.object(causal.campaign, "metric_report", return_value={"fixture": True}) as metrics:
            summary = causal.partition_summary(rows, observations, .7)
        self.assertAlmostEqual(summary["loss_terms"]["class"]["nll_per_token"], 1.6 / 6)
        self.assertEqual(summary["controls"]["clip_frequency_from_500_fit_only"]["probability_leak"], .7)
        self.assertTrue(all(call.kwargs == {"threshold": .5} for call in metrics.call_args_list))
        self.assertFalse(summary["threshold_fitted"])


if __name__ == "__main__":
    unittest.main()
