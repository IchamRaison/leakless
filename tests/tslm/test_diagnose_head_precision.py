"""D1 : fixtures CPU/mock, plus vrai Linear PyTorch CPU si disponible."""
from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import diagnose_head_precision as head


def arm(policy, probability=.5, hidden="same"):
    return {"policy": policy, "probability_leak": probability, "binary_nll": .7,
        "prompt": {"hash": "prompt"}, "head": {"input": {"hash": hidden}, "output_dtype": "float32"},
        "head_weight_identity": {"weights": "same"}, "first_decision_margin": probability - .5,
        "llm_forwards": 1}


class HeadRunnerChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_original_d0_mismatch_stops_before_alternative_forward(self):
        with patch.object(head, "score_arm", return_value=arm(head.POLICIES[0], .2)) as score:
            with self.assertRaisesRegex(ValueError, "différent du score D0"):
                head.score_clip(None, [], 0, {}, .5)
            self.assertEqual(score.call_count, 1)

    def test_hidden_prompt_or_weight_mismatch_stops_but_rounded_score_difference_is_allowed(self):
        values = [arm(policy, .5 + i / 10) for i, policy in enumerate(head.POLICIES)]
        with patch.object(head, "score_arm", side_effect=values):
            result = head.score_clip(None, [], 0, {}, .5)
        self.assertEqual(result["llm_forwards"], 3)
        self.assertTrue(result["input_hidden_weight_identity_exact"])
        self.assertNotEqual(result["arms"][head.POLICIES[0]]["probability_leak"],
                            result["arms"][head.POLICIES[2]]["probability_leak"])
        for field in ("prompt", "head", "head_weight_identity"):
            bad = copy.deepcopy(values)
            bad[1][field] = {"input": "changed"} if field == "head" else {"changed": True}
            with patch.object(head, "score_arm", side_effect=bad) as score:
                with self.assertRaisesRegex(ValueError, "Inputs, hidden"):
                    head.score_clip(None, [], 0, {}, .5)
                self.assertEqual(score.call_count, 2)

    def test_d0_receipt_digest_is_explicit_and_finished_validates_files(self):
        ctx = {"output": self.root, "preregistration_sha256": "a" * 64}
        def receipt(variant):
            return {"variant": variant, "receipt_sha256": head.D0_RECEIPTS[variant],
                "optimizer_steps": 0, "validation_test_external_read": False,
                "weights_unchanged_within_each_state": True,
                "state_sha256": {"terminal_before": {"weights": "same"}, "terminal_after": {"weights": "same"}}}
        with patch.object(head.d0, "context", return_value=ctx), \
                patch.object(head.campaign, "sha256_file", return_value="file-sha"), \
                patch.object(head.campaign, "finished", side_effect=[receipt("A"), receipt("C")]) as finished:
            _, _, identities = head.checked_d0(self.root)
            self.assertEqual(finished.call_count, 2)
            self.assertEqual(identities["A"]["receipt_sha256"], head.D0_RECEIPTS["A"])
            self.assertEqual(identities["A"]["complete_sha256"], "file-sha")
        bad = receipt("A")
        bad["receipt_sha256"] = "f" * 64
        with patch.object(head.d0, "context", return_value=ctx), \
                patch.object(head.campaign, "finished", return_value=bad):
            with self.assertRaisesRegex(ValueError, "deux reçus D0"):
                head.checked_d0(self.root)

    def test_preregister_binds_runtime_source_cohorts_and_does_not_load_model(self):
        d0 = self.root / "d0"
        d0.mkdir()
        ctx = {"preregistration_sha256": "a" * 64,
            "registration": {"campaign": str(self.root / "cv"), "cohorts": {"fit": "fixed"}, "input_sha256": {}},
            "campaign": {"gates": {v: {"runtime": {"frozen": True}} for v in ("A", "C")}}}
        args = SimpleNamespace(d0=d0, output=self.root / "d1", code_revision="b" * 40)
        with patch.object(head, "checked_d0", return_value=(ctx, {}, {"A": "receipt", "C": "receipt"})), \
                patch.object(head.campaign, "verify_runtime") as runtime, \
                patch.object(head, "runtime_identity", return_value={"qwen_source_sha256": "qwen-source"}), \
                patch.object(head, "source_hashes", return_value={"runner": "sha"}), \
                patch.object(head.d0, "initialize") as initialize:
            result = head.preregister(args)
            self.assertTrue(result["preregistered"])
            self.assertEqual(runtime.call_count, 2)
            initialize.assert_not_called()
            saved = json.loads((args.output / "preregistration.json").read_text())
            self.assertFalse(saved["model_loaded"])
            self.assertEqual(saved["runtime"]["qwen_source_sha256"], "qwen-source")
            self.assertEqual(saved["protocol"]["total_forwards"], 3588)
            with self.assertRaises(FileExistsError):
                head.preregister(args)
            args.output = d0 / "forbidden"
            with self.assertRaisesRegex(ValueError, "dans les reçus D0"):
                head.preregister(args)

    def test_changed_runtime_rejected_before_model_or_d0_load(self):
        value = {"schema": head.SCHEMA, "protocol": head.PROTOCOL, "model_loaded": False,
                 "source_sha256": {}, "runtime": {"old": True}}
        head.campaign.write_json(self.root / "preregistration.json", value)
        with patch.object(head, "source_hashes", return_value={}), \
                patch.object(head, "runtime_identity", return_value={"new": True}), \
                patch.object(head, "checked_d0") as checked:
            with self.assertRaisesRegex(ValueError, "runtime/source"):
                head.context(self.root)
            checked.assert_not_called()

    def test_terminal_d0_reader_rejects_other_fold_population_and_duplicate(self):
        rows = [{"clip_id": "a", "group_id": "ga", "label": "leak"},
                {"clip_id": "b", "group_id": "gb", "label": "no_leak"}]
        records = [{**row, "target_class": row["label"], "source_fold": "train", "variant": "A",
                    "stage": "terminal", "partition": "fit" if row["clip_id"] == "a" else "heldout",
                    "observation": {"scoring": {"probability_leak": .5}}} for row in rows]
        path = self.root / "terminal.jsonl"
        def save(values):
            path.write_text("\n".join(json.dumps(v) for v in values) + "\n")
        save(records)
        self.assertEqual(head.d0_scores(path, rows, "A", {"a"}), {"a": .5, "b": .5})
        for values in ([{**records[0], "source_fold": "test"}, records[1]], records[:1],
                       [records[0], records[0]], [{**records[0], "variant": "C"}, records[1]]):
            save(values)
            with self.assertRaises(ValueError):
                head.d0_scores(path, rows, "A", {"a"})

    def test_incomplete_output_refuses_relaunch_complete_receipt_is_reused(self):
        ctx = {"output": self.root, "preregistration_sha256": "a" * 64}
        (self.root / "A").mkdir()
        with patch.object(head.d0, "initialize") as initialize:
            with self.assertRaisesRegex(ValueError, "incomplète"):
                head.observe(ctx, "A")
            initialize.assert_not_called()
        expected = head.campaign.finish(self.root / "A", {"preregistration_sha256": "a" * 64, "test": True})
        with patch.object(head.campaign, "load_fold") as load:
            self.assertEqual(head.observe(ctx, "A"), expected)
            load.assert_not_called()

    def test_partition_comparisons_include_rounded_witness_without_threshold_tuning(self):
        rows = [{"clip_id": "a"}, {"clip_id": "b"}]
        observations = {r["clip_id"]: {"arms": {p: arm(p, .5 + i/10) for i, p in enumerate(head.POLICIES)}} for r in rows}
        metrics = {"clip_roc_auc_full": .6, "group_roc_auc_full": .7}
        with patch.object(head.campaign, "metric_report", return_value=metrics) as report:
            result = head.partition_summary(rows, observations)
        self.assertEqual(len(result["comparisons"]), 3)
        witness = result["comparisons"]["fp32_linear_rounded_bf16_minus_original_bf16"]
        self.assertAlmostEqual(witness["score_abs_change_max"], .2)
        self.assertEqual(witness["scores_differing_count"], 2)
        self.assertTrue(all(call.kwargs == {"threshold": .5} for call in report.call_args_list))
        self.assertFalse(result["threshold_fitted"])

    def test_full_mock_runner_counts_598_triplets_and_only_two_partitions(self):
        rows = [{"clip_id": f"c{i}", "group_id": f"g{i}", "label": "leak" if i % 2 else "no_leak", "fold": "train"}
                for i in range(598)]
        scope = {"fit_ids": [r["clip_id"] for r in rows[:500]], "heldout_ids": [r["clip_id"] for r in rows[500:]]}
        state = {"weights": "same"}
        old = {"output": self.root / "d0", "registration": {"campaign": str(self.root / "cv")},
               "campaign": {"registration": {"paths": {"base": "base"}, "folds": [{}]}, "split": None,
                            "gates": {"A": {"runtime": {}}}},
               "receipts": {"A": {0: {"state_sha256_after": state, "scoring_spec": {}}}}}
        ctx = {"output": self.root / "d1", "registration": {"cohorts": scope, "runtime": {}}, "d0": old,
               "preregistration_sha256": "a" * 64, "receipts": {"A": {"state_sha256": {"terminal_after": state}}}}
        ctx["output"].mkdir()
        model = SimpleNamespace(scoring_spec=lambda: {})
        value = {"arms": {p: arm(p) for p in head.POLICIES}, "llm_forwards": 3,
                 "input_hidden_weight_identity_exact": True, "original_vs_d0_abs_difference": 0.0}
        with ExitStack() as stack:
            stack.enter_context(patch.object(head.campaign, "load_fold", return_value=(rows, {}, {}, None)))
            stack.enter_context(patch.object(head.d0, "cohorts", return_value=scope))
            stack.enter_context(patch.object(head, "d0_scores", return_value={r["clip_id"]: .5 for r in rows}))
            stack.enter_context(patch.object(head.campaign, "verify_runtime"))
            stack.enter_context(patch.object(head, "runtime_identity", return_value={}))
            stack.enter_context(patch.object(head.d0, "initialize", return_value=model))
            stack.enter_context(patch.object(head.d0, "load_terminal"))
            stack.enter_context(patch.object(head.d0, "state_hashes", return_value=state))
            stack.enter_context(patch.object(head, "prepare_head", return_value={"identity": {}}))
            stack.enter_context(patch.object(head.campaign, "variant_metadata", return_value={}))
            examples = stack.enter_context(patch.object(head.campaign, "examples", return_value=[{}]))
            score = stack.enter_context(patch.object(head, "score_clip", return_value=value))
            stack.enter_context(patch.object(head, "context", return_value=ctx))
            summary = stack.enter_context(patch.object(head, "partition_summary", return_value={"fixture": True}))
            stack.enter_context(patch.dict(sys.modules, {"opentslm.time_series_datasets.util": SimpleNamespace(
                extend_time_series_to_match_patch_size_and_aggregate=lambda batch, normalize: batch)}))
            stack.enter_context(patch("builtins.print"))
            result = head.observe(ctx, "A")
        self.assertEqual(score.call_count, 598)
        self.assertEqual(result["llm_forwards"], 1794)
        self.assertEqual(set(result["partitions"]), {"fit", "heldout"})
        self.assertEqual([len(call.args[0]) for call in summary.call_args_list], [500, 98])
        self.assertTrue(all(call.kwargs == {"training": False} for call in examples.call_args_list))
        self.assertIsNone(result["policy_selected"])
        self.assertFalse(result["pooled_598_performance_calculated"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch CPU disponible dans le runtime ML")
class HeadTorchChecks(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        module = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            module.weight.copy_(torch.tensor([[32., .0625], [32., 0.]], dtype=torch.bfloat16))
        module.requires_grad_(False)
        self.model = SimpleNamespace(llm=SimpleNamespace(lm_head=module, get_output_embeddings=lambda: module))
        self.prepared = head.prepare_head(self.model)
        self.input = torch.ones((1, 2), dtype=torch.bfloat16)

    def test_fp32_recomputes_linear_not_cast_quantized_logits_and_rounds_only_last(self):
        outputs = {}
        before = head.tensor_identity(self.model.llm.lm_head.weight)
        for policy in head.POLICIES:
            with head.precision_head(self.prepared, policy) as calls:
                outputs[policy] = self.model.llm.lm_head(self.input)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["input"], head.tensor_identity(self.input))
            self.assertNotIn("forward", vars(self.model.llm.lm_head))
        self.assertEqual(outputs["fp32_linear"].dtype, self.torch.float32)
        self.assertEqual(outputs["original_bf16"][0, 0].item(), 32.)
        self.assertEqual(outputs["fp32_linear"][0, 0].item(), 32.0625)
        self.assertFalse(self.torch.equal(outputs["original_bf16"].float(), outputs["fp32_linear"]))
        self.assertTrue(self.torch.equal(outputs["fp32_linear"].bfloat16(), outputs["fp32_linear_rounded_bf16"]))
        self.assertEqual(head.tensor_identity(self.model.llm.lm_head.weight), before)
        self.assertFalse(self.prepared["weight_fp32"].requires_grad)

    def test_observer_sees_real_head_dtype_and_fp32_values_before_witness_rounding(self):
        seen = []
        with head.precision_head(self.prepared, "fp32_linear_rounded_bf16",
                output_observer=lambda tensor, stage: seen.append((stage, str(tensor.dtype), tensor[0, 0].item()))):
            self.model.llm.lm_head(self.input)
        self.assertEqual(seen, [("fp32_before_round", "torch.float32", 32.0625),
                                ("head_return", "torch.bfloat16", 32.)])

    def test_forward_is_restored_after_exception_in_body_and_head(self):
        module = self.model.llm.lm_head
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with head.precision_head(self.prepared, "fp32_linear"):
                raise RuntimeError("fixture")
        self.assertNotIn("forward", vars(module))
        with self.assertRaisesRegex(ValueError, "Entrée"):
            with head.precision_head(self.prepared, "fp32_linear"):
                module(self.input.float())
        self.assertNotIn("forward", vars(module))
        original = lambda x: x + 1
        module.forward = original
        with head.precision_head(self.prepared, "original_bf16"):
            module(self.input)
        self.assertIs(module.forward, original)

    def test_trainable_non_linear_or_non_bf16_heads_are_rejected(self):
        module = self.model.llm.lm_head
        module.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "nn.Linear"):
            head.prepare_head(self.model)
        module.requires_grad_(False)
        module.float()
        with self.assertRaisesRegex(ValueError, "nn.Linear"):
            head.prepare_head(self.model)


if __name__ == "__main__":
    unittest.main()
