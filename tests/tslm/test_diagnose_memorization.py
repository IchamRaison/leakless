"""D3 : cohortes/arrêt/reload synthétiques ; aucun vrai Qwen, WAV ou GPU."""
from contextlib import contextmanager
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/tslm"), str(ROOT / "tests/tslm")]
import diagnose_memorization as d3
from test_diagnose_causal import observation


def rows32():
    return [{"clip_id": f"train-{i:02}", "group_id": f"group-{i:02}", "fold": "train",
             "label": "leak" if i % 2 else "no_leak"} for i in range(32)]


def dummy_model():
    return SimpleNamespace(compute_loss=lambda *args: None, eval=Mock())


def fake_point(rows, step, success):
    return ({"step": step, "n_correct": 32 if success else 16,
             "binary_nll_mean": .05 if success else math.log(2), "criterion_met": success,
             "observations": 32, "llm_forwards": 64},
            {r["clip_id"]: .9 if r["label"] == "leak" else .1 for r in rows})


class MemorizationChecks(unittest.TestCase):
    def test_recipe_is_one_attempt_original_A_and_exact_budgets(self):
        self.assertEqual(d3.CONFIG, {**d3.campaign.RECIPE, "epochs": 250})
        self.assertEqual(d3.PROTOCOL["observation_steps"], list(range(0, 1001, 100)))
        self.assertEqual(d3.PROTOCOL["binary_nll_limit_strict"], .1)
        self.assertEqual(d3.PROTOCOL["n_training_attempts"], 1)
        self.assertEqual(d3.PROTOCOL["max_training_presentations"], 250 * 32)
        self.assertEqual(d3.PROTOCOL["max_observations_including_reload"], 12 * 32)
        self.assertEqual(d3.PROTOCOL["max_observation_forwards_including_reload"], 12 * 32 * 2)
        valid = rows32()
        d3.validate_rows(valid)
        for name in ("val", "test", "external"):
            with self.assertRaises(ValueError):
                d3.validate_rows([{**valid[0], "fold": name}, *valid[1:]])
        for rows in (valid[:-1], [valid[0], valid[0], *valid[2:]],
                     [{**valid[0], "group_id": valid[1]["group_id"]}, *valid[1:]],
                     [{**valid[0], "label": "leak"}, *valid[1:]]):
            with self.assertRaises(ValueError):
                d3.validate_rows(rows)

    def test_observation_uses_only_32_and_strict_nll_and_all_decisions(self):
        rows, model = rows32(), dummy_model()
        for nll, wrong, success in ((.1, False, False), (.099, False, True), (.001, True, False)):
            called = []
            def observe(model, row, series, amplitudes, metadata):
                called.append(row["clip_id"])
                p = .9 if row["label"] == "leak" else .1
                if wrong and row == rows[0]:
                    p = .9
                value = observation(p)
                value["scoring"]["binary_nll"] = nll
                return value
            with tempfile.TemporaryDirectory() as tmp, patch.object(d3.d0, "observe_example", side_effect=observe):
                point, scores = d3.observation_point(model, rows, {}, {}, 100, Path(tmp))
            self.assertEqual(called, [r["clip_id"] for r in rows])
            self.assertEqual(set(scores), set(called))
            self.assertEqual(point["criterion_met"], success)
            self.assertEqual(point["llm_forwards"], 64)
            self.assertNotIn("clip_frequency_from_500_fit_only", point["controls"])
        with tempfile.TemporaryDirectory() as tmp, patch.object(d3.d0, "observe_example") as observe:
            model.compute_loss._training_observer = True
            with self.assertRaisesRegex(ValueError, "logger"):
                d3.observation_point(model, rows, {}, {}, 100, Path(tmp))
            model.compute_loss._training_observer = False
            with self.assertRaisesRegex(ValueError, "préinscrits"):
                d3.observation_point(model, rows, {}, {}, 1, Path(tmp))
            observe.assert_not_called()

    def run_mock_loop(self, stop_at):
        rows, model, log_active, seen, epoch_calls = rows32(), dummy_model(), [False], [], []
        @contextmanager
        def capture(model, optimizer):
            self.assertFalse(log_active[0])
            log_active[0] = True
            class Audit:
                def pop_step(self, ids, *, training_record):
                    seen.append(ids)
                    return {"extra_forward_count": 0, "ids": ids, "record": training_record}
                def epoch_summary(self, *, reset):
                    return {"n_steps": 4, "n_clips": 32, "extra_forward_count": 0}
            try:
                yield Audit()
            finally:
                log_active[0] = False
        def epoch(model, optimizer, samples, config, number):
            self.assertTrue(log_active[0])
            self.assertEqual(config, d3.CONFIG)
            self.assertEqual(samples, rows)
            epoch_calls.append(number)
            for _ in range(4):
                yield {"epoch": number, "batch_samples": 8}
        def point(model, observed, series, metadata, step, directory):
            self.assertFalse(log_active[0])
            self.assertEqual(observed, rows)
            # Même un succès initial ne doit pas arrêter l'essai avant le premier point positif.
            return fake_point(rows, step, step == 0 or (stop_at is not None and step >= stop_at))
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(d3.campaign, "examples", return_value=rows), \
                patch.object(d3, "capture_training_diagnostics", side_effect=capture), \
                patch.object(d3, "train_epoch", side_effect=epoch), \
                patch.object(d3, "observation_point", side_effect=point), patch("builtins.print"):
            result = d3.training_loop(model, object(), rows, {}, {}, Path(tmp))
        for epoch_no in epoch_calls:
            actual = seen[(epoch_no-1)*4:epoch_no*4]
            expected = [[rows[i]["clip_id"] for i in ids]
                        for ids in d3.epoch_batches(32, 8, 20260912, epoch_no)]
            self.assertEqual(actual, expected)
        self.assertFalse(log_active[0])
        return result

    def test_first_positive_point_stops_and_exhaustion_never_extends_budget(self):
        early = self.run_mock_loop(100)
        self.assertEqual(early["steps"], 100)
        self.assertEqual(early["epochs"], 25)
        self.assertEqual(early["training_presentations"], 800)
        self.assertEqual(early["stop_reason"], "memorization_criterion")
        maximum = self.run_mock_loop(None)
        self.assertEqual(maximum["steps"], 1000)
        self.assertEqual(maximum["training_presentations"], 8000)
        self.assertEqual(maximum["observations"] + 32, 384)
        self.assertEqual(maximum["observation_forwards"] + 64, 768)
        self.assertEqual(maximum["stop_reason"], "budget_exhausted")

    def test_preregister_no_model_or_fit_and_context_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, rows = Path(tmp), rows32()
            original = {"output": root / "d0", "preregistration_sha256": "old-prereg",
                "registration": {"campaign": str(root / "cv"), "input_sha256": {"cache": "sha"},
                                 "campaign_identity": {"base": "sha"}},
                "receipts": {"A": {0: {"scoring_spec": {"original": True}}}}}
            receipt = {"receipt_sha256": d3.D0_RECEIPTS["A"],
                       "state_sha256": {"initial_before": {"encoder": "e", "projector": "p", "llm": "q"}}}
            args = SimpleNamespace(d0=original["output"], output=root / "d3", code_revision="a" * 40)
            with patch.object(d3, "checked_d0", return_value=(original, receipt)), \
                    patch.object(d3, "subset", return_value=(rows, {})), \
                    patch.object(d3, "source_hashes", return_value={"runner": "sha"}), \
                    patch.object(d3, "runtime_identity", return_value={"runtime": "sha"}), \
                    patch.object(d3, "default_optimizer_options", return_value=[{"lr": .0002}]), \
                    patch.object(d3.campaign, "initialize_training") as initialize:
                d3.preregister(args)
                initialize.assert_not_called()
                self.assertEqual(d3.context(args.output)["rows"], rows)
                with self.assertRaises(FileExistsError):
                    d3.preregister(args)
                with patch.object(d3, "source_hashes", return_value={"runner": "changed"}), self.assertRaises(ValueError):
                    d3.context(args.output)
                with patch.object(d3, "subset", return_value=(list(reversed(rows)), {})), self.assertRaises(ValueError):
                    d3.context(args.output)
                with patch.object(d3, "default_optimizer_options", return_value=[{"lr": .1}]), self.assertRaises(ValueError):
                    d3.context(args.output)
                args.output = original["output"] / "new"
                with self.assertRaisesRegex(ValueError, "reçus"):
                    d3.preregister(args)

    def test_reload_checks_weights_scores_decisions_criterion_and_no_optimizer(self):
        rows = rows32()
        state = {"encoder": "e", "projector": "p", "llm": "q"}
        point, scores = fake_point(rows, 100, True)
        for fault in (None, "weights", "scores", "decision", "criterion", "same_process"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                ctx = {"output": output, "preregistration_sha256": "prereg", "registration": {"scoring_spec": {}},
                       "rows": rows, "series": {}, "d0": {"campaign": {"registration": {"paths": {"base": "base"}}}}}
                trained = {"pid": d3.os.getpid() if fault == "same_process" else -1,
                    "hostname": d3.platform.node(), "steps": 100, "state_sha256_after": state,
                    "final_scores": dict(scores), "observation_points": [point],
                    "observations": 64, "observation_forwards": 128, "receipt_sha256": "receipt"}
                actual_scores, actual_point = dict(scores), dict(point)
                if fault == "scores":
                    actual_scores[rows[0]["clip_id"]] += .01
                if fault == "decision":
                    trained["final_scores"][rows[0]["clip_id"]] = .5 - 1e-8
                    actual_scores[rows[0]["clip_id"]] = .5 + 1e-8
                if fault == "criterion":
                    actual_point["criterion_met"] = False
                def finished(path, **kwargs):
                    return trained if Path(path).name == "train" else None
                model = dummy_model()
                with patch.object(d3.campaign, "finished", side_effect=finished), \
                        patch.object(d3.d0, "initialize", return_value=model) as initialize, \
                        patch.object(d3.d0, "load_terminal") as load, \
                        patch.object(d3, "validate_model"), patch.object(d3, "context", return_value=ctx), \
                        patch.object(d3.campaign, "variant_metadata", return_value={}), \
                        patch.object(d3.d0, "state_hashes", return_value=state if fault != "weights" else {}), \
                        patch.object(d3, "observation_point", return_value=(actual_point, actual_scores)), \
                        patch.object(d3.campaign, "initialize_training", side_effect=AssertionError("Aucun optimiseur")):
                    if fault is None:
                        result = d3.reload(ctx)
                        self.assertTrue(result["passed"])
                        self.assertEqual(result["optimizer_steps"], 0)
                        self.assertEqual(result["observation_forwards_including_training"], 192)
                        initialize.assert_called_once_with("base", "A")
                        load.assert_called_once_with(model, output / "train/temporal.pt")
                    else:
                        with self.assertRaises(ValueError):
                            d3.reload(ctx)
                        self.assertFalse((output / "reload/complete.json").exists())
                        if fault == "same_process":
                            initialize.assert_not_called()

    def test_incomplete_reload_is_preserved_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "reload").mkdir()
            ctx = {"output": output, "preregistration_sha256": "p"}
            with patch.object(d3.d0, "initialize") as initialize, self.assertRaisesRegex(ValueError, "incomplète"):
                d3.reload(ctx)
            initialize.assert_not_called()
            self.assertTrue((output / "reload").exists())


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("opentslm"),
                     "Chemin différentiable synthétique réservé au runtime ML CPU")
class TorchMemorizationChecks(unittest.TestCase):
    def test_initial_temporal_gradients_and_optimizer_state_must_be_empty(self):
        import torch
        from test_training_observations import TorchTrainingObservationChecks
        model, _ = TorchTrainingObservationChecks().make_model(False)
        model.llm.head.to(dtype=torch.bfloat16)
        model.llm.get_output_embeddings = lambda: model.llm.head
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": d3.CONFIG["encoder_lr"]},
            {"params": model.projector.parameters(), "lr": d3.CONFIG["projector_lr"]}], weight_decay=.01)
        expected = d3.default_optimizer_options()
        d3.validate_model(model, model.scoring_spec(), optimizer, expected)
        first = next(model.encoder.parameters())
        first.grad = torch.zeros_like(first)
        with self.assertRaisesRegex(ValueError, "AdamW neuf"):
            d3.validate_model(model, model.scoring_spec(), optimizer, expected)
        first.grad = None
        optimizer.state[first] = {"step": torch.tensor(1.)}
        with self.assertRaisesRegex(ValueError, "AdamW neuf"):
            d3.validate_model(model, model.scoring_spec(), optimizer, expected)

    def test_real_original_loss_and_logger_four_steps_without_extra_forwards(self):
        import torch
        from test_training_observations import TorchTrainingObservationChecks
        helper = TorchTrainingObservationChecks()
        torch.manual_seed(17)
        model, originals = helper.make_model(False)
        samples = [dict(originals[i % 2]) for i in range(32)]
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": d3.CONFIG["encoder_lr"]},
            {"params": model.projector.parameters(), "lr": d3.CONFIG["projector_lr"]}], weight_decay=.01)
        self.assertFalse(optimizer.state)
        self.assertEqual(d3.optimizer_options(optimizer), d3.default_optimizer_options())
        llm = {name: value.clone() for name, value in model.llm.state_dict().items()}
        before = model.projector.weight.detach().clone()
        reports = []
        with helper.collator(), d3.capture_training_diagnostics(model, optimizer) as audit:
            for record, indices in zip(d3.train_epoch(model, optimizer, samples, d3.CONFIG, 1),
                                       d3.epoch_batches(32, 8, 20260912, 1)):
                reports.append(audit.pop_step([f"synthetic-{i}" for i in indices], training_record=record))
            aggregate = audit.epoch_summary(reset=True)
        self.assertEqual((aggregate["n_steps"], aggregate["n_clips"]), (4, 32))
        self.assertEqual(model.llm.n_calls, 32)
        self.assertTrue(all(r["extra_forward_count"] == 0 for r in reports))
        self.assertFalse(torch.equal(model.projector.weight, before))
        for name, value in model.llm.state_dict().items():
            torch.testing.assert_close(value, llm[name], rtol=0, atol=0)
        self.assertNotIn("compute_loss", vars(model))


if __name__ == "__main__":
    unittest.main()
