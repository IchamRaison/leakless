"""Observation TRAIN : contrôles synthétiques seulement, aucun dataset ni GPU."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import training_observations as observation
from pipe.tslm.campaign import epoch_batches, train_epoch


class AggregationChecks(unittest.TestCase):
    def test_token_weighted_sum_and_no_discriminant_double_count(self):
        total = observation._empty_terms()
        for values in ([1., 2., 3., 4.], [5., 6., 7., 8., 9.]):
            count = len(values)
            masks = {"class": np.arange(count) < count - 2,
                     "first_discriminating_token": np.arange(count) == 0,
                     "description": np.arange(count) == count - 2,
                     "eos": np.arange(count) == count - 1,
                     "response": np.ones(count, dtype=bool)}
            observation._add_terms(total, observation.nll_terms(-np.asarray(values), masks))
        self.assertEqual(total["response"]["n_tokens"], 9)
        self.assertEqual(total["response"]["nll_sum"], 45)
        self.assertEqual(total["response"]["nll_mean"], 5)
        self.assertEqual(sum(total[key]["nll_sum"] for key in ("class", "description", "eos")), 45)
        self.assertEqual(total["first_discriminating_token"]["nll_sum"], 6)
        self.assertNotEqual((10/4 + 35/5)/2, total["response"]["nll_mean"])

    def test_zero_denominator_is_null_not_epsilon(self):
        self.assertIsNone(observation._relative_update(1, 0))
        self.assertEqual(observation._relative_update(0, 2), 0)
        self.assertEqual(observation._relative_update(1, 2), .5)
        empty = observation._empty_terms()
        observation._add_terms(empty, observation._empty_terms())
        self.assertIsNone(empty["description"]["nll_mean"])
        json.dumps(empty, allow_nan=False)

    def test_nonfinite_or_negative_nll_rejected(self):
        for value in (float("nan"), float("inf"), -1):
            terms = observation._empty_terms()
            terms["class"]["nll_sum"] = value
            with self.assertRaises(ValueError):
                observation._add_terms(observation._empty_terms(), terms)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("opentslm"),
                     "Tests PyTorch CPU à exécuter dans le runtime ML de référence")
class TorchTrainingObservationChecks(unittest.TestCase):
    def make_model(self, amplitude=False):
        import torch
        from opentslm.model.llm.TimeSeriesLLM import TimeSeriesLLM
        from pipe.tslm.model import AcousticQwenSP

        class Tokenizer:
            pad_token_id = 0
            table = {"leak;": [1, 3], "no_leak;": [2, 4, 3], "<eos>": [6],
                     "leak; D<eos>": [1, 3, 5, 6], "no_leak; D<eos>": [2, 4, 3, 5, 6]}
            def encode(self, text, **kwargs):
                return self.table[text]
            def decode(self, ids):
                return next(text for text in ("leak;", "no_leak;") if self.table[text] == list(ids))
            def __call__(self, answers, **kwargs):
                ids = [self.encode(answer) for answer in answers]
                width = max(map(len, ids))
                return SimpleNamespace(input_ids=torch.tensor([x + [0]*(width-len(x)) for x in ids]),
                    attention_mask=torch.tensor([[1]*len(x)+[0]*(width-len(x)) for x in ids]))

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(7, 2)
                self.head = torch.nn.Linear(2, 7)
                self.dropout = torch.nn.Dropout(.5)
                self.requires_grad_(False)
                self.n_calls = 0
            def get_input_embeddings(self):
                return self.embedding
            def forward(self, *, inputs_embeds, attention_mask, labels, use_cache, return_dict):
                self.n_calls += 1
                if hasattr(self, "lora_adapter"):
                    inputs_embeds = inputs_embeds + self.lora_adapter(inputs_embeds)
                logits = self.head(self.dropout(inputs_embeds.cumsum(dim=1)))
                loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 7),
                                                        labels[:, 1:].reshape(-1), ignore_index=-100)
                self.last_loss = loss  # Oracle d'identité du tenseur, hors observateur.
                return SimpleNamespace(logits=logits, loss=loss)

        class Model(AcousticQwenSP):
            def __init__(self):
                TimeSeriesLLM.__init__(self, "cpu")
                self.single_clip_acoustic_encoding = True
                self.amplitude_evidence = amplitude
                self.tokenizer, self.llm = Tokenizer(), Decoder()
                self.encoder = torch.nn.Sequential(torch.nn.Linear(1, 4, bias=False), torch.nn.Dropout(.15))
                self.projector = torch.nn.Linear(4, 2, bias=False)
            def get_eos_token(self):
                return "<eos>"
            def pad_and_apply_batch(self, batch):
                if len(batch) > 1:
                    parts = [self.pad_and_apply_batch([sample]) for sample in batch]
                    return (torch.nn.utils.rnn.pad_sequence([x[0] for x, _ in parts], batch_first=True),
                            torch.nn.utils.rnn.pad_sequence([m[0] for _, m in parts], batch_first=True))
                self.assert_batch_one = len(batch)
                sample = batch[0]
                projected = self.projector(self.encoder(sample["time_series"].reshape(4, 1))).reshape(1, 4, 2)
                text = self.llm.get_input_embeddings()(torch.zeros((1, len(sample["pre_prompt"])), dtype=torch.long))
                prefix = torch.cat((text, projected), dim=1)
                # Le vrai compute_loss doit enlever cette sentinelle de padding.
                return torch.cat((prefix, torch.full((1, 1, 2), 99.)), dim=1), torch.tensor([[1]*prefix.shape[1]+[0]])
        model = Model()
        model.eval()
        model.llm.train()  # train_epoch doit légitimement inverser ces modes.
        samples = [{"pre_prompt": "P"*(i % 3+1), "post_prompt": "", "time_series_text": [""]*4,
                    "time_series": torch.tensor([.3, .4, .5, .6]) + i*.05,
                    "answer": "no_leak; D" if i % 2 else "leak; D"} for i in range(9)]
        return model, samples

    def optimizer(self, model):
        import torch
        return torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": .0002},
            {"params": model.projector.parameters(), "lr": .0001},
        ], weight_decay=.01, eps=3e-7, betas=(.85, .97), amsgrad=True)

    def config(self):
        return {"batch_size": 8, "microbatch_size": 1, "seed": 20260912, "gradient_clip": .0001}

    def collator(self):
        return patch.dict(sys.modules, {"opentslm.time_series_datasets.util": SimpleNamespace(
            extend_time_series_to_match_patch_size_and_aggregate=lambda samples, normalize: samples if normalize is False else None)})

    def test_real_compute_loss_train_epoch_bitwise_updates_rng_modes_and_weighting(self):
        import torch
        for amplitude in (False, True):
            with self.subTest(amplitude=amplitude):
                torch.manual_seed(51)
                model, samples = self.make_model(amplitude)
                reference = copy.deepcopy(model)
                opt, refopt = self.optimizer(model), self.optimizer(reference)
                initial_hooks = (len(opt._optimizer_step_pre_hooks), len(opt._optimizer_step_post_hooks))
                def reset_rng():
                    torch.manual_seed(791)
                    random.seed(792)
                    np.random.seed(793)
                reset_rng()
                with self.collator():
                    wanted = list(train_epoch(reference, refopt, samples, self.config(), 1))
                wanted_rng = (torch.get_rng_state(), random.getstate(), np.random.get_state())
                reset_rng()
                actual, reports = [], []
                with self.collator(), observation.capture_training_diagnostics(model, opt) as audit:
                    for record, ids in zip(train_epoch(model, opt, samples, self.config(), 1),
                                           epoch_batches(9, 8, 20260912, 1)):
                        actual.append(record)
                        reports.append(audit.pop_step([f"train-{i}" for i in ids], training_record=record))
                        self.assertIsNone(audit.completed)
                        self.assertFalse(audit.pending)
                        self.assertIsNone(audit.before)
                    epoch = audit.epoch_summary(reset=True)
                    self.assertEqual(audit.epoch_summary()["n_clips"], 0)
                self.assertEqual(actual, wanted)
                self.assertTrue(torch.equal(torch.get_rng_state(), wanted_rng[0]))
                self.assertEqual(random.getstate(), wanted_rng[1])
                self.assertEqual(np.random.get_state()[0], wanted_rng[2][0])
                np.testing.assert_array_equal(np.random.get_state()[1], wanted_rng[2][1])
                self.assertEqual(np.random.get_state()[2:], wanted_rng[2][2:])
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
                for p, ref in zip(model.parameters(), reference.parameters()):
                    if p.requires_grad:
                        torch.testing.assert_close(p.grad, ref.grad, rtol=0, atol=0)
                        for key, value in opt.state[p].items():
                            torch.testing.assert_close(value, refopt.state[ref][key], rtol=0, atol=0)
                    else:
                        self.assertIsNone(p.grad)
                self.assertEqual(model.llm.n_calls, 9)
                self.assertEqual(reference.llm.n_calls, 9)
                self.assertTrue(model.encoder.training and model.projector.training)
                self.assertFalse(model.llm.training)
                self.assertNotIn("compute_loss", vars(model))
                self.assertNotIn("forward", vars(model.llm))
                self.assertEqual(initial_hooks, (len(opt._optimizer_step_pre_hooks), len(opt._optimizer_step_post_hooks)))
                self.assertEqual([len(r["clips"]) for r in reports], [8, 1])
                self.assertEqual(epoch["n_clips"], 9)
                self.assertEqual(epoch["n_steps"], 2)
                self.assertEqual(epoch["terms"]["response"]["n_tokens"], 40)
                self.assertEqual(epoch["terms"]["first_discriminating_token"]["n_tokens"], 9)
                for key in observation.TERMS:
                    self.assertAlmostEqual(epoch["terms"][key]["nll_sum"],
                                           sum(r["terms"][key]["nll_sum"] for r in reports), places=10)
                for report in reports:
                    self.assertIsNone(report["binary_nll"])
                    self.assertEqual(report["extra_forward_count"], 0)
                    self.assertEqual(report["gradient_norms_pre_clip"], report["training_record"]["gradient_norms"])
                    self.assertGreater(math.hypot(*report["gradient_norms_pre_clip"].values()), .0001)
                    self.assertLessEqual(math.hypot(*report["gradient_norms_post_clip"].values()), .00010001)
                    for name in observation.COMPONENTS:
                        self.assertLess(report["gradient_norms_post_clip"][name], report["gradient_norms_pre_clip"][name])
                        update = report["parameter_updates"][name]
                        self.assertGreater(update["update_l2"], 0)
                        self.assertEqual(update["relative_update"], update["update_l2"]/update["parameter_l2_before"])
                        self.assertEqual(report["optimizer"]["dtypes"][name]["parameters"], ["torch.float32"])
                    for group in report["optimizer"]["param_groups"]:
                        self.assertEqual(group["options"]["eps"], 3e-7)
                        self.assertEqual(group["options"]["betas"], [.85, .97])
                    for clip in report["clips"]:
                        self.assertLess(clip["reconstruction_absolute_difference"], 1e-6)
                        self.assertTrue(clip["labels_and_attention_verified"])
                        self.assertEqual(clip["first_causal_logit_position"], clip["prompt_length"] - 1)
                self.assertEqual(reports[0]["optimizer"]["state_before"]["encoder"]["parameters_with_state"], 0)
                self.assertEqual(reports[1]["optimizer"]["state_after"]["encoder"]["step_max"], 2)
                json.dumps({"steps": reports, "epoch": epoch}, allow_nan=False)

    def test_exact_tensor_return_no_extra_forward_and_cleanup_on_body_failure(self):
        import torch
        torch.manual_seed(81)
        model, samples = self.make_model()
        opt = self.optimizer(model)
        modes = [module.training for module in model.modules()]
        initial_rng = torch.get_rng_state().clone()
        class Stop(Exception):
            pass
        with self.assertRaises(Stop):
            with observation.capture_training_diagnostics(model, opt) as audit:
                self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
                loss = model.compute_loss(samples[:1])
                self.assertIs(loss, model.llm.last_loss)
                self.assertTrue(loss.requires_grad)
                self.assertEqual(model.llm.n_calls, 1)
                self.assertEqual(len(audit.pending), 1)
                # Les observations sont sérialisables ; aucune sortie/graph capturé.
                json.dumps(audit.pending, allow_nan=False)
                raise Stop()
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertNotIn("compute_loss", vars(model))
        self.assertNotIn("forward", vars(model.llm))
        self.assertFalse(opt._optimizer_step_pre_hooks or opt._optimizer_step_post_hooks)
        self.assertFalse(audit.pending)
        self.assertIsNone(audit.before)

    def test_capture_failure_restores_existing_forward_and_compute_loss_overrides(self):
        model, samples = self.make_model()
        opt = self.optimizer(model)
        original = model.compute_loss
        model.compute_loss = lambda batch: original(batch)
        existing_compute = model.compute_loss
        def broken(**kwargs):
            raise RuntimeError("décodeur indisponible")
        model.llm.forward = broken
        with self.assertRaisesRegex(RuntimeError, "décodeur indisponible"):
            with observation.capture_training_diagnostics(model, opt):
                model.compute_loss(samples[:1])
        self.assertIs(model.compute_loss, existing_compute)
        self.assertIs(model.llm.forward, broken)
        self.assertFalse(opt._optimizer_step_pre_hooks or opt._optimizer_step_post_hooks)

    def test_bad_labels_attention_and_batch_fail_explicitly(self):
        for fault in ("labels", "attention_mask"):
            model, samples = self.make_model()
            opt = self.optimizer(model)
            original = model.llm.forward
            def wrong(_fault=fault, **kwargs):
                output = original(**kwargs)
                kwargs[_fault][0, 0] = 1 if _fault == "labels" else 0
                return output
            model.llm.forward = wrong
            with self.assertRaisesRegex(ValueError, "Positions causales"):
                with observation.capture_training_diagnostics(model, opt):
                    model.compute_loss(samples[:1])
            self.assertIs(model.llm.forward, wrong)
            self.assertFalse(opt._optimizer_step_pre_hooks or opt._optimizer_step_post_hooks)
        model, samples = self.make_model()
        opt = self.optimizer(model)
        with self.assertRaises(ValueError):
            with observation.capture_training_diagnostics(model, opt):
                model.compute_loss(samples[:2])
        self.assertEqual(model.llm.n_calls, 0)

    def test_buffer_is_bounded_and_requires_consumption_before_next_forward(self):
        model, samples = self.make_model()
        opt = self.optimizer(model)
        with self.assertRaisesRegex(RuntimeError, "Consommer le step précédent"):
            with self.collator(), observation.capture_training_diagnostics(model, opt) as audit:
                iterator = train_epoch(model, opt, samples, self.config(), 1)
                next(iterator)  # Huit observations, step réel, délibérément non consommé.
                self.assertEqual(len(audit.completed["clips"]), 8)
                self.assertIsNone(audit.before)
                self.assertFalse(audit.pending)
                with self.assertRaises(RuntimeError):
                    audit.epoch_summary()
                next(iterator)
        self.assertEqual(model.llm.n_calls, 8)
        self.assertFalse(opt._optimizer_step_pre_hooks or opt._optimizer_step_post_hooks)
        self.assertIsNone(audit.completed)
        model, samples = self.make_model()
        opt = self.optimizer(model)
        with self.assertRaisesRegex(RuntimeError, "Consommer le step précédent"):
            with observation.capture_training_diagnostics(model, opt, max_clips_per_step=1):
                model.compute_loss(samples[:1])
                model.compute_loss(samples[:1])
        self.assertEqual(model.llm.n_calls, 1)

    def test_mismatched_record_does_not_consume_or_double_count(self):
        model, samples = self.make_model()
        opt = self.optimizer(model)
        with self.collator(), observation.capture_training_diagnostics(model, opt) as audit:
            record = next(train_epoch(model, opt, samples[:1], self.config(), 1))
            for invalid in ({**record, "supervised_tokens": 999}, {**record, "loss": record["loss"] + .1}):
                with self.assertRaises(ValueError):
                    audit.pop_step([0], training_record=invalid)
                self.assertEqual(audit.n_clips, 0)
            audit.pop_step([0], training_record=record)
            with self.assertRaises(RuntimeError):
                audit.pop_step([0], training_record=record)
            self.assertEqual(audit.epoch_summary()["n_clips"], 1)

    def test_opt_in_lora_and_mixed_microbatches_preserve_updates_and_count_every_clip(self):
        import torch
        for microbatch in (4, 2, 1):
            torch.manual_seed(52)
            model, samples = self.make_model()
            model.llm.lora_adapter = torch.nn.Linear(2, 2, bias=False)
            model.lora_enabled = True
            reference = copy.deepcopy(model)
            opt, refopt = self.optimizer(model), self.optimizer(reference)
            opt.add_param_group({"params": model.get_lora_parameters()})
            refopt.add_param_group({"params": reference.get_lora_parameters()})
            config = {**self.config(), "microbatch_size": microbatch, "lora": {"r": 8}}
            torch.manual_seed(83)
            with self.collator():
                expected = list(train_epoch(reference, refopt, samples, config, 1))
            torch.manual_seed(83)
            with self.collator(), observation.capture_training_diagnostics(
                    model, opt, allow_lora=True, allow_microbatches=True) as audit:
                actual = []
                for record, ids in zip(train_epoch(model, opt, samples, config, 1),
                                      epoch_batches(len(samples), 8, config["seed"], 1)):
                    actual.append(record)
                    result = audit.pop_step(ids, training_record=record)
                    self.assertEqual(set(result["gradient_norms_pre_clip"]), {"encoder", "projector", "lora"})
                    self.assertEqual(len(result["clips"]), len(ids))
                    self.assertTrue(model.llm.training)
                    for clip in result["clips"]:
                        self.assertLess(clip["reconstruction_absolute_difference"], 1e-6)
                self.assertEqual(audit.epoch_summary()["n_clips"], 9)
            self.assertEqual(actual, expected)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(model.llm.n_calls, math.ceil(8/microbatch) + 1)
            self.assertTrue(all(p.grad is None for name, p in model.llm.named_parameters() if "lora_" not in name))


if __name__ == "__main__":
    unittest.main()
