"""Observations causales synthétiques ; aucun dataset, téléchargement ou GPU."""
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import causal_observations as observation


class Tokenizer:
    pad_token_id = 0
    table = {"leak;": [1, 3], "no_leak;": [2, 4, 3], "<eos>": [6],
             "leak; D<eos>": [1, 3, 5, 6], "no_leak; D<eos>": [2, 4, 3, 5, 6],
             "leak;<eos>": [1, 3, 6]}

    def encode(self, text, **kwargs):
        return self.table[text]

    def decode(self, ids):
        for text in ("leak;", "no_leak;"):
            if self.table[text] == list(ids):
                return text
        return {0: "pad", 1: "leak", 2: "no", 3: ";", 4: "_leak", 5: " D", 6: "<eos>"}[ids[0]]

    def __call__(self, answers, **kwargs):
        import torch
        ids = [self.encode(answer) for answer in answers]
        return SimpleNamespace(input_ids=torch.tensor(ids), attention_mask=torch.ones((len(ids), len(ids[0])), dtype=torch.long))


def specification():
    return {"class_continuations": ["leak;", "no_leak;"], "class_token_ids": [[1, 3], [2, 4, 3]]}


class PureObservationChecks(unittest.TestCase):
    def test_disjoint_class_description_eos_and_overlapping_discriminant(self):
        for answer, nclass, target in (("leak; D", 2, 0), ("no_leak; D", 3, 1)):
            got, ids, masks, first = observation.answer_partition(Tokenizer(), specification(), answer, "<eos>")
            self.assertEqual(got, target)
            self.assertEqual(first, 0)
            self.assertEqual(int(masks["class"].sum()), nclass)
            self.assertEqual(int(masks["description"].sum()), 1)
            self.assertEqual(int(masks["eos"].sum()), 1)
            self.assertEqual(int(masks["first_discriminating_token"].sum()), 1)
            values = -np.arange(1, len(ids)+1, dtype=float)
            terms = observation.nll_terms(values, masks)
            self.assertEqual(sum(terms[key]["nll_sum"] for key in ("class", "description", "eos")),
                             terms["response"]["nll_sum"])
            self.assertEqual(terms["response"]["n_tokens"], len(ids))
            self.assertEqual(terms["first_discriminating_token"]["nll_sum"], 1)

    def test_shared_prefix_discriminant_and_empty_description(self):
        tokenizer = Tokenizer()
        tokenizer.table = {**tokenizer.table, "leak; D<eos>": [7, 1, 3, 5, 6]}
        spec = {**specification(), "class_token_ids": [[7, 1, 3], [7, 2, 4, 3]]}
        _, _, masks, first = observation.answer_partition(tokenizer, spec, "leak; D", "<eos>")
        self.assertEqual(first, 1)
        self.assertEqual(np.flatnonzero(masks["first_discriminating_token"]).tolist(), [1])
        _, ids, masks, _ = observation.answer_partition(Tokenizer(), specification(), "leak;", "<eos>")
        self.assertIsNone(observation.nll_terms(np.zeros(len(ids)), masks)["description"]["nll_mean"])
        with self.assertRaises(ValueError):
            observation.answer_partition(Tokenizer(), {**specification(), "class_token_ids": [[1], [1, 2]]}, "leak;", "<eos>")

    def test_binary_nll_stable_and_not_nll_of_answer_or_first_token(self):
        self.assertAlmostEqual(observation.binary_nll([math.log(.2), math.log(.1)], 0), -math.log(2/3))
        self.assertAlmostEqual(observation.binary_nll([math.log(.2), math.log(.1)], 1), -math.log(1/3))
        self.assertAlmostEqual(observation.binary_nll([-1e20, -1e20], 0), math.log(2))
        self.assertAlmostEqual(observation.binary_nll([-1000, -1001], 0), math.log1p(math.exp(-1)))
        for values in ([float("nan"), -1], [-1], [float("inf"), -1]):
            with self.assertRaises(ValueError):
                observation.binary_nll(values, 0)
        with self.assertRaises(ValueError):
            observation.nll_terms([float("nan")], {"response": [True]})

    def test_norm_summaries_preserve_dtype_and_report_only_small_statistics(self):
        value = observation.norm_summary(np.array([[3, 4], [0, 0]], dtype=np.float32), "torch.bfloat16")
        self.assertEqual(value["l2_mean"], 2.5)
        self.assertEqual(value["source_dtype"], "torch.bfloat16")
        self.assertEqual(value["n_vectors"], 2)
        self.assertEqual(len(value["float32_values_sha256"]), 64)
        json.dumps(value, allow_nan=False)
        with self.assertRaises(ValueError):
            observation.norm_summary(np.array([[float("nan")]]), "float32")


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("opentslm"),
                     "Tests PyTorch CPU à exécuter dans le runtime ML de référence")
class TorchObservationChecks(unittest.TestCase):
    def make_model(self, amplitude=False):
        import torch
        from opentslm.model.llm.TimeSeriesLLM import TimeSeriesLLM
        from pipe.tslm.model import AcousticQwenSP
        from pipe.tslm.preprocessing import AMPLITUDE_TEXT_PREFIX

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding.from_pretrained(torch.arange(7).float()[:, None])
                probabilities = torch.full((7, 7), 1/7)
                probabilities[0] = torch.tensor([.05, .5, .25, .05, .05, .05, .05])
                for source, target, value in ((1, 3, .4), (2, 4, .8), (4, 3, .5), (3, 5, .9), (5, 6, .95)):
                    probabilities[source].fill_((1-value)/6)
                    probabilities[source, target] = value
                self.register_buffer("probabilities", probabilities)
                self.n_calls = 0
            def get_input_embeddings(self):
                return self.embedding
            def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict, labels=None):
                self.n_calls += 1
                logits = self.probabilities.log()[inputs_embeds[..., 0].long()]
                loss = (torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 7),
                        labels[:, 1:].reshape(-1), ignore_index=-100) if labels is not None else None)
                return SimpleNamespace(logits=logits, loss=loss)

        class Model(AcousticQwenSP):
            def __init__(self):
                TimeSeriesLLM.__init__(self, "cpu")
                self.single_clip_acoustic_encoding = True
                self.amplitude_evidence = amplitude
                self.tokenizer, self.llm = Tokenizer(), Decoder()
                self.encoder, self.projector = torch.nn.Identity(), torch.nn.Identity()
            def get_eos_token(self):
                return "<eos>"
            def pad_and_apply_batch(self, batch):
                projected = self.projector(self.encoder(torch.zeros((4, 1, 1))))
                text = self.llm.get_input_embeddings()(torch.zeros((1, 3 if amplitude else 1), dtype=torch.long))
                prefix = torch.cat((text, projected.reshape(1, 4, 1)), dim=1)
                # Sentinelle paddée qui ne doit être ni supervisée ni scorée.
                return torch.cat((prefix, torch.tensor([[[99.]]])), dim=1), torch.tensor([[1]*prefix.shape[1]+[0]])
        model = Model()
        model.train()
        model.encoder.eval()  # Vérifier aussi la restauration de modes mixtes.
        batch = [{"pre_prompt": AMPLITUDE_TEXT_PREFIX if amplitude else "fixture", "post_prompt": "",
                  "time_series": torch.zeros((4, 64)), "time_series_text": [""]*4, "answer": "leak; D"}]
        return model, batch

    def test_real_loss_and_official_score_two_forwards_a_and_c_restored(self):
        for amplitude in (False, True):
            model, batch = self.make_model(amplitude)
            modes = [module.training for module in model.modules()]
            original = model.llm.forward
            result = observation.observe_clip(model, batch)
            self.assertEqual(model.llm.n_calls, 2)
            self.assertEqual(result["llm_forward_counts"], {"supervision": 1, "scoring": 1})
            self.assertAlmostEqual(result["scoring"]["probability_leak"], 2/3, places=6)
            self.assertAlmostEqual(result["scoring"]["binary_nll"], -math.log(2/3), places=6)
            self.assertAlmostEqual(result["loss"]["terms"]["class"]["nll_sum"], -math.log(.2), places=6)
            self.assertEqual(result["loss"]["terms"]["response"]["n_tokens"], 4)
            self.assertTrue(result["loss"]["matches_atol_1e_6"])
            self.assertTrue(result["alignment"]["prompts_exact"])
            self.assertEqual(result["alignment"]["class_token_logprob_max_abs_difference"], 0)
            self.assertFalse(result["scoring"]["tokens"][0][-1]["included"])
            self.assertIsNone(result["scoring"]["tokens"][0][-1]["logprob"])
            self.assertEqual(result["loss"]["tokens"][0]["causal_logit_position"], result["loss"]["prompt_length"]-1)
            self.assertEqual(modes, [module.training for module in model.modules()])
            self.assertEqual(original, model.llm.forward)
            self.assertNotIn("score_class_logprobs", vars(model))
            self.assertTrue(all(not module._forward_hooks for module in model.modules()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            json.dumps(result, allow_nan=False)

    def test_no_leak_and_failure_restore_modes_hooks_methods(self):
        model, batch = self.make_model()
        batch[0]["answer"] = "no_leak; D"
        result = observation.observe_clip(model, batch)
        self.assertEqual(result["target_class_index"], 1)
        self.assertEqual(result["loss"]["terms"]["class"]["n_tokens"], 3)
        self.assertAlmostEqual(result["scoring"]["binary_nll"], math.log(3), places=6)
        modes = [module.training for module in model.modules()]
        original = model.llm.forward
        model.llm.probabilities[0, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "loss finie"):
            observation.observe_clip(model, batch)
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertEqual(original, model.llm.forward)
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def test_batch_and_metadata_rejected_before_forward(self):
        model, batch = self.make_model()
        for invalid in ([], batch*2, [{**batch[0], "clip_id": "forbidden"}]):
            with self.assertRaises(ValueError):
                observation.observe_clip(model, invalid)
        self.assertEqual(model.llm.n_calls, 0)


if __name__ == "__main__":
    unittest.main()
