"""Extraction V1 : mêmes mises à jour CPU et arrêt avant step sur gradient invalide."""
import copy
import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipe.tslm.campaign import epoch_batches, train_epoch


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch requis pour le contrôle autograd CPU")
class TrainingEpochChecks(unittest.TestCase):
    def setUp(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(1, 1, bias=False)
                self.projector = torch.nn.Linear(1, 1, bias=False)
                self.llm = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
                for parameter in self.parameters():
                    torch.nn.init.constant_(parameter, 0.5)
                self.seen = []

            def compute_loss(self, batch):
                self.seen.extend(sample["index"] for sample in batch)
                values = torch.tensor([[sample["value"]] for sample in batch])
                return self.llm(self.projector(self.encoder(values))).square().mean()

        self.torch, self.model = torch, Model()
        self.config = {"batch_size": 2, "seed": 19, "gradient_clip": 0.1}
        self.samples = [{"index": i, "value": float(i + 1)} for i in range(5)]
        self.collator = SimpleNamespace(extend_time_series_to_match_patch_size_and_aggregate=
                                      lambda batch, normalize: batch if normalize is False else None)

    def optimizer(self, model):
        return self.torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": 0.0002},
            {"params": model.projector.parameters(), "lr": 0.0001},
        ], weight_decay=0.01)

    def test_matches_inline_v1_updates_records_order_and_frozen_decoder(self):
        torch = self.torch
        legacy, original = copy.deepcopy(self.model), copy.deepcopy(self.model.state_dict())
        optimizer = self.optimizer(legacy)
        expected = []
        legacy.train()
        legacy.llm.eval()
        # Oracle : boucle V1 avant extraction, sur une époque et un dernier lot partiel.
        for indices in epoch_batches(len(self.samples), 2, 19, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = legacy.compute_loss([{**self.samples[i]} for i in indices])
            loss.backward()
            gradients = {name: torch.sqrt(sum(p.grad.float().square().sum()
                         for p in module.parameters())).item()
                         for name, module in (("encoder", legacy.encoder), ("projector", legacy.projector))}
            torch.nn.utils.clip_grad_norm_([p for p in legacy.parameters() if p.requires_grad],
                                          0.1, error_if_nonfinite=True)
            optimizer.step()
            expected.append({"epoch": 1, "batch_samples": len(indices), "loss": loss.item(),
                             "gradient_norms": gradients})
        with patch.dict(sys.modules, {"opentslm.time_series_datasets.util": self.collator}):
            actual = list(train_epoch(self.model, self.optimizer(self.model), self.samples, self.config, 1))
        self.assertEqual(actual, expected)
        self.assertEqual(self.model.seen, legacy.seen)
        self.assertEqual(sorted(self.model.seen), list(range(5)))
        self.assertEqual([r["batch_samples"] for r in actual], [2, 2, 1])
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, legacy.state_dict()[name], rtol=0, atol=0)
            if name.startswith("llm."):
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)
            else:
                self.assertFalse(torch.equal(value, original[name]))
        self.assertTrue(self.model.encoder.training and self.model.projector.training)
        self.assertFalse(self.model.llm.training)
        self.assertTrue(all(p.grad is None for p in self.model.llm.parameters()))

    def test_invalid_loss_missing_or_zero_gradient_never_steps(self):
        cases = (
            ("Loss non finie", lambda batch: self.model.encoder.weight.sum() * float("nan")),
            ("Gradients invalides : projector", lambda batch: self.model.encoder.weight.sum()),
            ("Gradient nul : encoder", lambda batch: (self.model.encoder.weight.sum()
                                                       + self.model.projector.weight.sum()) * 0),
        )
        for message, compute_loss in cases:
            with self.subTest(message=message):
                optimizer = self.optimizer(self.model)
                with patch.dict(sys.modules, {"opentslm.time_series_datasets.util": self.collator}), \
                        patch.object(self.model, "compute_loss", compute_loss), \
                        patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    with self.assertRaisesRegex(RuntimeError, message):
                        list(train_epoch(self.model, optimizer, self.samples, self.config, 1))
                    step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
