"""Contrat externe opt-in et non-régression V1, sur fixtures synthétiques seules."""
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from harness.contract import ContractError, load_run, write_run  # noqa: E402
from harness.split_loader import Clip, FROZEN_SPLIT_NAME, FROZEN_SPLIT_SHA256, Split  # noqa: E402


class ExternalContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "external_synthetic_v1.json"
        self.manifest.write_text('{"fixture": "synthetic only"}\n')
        self.digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.split = Split(clips=(Clip("opaque-a", 1, "leak", "condition-a", "external"),
                                  Clip("opaque-b", 0, "no_leak", "condition-b", "external")),
                           sha256=self.digest, manifest_path=self.manifest, _paths={})
        self.identity = {"external_manifest_name": self.manifest.name,
                         "external_manifest_sha256": self.digest}
        self.probabilities = {"opaque-a": 1.2345678901234567e-12, "opaque-b": .12345678901234567}

    def write(self, directory="run", **changes):
        kwargs = dict(run_id="synthetic", model_name="synthetic-model", checkpoint="frozen",
                      training_commit="a" * 40, split=self.split,
                      threshold_rule="Fixed elsewhere; no threshold fitting by this validator",
                      probabilities=self.probabilities, **self.identity)
        kwargs.update(changes)
        return write_run(self.root / directory, **kwargs)

    def load(self, directory="run", **changes):
        kwargs = dict(folds=("external",), **self.identity)
        kwargs.update(changes)
        return load_run(self.root / directory, self.split, **kwargs)

    def test_round_trip_precision_identity_and_separate_targets(self):
        run_dir = self.write(extra={"config_hash": "b" * 64})
        run = self.load()
        self.assertEqual(run.probabilities, self.probabilities)
        self.assertEqual(run.metadata["split_filename"], self.manifest.name)
        self.assertEqual(run.metadata["split_sha256"], self.digest)
        self.assertEqual(run.metadata["config_hash"], "b" * 64)
        self.assertEqual({p.name for p in run_dir.iterdir()}, {"metadata.json", "predictions.csv"})
        with (run_dir / "predictions.csv").open() as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ["clip_id", "probability_leak"])
        self.assertTrue(all(len(row) == 2 for row in rows))
        self.assertEqual(len(self.split.fold("external")), 2)
        self.assertTrue(any(c.name == "octets du manifeste externe épinglés" for c in run.checks))

    def test_external_opt_in_and_fold_are_required(self):
        self.write()
        for require_sha in (True, False):
            with self.subTest(require_frozen_sha=require_sha), self.assertRaises(ContractError):
                load_run(self.root / "run", self.split, require_frozen_sha=require_sha)
        with self.assertRaisesRegex(ContractError, "scope vide"):
            load_run(self.root / "run", self.split, **self.identity)

    def test_both_valid_identity_arguments_are_required(self):
        cases = [dict(external_manifest_name=None), dict(external_manifest_sha256=None),
                 dict(external_manifest_name="split_v2.csv"),
                 dict(external_manifest_name="external_SPLIT_V1.csv"),
                 dict(external_manifest_name="../external.json"),
                 dict(external_manifest_sha256="short"),
                 dict(external_manifest_sha256=FROZEN_SPLIT_SHA256),
                 dict(external_manifest_name="different.json"),
                 dict(external_manifest_sha256="f" * 64)]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                self.write(**changes)
            self.assertFalse((self.root / "run").exists())

    def test_actual_manifest_bytes_are_pinned_before_read_or_write(self):
        self.write()
        self.manifest.write_text('{"fixture": "mutated after freeze"}\n')
        with self.assertRaisesRegex(ContractError, "Octets"):
            self.load(require_frozen_sha=False)
        with self.assertRaisesRegex(ContractError, "Octets"):
            self.write("new")
        self.assertFalse((self.root / "new").exists())

    def test_metadata_cannot_replace_the_pinned_manifest_or_tuning_declaration(self):
        self.write()
        meta_path = self.root / "run/metadata.json"
        original = json.loads(meta_path.read_text())
        for field, value in (("split_filename", FROZEN_SPLIT_NAME),
                             ("split_sha256", "e" * 64),
                             ("test_labels_not_used_for_tuning", False)):
            with self.subTest(field=field):
                meta_path.write_text(json.dumps({**original, field: value}))
                with self.assertRaises(ContractError):
                    self.load(require_frozen_sha=False)
                with self.assertRaises(ContractError):
                    self.write("new", extra={field: value})
                self.assertFalse((self.root / "new").exists())

    def test_external_columns_must_be_exact_and_ordered(self):
        self.write()
        cases = ["clip_id,probability_leak,comment\nopaque-a,.1,x\nopaque-b,.2,x\n",
                 "probability_leak,clip_id\n.1,opaque-a\n.2,opaque-b\n",
                 "clip_id,probability_leak,probability_leak\nopaque-a,.1,.1\nopaque-b,.2,.2\n",
                 "clip_id,probability_leak\nopaque-a,.1,extra\nopaque-b,.2\n",
                 "clip_id,probability_leak,label\nopaque-a,.1,1\nopaque-b,.2,0\n"]
        for text in cases:
            with self.subTest(header=text.splitlines()[0]):
                (self.root / "run/predictions.csv").write_text(text)
                with self.assertRaisesRegex(ContractError, "colonnes externes exactes"):
                    self.load()

    def test_missing_duplicate_unknown_and_out_of_scope_ids_fail(self):
        self.write()
        for rows in ("opaque-a,.1\n", "opaque-a,.1\nopaque-a,.2\n",
                     "opaque-a,.1\nunknown,.2\n"):
            with self.subTest(rows=rows):
                (self.root / "run/predictions.csv").write_text("clip_id,probability_leak\n" + rows)
                with self.assertRaises(ContractError):
                    self.load()
        with self.assertRaises(ContractError):
            self.write("new", probabilities={"opaque-a": .1})
        self.split = replace(self.split, clips=(self.split.clips[0],
                                               replace(self.split.clips[1], fold="background")))
        self.write("all")
        with self.assertRaisesRegex(ContractError, "hors scope"):
            self.load("all")

    def test_invalid_probabilities_fail_without_replacements(self):
        self.write()
        for value in (float("nan"), float("inf"), -0.1, 1.1, "", None, True):
            with self.subTest(value=value), self.assertRaises(ContractError):
                self.write("new", probabilities={"opaque-a": value, "opaque-b": .2})
            self.assertFalse((self.root / "new").exists())
        for token in ("nan", "inf", "-0.1", "1.1", "", "nonnumeric"):
            with self.subTest(token=token):
                (self.root / "run/predictions.csv").write_text(
                    f"clip_id,probability_leak\nopaque-a,{token}\nopaque-b,.2\n")
                with self.assertRaises(ContractError):
                    self.load()

    def test_existing_outputs_are_never_overwritten(self):
        self.write()
        before = {p.name: p.read_bytes() for p in (self.root / "run").iterdir()}
        with self.assertRaises(ContractError):
            self.write()
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.root / "run").iterdir()})
        (self.root / "empty").mkdir()
        with self.assertRaises(ContractError):
            self.write("empty")
        self.assertEqual(list((self.root / "empty").iterdir()), [])
        (self.root / "link").symlink_to(self.root / "absent")
        with self.assertRaises(ContractError):
            self.write("link")
        self.assertFalse((self.root / "absent").exists())

    def test_empty_or_duplicate_manifest_scope_is_rejected(self):
        for clips in ((), (self.split.clips[0], self.split.clips[0])):
            with self.subTest(clips=clips), self.assertRaises(ContractError):
                self.write(split=replace(self.split, clips=clips))

    def test_v1_default_behaviour_is_unchanged(self):
        # Le loader historique vérifie le manifeste avant ce contrat ; cette
        # fixture sans fichier réel préserve cette frontière de confiance V1.
        split = Split(clips=(Clip("train", 1, "leak", "a", "train"),
                             Clip("val", 0, "no_leak", "b", "val"),
                             Clip("test", 1, "leak", "c", "test")),
                      sha256=FROZEN_SPLIT_SHA256,
                      manifest_path=self.root / FROZEN_SPLIT_NAME, _paths={})
        kwargs = dict(run_id="legacy", model_name="legacy", checkpoint="frozen",
                      training_commit="a" * 40, split=split, threshold_rule="synthetic",
                      probabilities={"train": .1, "val": .2, "test": .3})
        output = self.root / "legacy"
        write_run(output, **kwargs)
        self.assertIn("val,0.2000000000\n", (output / "predictions.csv").read_text())
        write_run(output, **kwargs)  # Réécriture historique : inchangée hors opt-in.
        (output / "predictions.csv").write_text(
            "probability_leak,clip_id,comment\n.1,train,legacy\n.2,val,legacy\n.3,test,legacy\n")
        run = load_run(output, split)
        self.assertEqual(set(run.probabilities), {"train", "val", "test"})
        self.assertFalse(any(c.name == "octets du manifeste externe épinglés" for c in run.checks))
        modified = replace(split, sha256="d" * 64)
        write_run(output, **{**kwargs, "split": modified})
        with self.assertRaises(ContractError):
            load_run(output, modified)
        self.assertEqual(load_run(output, modified, require_frozen_sha=False).run_id, "legacy")


if __name__ == "__main__":
    unittest.main()
