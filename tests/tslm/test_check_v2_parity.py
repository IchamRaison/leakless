"""Tests CPU du gate, sans modèle ni métrique sur des données réelles."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import check_v2_parity as gate


def fixture_report(pid=1):
    scores = {name: 0.4 for name in gate.ADAPTERS}
    scores.update({f"batch_{size}_{order}": 0.4 for size in gate.BATCH_SIZES
                   for order in ("forward", "reversed")})
    provenance = {key: "same" for key in (
        "preprocessing_version", "historical_preprocessing_version", "manifest_sha256",
        "cache_sha256", "historical_cache_sha256", "checkpoint_checksums_sha256", "temporal_sha256",
        "timef_manifest_sha256", "source_sha256", "runtime", "scoring_spec", "state_sha256")}
    provenance.update(process={"hostname": "same-host", "pid": pid}, clip_ids=["val"],
                      seed=gate.SEED, score_atol=gate.SCORE_ATOL, threshold_diagnostic_only=None,
                      amplitude_evidence=False, amplitude_evidence_version=None, amplitude_tokens={})
    return {"schema": gate.SCHEMA, "all_checks_pass": False,
            "checks": {key: key != "fresh_process_verified" for key in gate.CHECKS},
            "provenance": provenance, "records": [{"clip_id": "val", "exact": True}],
            "scores": {"val": scores}}


class V2ParityGateChecks(unittest.TestCase):
    def test_canonical_transform_matches_timef_and_refuses_legacy_bundle(self):
        def normalise(x):
            x = np.asarray(x, dtype=np.float64)
            centered = x - x.mean()
            return centered / np.sqrt(np.mean(centered ** 2))
        spec = importlib.util.spec_from_file_location("v2_gate_preprocessing", ROOT / "src/pipe/tslm/preprocessing.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"leakless_acoustic.connector": SimpleNamespace(_normalise=normalise)}):
            spec.loader.exec_module(module)
        waveform = np.random.default_rng(19).integers(-14000, 14000, 8000).astype(np.float64)
        timef = normalise(waveform).astype(np.float32)
        canonical = module.preprocess_audio(waveform, 8000, version=module.CANONICAL_VERSION)
        routes = {"canonical": canonical, "timef": module.band_series(timef), "cache": canonical.copy()}
        self.assertTrue(gate.compare_routes(routes)["exact"])
        routes["legacy"] = module.preprocess_audio(waveform, 8000)
        self.assertFalse(gate.compare_routes(routes)["exact"])
        self.assertFalse(gate.exact_arrays(canonical, canonical.astype(np.float64)))
        gate.require_canonical({"preprocessing_version": module.CANONICAL_VERSION}, module.CANONICAL_VERSION)
        for metadata in ({}, {"preprocessing_version": module.VERSION}):
            with self.assertRaisesRegex(ValueError, "legacy"):
                gate.require_canonical(metadata, module.CANONICAL_VERSION)

    def test_real_batch_sizes_and_reverse_preserve_ids_without_metadata(self):
        ids = [f"clip{i}" for i in range(9)]
        series = {cid: np.full((4, 64), i, dtype=np.float32) for i, cid in enumerate(ids)}
        calls = []
        def score_batch(arrays):
            self.assertTrue(all(isinstance(a, np.ndarray) for a in arrays))
            calls.append([float(a[0, 0]) for a in arrays])
            return [float(a[0, 0]) / 10 for a in arrays]
        results = gate.score_batches(ids, series, score_batch)
        self.assertEqual(len(results), 6)
        self.assertEqual({len(chunk) for chunk in calls}, {1, 2, 4})
        self.assertIn([8, 7, 6, 5], calls)
        for result in results.values():
            self.assertEqual(result, {cid: i / 10 for i, cid in enumerate(ids)})
        with self.assertRaises(ValueError):
            gate.score_batches(ids, series, lambda arrays: [0.5])
        with self.assertRaises(ValueError):
            gate.score_batches([*ids, ids[0]], series, score_batch)
        with self.assertRaises(ValueError):
            gate.score_batches(ids, series, lambda arrays: [float("nan")] * len(arrays))

    def test_c_amplitude_cache_and_text_are_checked_before_model_and_preserved_in_batches(self):
        # Le vrai prepare_inputs et les vrais calculs C sur deux WAV synthétiques.
        # Doubles uniquement pour les lecteurs/installations TimeF et ML absents.
        def normalise(x):
            centered = np.asarray(x, dtype=np.float64) - np.mean(x)
            return centered / np.sqrt(np.mean(centered ** 2))
        connector = SimpleNamespace(_normalise=normalise,
            __file__=str(ROOT / "scripts/timenet/leakless_acoustic/connector.py"))
        spec = importlib.util.spec_from_file_location("gate_amplitude_preprocessing",
                                                     ROOT / "src/pipe/tslm/preprocessing.py")
        preprocessing = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"leakless_acoustic.connector": connector}):
            spec.loader.exec_module(preprocessing)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("checkpoint", "prepared", "historical", "timef", "audio"):
                (root / name).mkdir()
            metadata = {"preprocessing_version": preprocessing.CANONICAL_VERSION,
                        "amplitude_evidence": True, "single_clip_acoustic_encoding": True,
                        "amplitude_evidence_version": preprocessing.AMPLITUDE_EVIDENCE_VERSION}
            (root / "checkpoint/metadata.json").write_text(json.dumps(metadata))
            (root / "checkpoint/checksums.json").write_text("{}")
            (root / "checkpoint/temporal.pt").write_bytes(b"synthetic weights never loaded")
            (root / "timef/manifest.json").write_text("{}")
            clips = [SimpleNamespace(clip_id="train", fold="train"),
                     SimpleNamespace(clip_id="val", fold="val"),
                     SimpleNamespace(clip_id="unread-test", fold="test")]
            split = SimpleNamespace(clips=clips, sha256="f" * 64,
                fold=lambda fold: [c for c in clips if c.fold == fold],
                path_of=lambda cid: f"{cid}.wav")
            selected, records, md5 = ["val", "train"], {}, {"unread-test": "0" * 32}
            for index, cid in enumerate(("train", "val")):
                waveform = np.random.default_rng(index + 70).integers(-12000, 12000, 8000).astype("<i2")
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(8000)
                    writer.writeframes(waveform.tobytes())
                raw = buffer.getvalue()
                (root / f"audio/{cid}.wav").write_bytes(raw)
                md5[cid] = hashlib.md5(raw).hexdigest()
                normalized = normalise(waveform).astype(np.float32)
                records[cid] = SimpleNamespace(record_id=cid,
                    time_series=[SimpleNamespace(to_numpy=lambda x=normalized: x)])
                series = preprocessing.band_series(normalized)
                amplitude = preprocessing.amplitude_from_normalized(normalized)
                np.savez_compressed(root / f"prepared/{cid}.npz", ids=np.array([cid]),
                    series=series[None], preprocessing_version=preprocessing.CANONICAL_VERSION,
                    amplitude_features=amplitude[None],
                    amplitude_text=np.array([preprocessing.amplitude_text(amplitude)]),
                    amplitude_evidence_version=preprocessing.AMPLITUDE_EVIDENCE_VERSION)
                np.savez_compressed(root / f"historical/{cid}.npz", ids=np.array([cid]),
                    series=series[None], preprocessing_version=preprocessing.VERSION)

            class Reader:
                def __init__(self, version): pass
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def iter_records(self, record_ids, with_annotations):
                    self_case.assertEqual(record_ids, selected)
                    self_case.assertIs(with_annotations, False)
                    return iter(records[cid] for cid in record_ids)

            self_case = self
            def load_cache(prepared, fold, rows, version):
                with np.load(prepared / f"{fold}.npz", allow_pickle=False) as cached:
                    self.assertEqual(str(cached["preprocessing_version"]), version)
                    self.assertEqual(set(cached["ids"]), {r["clip_id"] for r in rows})
                    return dict(zip(cached["ids"].tolist(), cached["series"]))
            predict = SimpleNamespace(__file__=str(ROOT / "src/pipe/tslm/predict.py"),
                preprocess_for_model=lambda x, sr, meta: preprocessing.preprocess_audio(
                    x, sr, version=meta["preprocessing_version"]))
            model = SimpleNamespace(__file__=str(ROOT / "src/pipe/tslm/model.py"))
            tslm = SimpleNamespace(model=model, predict=predict, preprocessing=preprocessing)
            dependencies = {"leakless_acoustic": SimpleNamespace(connector=connector),
                "pipe": SimpleNamespace(tslm=tslm), "pipe.tslm": tslm,
                "pipe.tslm.campaign": SimpleNamespace(load_cache=load_cache),
                "timenet.reader.reader": SimpleNamespace(TimeFReader=Reader),
                "timenet.registry.version": SimpleNamespace(DatasetVersion=SimpleNamespace(open_local=lambda p: p))}
            args = SimpleNamespace(checkpoint=root / "checkpoint", prepared=root / "prepared",
                historical_prepared=root / "historical", timef_version=root / "timef",
                data_root=root / "audio", manifests=root, threshold=None)
            with patch.dict(sys.modules, dependencies), \
                    patch.object(gate.split_loader, "load_split", return_value=split), \
                    patch.object(gate, "selected_ids", return_value=selected), \
                    patch.object(gate, "load_expected_audio_md5", return_value=md5):
                inputs, checked, provenance = gate.prepare_inputs(args)
                self.assertTrue(all(row["exact"] and row["amplitude"]["text_exact"] for row in checked))
                self.assertEqual(provenance["source_sha256"]["features.py"], hashlib.sha256(
                    (ROOT / "scripts/eval/harness/features.py").read_bytes()).hexdigest())
                calls = []
                def score_batch(entries):
                    for series, amplitude in entries:
                        cid = next(cid for cid in selected if series is inputs[cid][2])
                        self.assertIs(amplitude, inputs[cid][3])
                        np.testing.assert_array_equal(amplitude,
                            preprocessing.amplitude_features(inputs[cid][1], 8000))
                        calls.append(cid)
                    return [.4] * len(entries)  # Double mécanique, jamais un modèle.
                gate.score_batches(selected, {cid: values[2:] for cid, values in inputs.items()}, score_batch)
                self.assertEqual(calls.count("train"), 6)
                self.assertEqual(calls.count("val"), 6)
                path = root / "prepared/train.npz"
                with np.load(path, allow_pickle=False) as cached:
                    original = {name: cached[name].copy() for name in cached.files}
                for field in ("amplitude_features", "amplitude_text"):
                    altered = {name: value.copy() for name, value in original.items()}
                    if field == "amplitude_features":
                        altered[field][0, 0] += 1
                    else:
                        altered[field][0] = "tampered"
                    np.savez_compressed(path, **altered)
                    output = root / f"failed-{field}"
                    with patch.object(gate, "run_scores") as scorer:
                        status = gate.main(["--checkpoint", str(args.checkpoint), "--prepared", str(args.prepared),
                            "--historical-prepared", str(args.historical_prepared), "--timef-version", str(args.timef_version),
                            "--data-root", str(args.data_root), "--manifests", str(root), "--output", str(output)])
                        self.assertEqual(status, 1)
                        scorer.assert_not_called()
                    receipt = json.loads((output / "report.json").read_text())
                    self.assertFalse(receipt["checks"]["canonical_inputs_exact"])
                    self.assertEqual(receipt["error"]["stage"], "prepare_inputs")
                    np.savez_compressed(path, **original)
            self.assertFalse((root / "audio/unread-test.wav").exists())

    def test_fixed_tolerance_pairwise_and_threshold_not_chosen(self):
        report = fixture_report()
        scores = report["scores"]
        scores["val"] = {name: 0.5 for name in scores["val"]}
        scores["val"]["wav_bytes"] = 0.5 - 0.4e-6
        scores["val"]["batch_4_reversed"] = 0.5 + 0.4e-6
        summary = gate.summarize_scores(scores, threshold=0.5)
        self.assertTrue(summary["batch_scores_within_atol"])
        self.assertEqual(summary["threshold_diagnostic"]["near_threshold_ids"], ["val"])
        self.assertFalse(summary["threshold_diagnostic"]["selected_here"])
        self.assertEqual(len(summary["threshold_diagnostic"]["decision_flips"]), 1)
        # Le sélecteur officiel peut rendre min(score)-1e-9 ou max(score)+1e-9.
        # Ne pas rabattre un seuil figé dans [0,1], ni le confondre avec un score.
        for threshold in (-1e-9, 1 + 1e-9):
            observed = gate.summarize_scores(scores, threshold)["threshold_diagnostic"]
            self.assertEqual(observed["threshold"], threshold)
            self.assertEqual(observed["decision_flips"], [])
        for threshold in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                gate.summarize_scores(scores, threshold)
        scores["val"]["batch_4_reversed"] = 0.5 + 0.8e-6
        self.assertFalse(gate.summarize_scores(scores)["batch_scores_within_atol"])
        del scores["val"]["waveform"]
        with self.assertRaises(ValueError):
            gate.summarize_scores(scores)

    def test_fresh_reference_identity_coverage_and_all_path_spread(self):
        first, second = fixture_report(1), fixture_report(2)
        self.assertTrue(gate.verify_reference(second, first)["all_scores_within_atol"])
        with self.assertRaisesRegex(ValueError, "processus"):
            gate.verify_reference(first, first)
        for key in ("source_sha256", "runtime", "state_sha256", "cache_sha256"):
            changed = copy.deepcopy(second)
            changed["provenance"][key] = "changed"
            with self.assertRaisesRegex(ValueError, key):
                gate.verify_reference(changed, first)
        changed = copy.deepcopy(first)
        changed["checks"]["batch_scores_within_atol"] = False
        with self.assertRaises(ValueError):
            gate.verify_reference(second, changed)
        changed = copy.deepcopy(second)
        changed["records"].append(changed["records"][0])
        with self.assertRaises(ValueError):
            gate.verify_reference(changed, first)
        changed = copy.deepcopy(second)
        changed["records"][0]["series"] = "changed"
        with self.assertRaises(ValueError):
            gate.verify_reference(changed, first)
        second["scores"]["val"]["series"] = 0.400002
        self.assertFalse(gate.verify_reference(second, first)["all_scores_within_atol"])

    def test_main_first_receipt_is_not_gate_and_second_process_can_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = ["--checkpoint", "unused", "--prepared", "unused", "--historical-prepared", "unused",
                      "--timef-version", "unused", "--data-root", "unused"]
            first = fixture_report(1)
            with patch.object(gate, "prepare_inputs", return_value=({}, first["records"], first["provenance"])), \
                    patch.object(gate, "run_scores", return_value=(first["scores"], True)):
                self.assertEqual(gate.main([*common, "--output", str(root / "first")]), 0)
            receipt = json.loads((root / "first/report.json").read_text())
            self.assertFalse(receipt["all_checks_pass"])
            self.assertEqual(receipt["status"], "awaiting_fresh_process")
            second = fixture_report(2)
            with patch.object(gate, "prepare_inputs", return_value=({}, second["records"], second["provenance"])), \
                    patch.object(gate, "run_scores", return_value=(second["scores"], True)):
                self.assertEqual(gate.main([*common, "--output", str(root / "second"),
                                            "--reference", str(root / "first/report.json")]), 0)
            receipt = json.loads((root / "second/report.json").read_text())
            self.assertTrue(receipt["all_checks_pass"])
            with self.assertRaises(FileExistsError):
                gate.main([*common, "--output", str(root / "first")])

    def test_input_divergence_stops_before_loading_model_and_keeps_failure(self):
        fixture = fixture_report()
        fixture["records"][0]["exact"] = False
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(gate, "prepare_inputs", return_value=({}, fixture["records"], fixture["provenance"])), \
                patch.object(gate, "run_scores") as scorer:
            output = Path(tmp) / "failed"
            code = gate.main(["--checkpoint", "unused", "--prepared", "unused", "--historical-prepared", "unused",
                              "--timef-version", "unused", "--data-root", "unused", "--output", str(output)])
            self.assertEqual(code, 1)
            scorer.assert_not_called()
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["all_checks_pass"])
            self.assertEqual(report["error"]["stage"], "prepare_inputs")


if __name__ == "__main__":
    unittest.main()
