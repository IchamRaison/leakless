"""Localiser la divergence cache/TimeF/WAV, sans entraînement ni métrique qualité."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from export_run import ROOT, sha256_file
from audit_text import load_expected_audio_md5, verify_audio_bytes
from harness import split_loader

VALIDATION_CASES = ("c00c343da6afa", "ce0594e503562", "c50c4f2d91963")
TRAIN_CASE = "c0128b879694e"
SCORE_ATOL = 1e-6


def array_info(value):
    a = np.asarray(value)
    if a.dtype.kind not in "biuf" or not np.isfinite(a).all():
        raise ValueError("Tableau numérique fini requis pour le diagnostic")
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
            "min": float(a.min()) if a.size else None, "max": float(a.max()) if a.size else None}


def array_difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    info = {"left": array_info(a), "right": array_info(b), "same_shape": a.shape == b.shape}
    if a.shape != b.shape:
        return {**info, "exact_values": False, "max_abs": None, "n_different": None}
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {**info, "exact_values": bool(np.array_equal(a, b)),
            "n_different": int(np.count_nonzero(delta)),
            "max_abs": float(delta.max()) if delta.size else 0.0,
            "mean_abs": float(delta.mean()) if delta.size else 0.0,
            "max_abs_index": [int(i) for i in np.unravel_index(int(delta.argmax()), delta.shape)] if delta.size else []}


def selected_ids(split, all_validation):
    by_id = split.by_id()
    if any(by_id[cid].fold != "val" for cid in VALIDATION_CASES) or by_id[TRAIN_CASE].fold != "train":
        raise ValueError("Les cas fixes ne correspondent pas aux folds de développement")
    validation = sorted(c.clip_id for c in split.fold("val")) if all_validation else list(VALIDATION_CASES)
    if all_validation and len(validation) != 208:
        raise ValueError("208 validations requises")
    return [*validation, TRAIN_CASE]


def make_routes(raw, cached, timef_waveform=None):
    from leakless_acoustic.connector import _normalise
    from pipe.tslm.preprocessing import band_series, decode_wav, preprocess_audio
    waveform = decode_wav(raw)
    normalized = _normalise(waveform)
    rounded = normalized.astype(np.float32)
    routes = {"cache": cached.copy(), "direct": preprocess_audio(waveform, 8000),
              "roundtrip_f32": band_series(rounded)}
    waveform_comparisons = {"normalization_f64_vs_roundtrip_f32": array_difference(normalized, rounded)}
    if timef_waveform is not None:
        routes["timef"] = band_series(timef_waveform)
        waveform_comparisons.update({"normalized_direct_vs_timef": array_difference(normalized, timef_waveform),
                                     "rounded_f32_vs_timef": array_difference(rounded, timef_waveform)})
    comparisons = {f"cache_vs_{name}": array_difference(cached, value)
                   for name, value in routes.items() if name != "cache"}
    comparisons["direct_vs_roundtrip_f32"] = array_difference(routes["direct"], routes["roundtrip_f32"])
    return routes, waveform_comparisons, comparisons


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def prepare_inputs(args):
    from pipe.tslm.campaign import load_cache
    from pipe.tslm import preprocessing
    from leakless_acoustic import connector
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    split = split_loader.load_split(args.manifests)
    ids = selected_ids(split, args.all_validation)
    expected_md5 = load_expected_audio_md5(args.manifests)
    if set(expected_md5) != {c.clip_id for c in split.clips}:
        raise ValueError("Empreintes audio incomplètes")
    caches = {}
    for fold in ("train", "val"):
        rows = [{"clip_id": c.clip_id, "fold": c.fold} for c in split.fold(fold)]
        caches.update(load_cache(args.prepared, fold, rows, metadata["preprocessing_version"]))
    timef_values = {}
    if args.timef_version:
        from timenet.reader.reader import TimeFReader
        from timenet.registry.version import DatasetVersion
        with TimeFReader(DatasetVersion.open_local(args.timef_version)) as reader:
            # Ne pas appeler verify()/tasks ni parcourir les signaux test.
            for record in reader.iter_records(record_ids=ids, with_annotations=False):
                if record.record_id not in ids or record.record_id in timef_values:
                    raise ValueError("Record TimeF inattendu ou répété")
                timef_values[record.record_id] = record.time_series[0].to_numpy()
        if set(timef_values) != set(ids):
            raise ValueError("Couverture TimeF incomplète")
    inputs, records = {}, []
    root = args.data_root.resolve()
    for cid in ids:
        path = (root / split.path_of(cid)).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Chemin WAV hors data-root")
        raw = path.read_bytes()
        verify_audio_bytes(raw, cid, expected_md5)
        routes, waveforms, comparisons = make_routes(raw, caches[cid], timef_values.get(cid))
        inputs[cid] = (raw, routes)
        records.append({"clip_id": cid, "fold": split.by_id()[cid].fold,
                        "input_sha256": hashlib.sha256(raw).hexdigest(),
                        "waveform_comparisons": waveforms, "series_comparisons": comparisons,
                        "series": {name: array_info(value) for name, value in routes.items()}})
    provenance = {"purpose": "parity_diagnosis_no_training_no_quality_metrics", "mode": args.mode,
        "created_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "hostname": platform.node(),
        "python": platform.python_version(), "numpy": np.__version__, "clip_ids": ids,
        "test_audio_or_cache_opened": False, "frozen_manifest_contains_test_metadata": True,
        "checkpoint_checksums_sha256": sha256_file(args.checkpoint / "checksums.json"),
        "temporal_sha256": sha256_file(args.checkpoint / "temporal.pt"),
        "manifest_sha256": split.sha256,
        "cache_sha256": {fold: sha256_file(args.prepared / f"{fold}.npz") for fold in ("train", "val")},
        "timef_manifest_sha256": sha256_file(args.timef_version / "manifest.json") if args.timef_version else None,
        "historical_preprocessing_version": metadata["preprocessing_version"],
        "current_preprocessing_version": preprocessing.VERSION,
        "score_atol": SCORE_ATOL, "repeats_fixed_cases": 3,
        "source_sha256": {"diagnose_parity.py": sha256_file(Path(__file__)),
                          "preprocessing.py": sha256_file(preprocessing.__file__),
                          "connector.py": sha256_file(connector.__file__)}}
    return inputs, records, provenance


def score_spread(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Scores de diagnostic non finis ou hors bornes")
    delta = float(values.max() - values.min())
    return {"values": values.tolist(), "max_abs": delta, "within_atol": delta <= SCORE_ATOL,
            "exact": bool(np.all(values == values.flat[0]))}


@contextmanager
def capture_stages(model):
    """Hooks sur modules entiers ; aucun paramètre, argument ou retour modifié."""
    import torch
    arrays, dtypes, handles = {}, {}, []
    counts = {"encoder": 0, "projector": 0}
    spec = model.scoring_spec()

    def keep(name, tensor):
        dtypes[name] = str(tensor.dtype)
        arrays[name] = tensor.detach().float().cpu().numpy().copy()

    def component_hook(name):
        def hook(module, args, output):
            index = counts[name]
            keep(f"{name}_{index}_input", args[0])
            keep(f"{name}_{index}_output", output)
            counts[name] += 1
        return hook

    def llm_hook(module, args, kwargs, output):
        if "class_logits" in arrays:
            raise ValueError("Trace limitée à un clip et deux continuations")
        inputs = kwargs["inputs_embeds"]
        keep("llm_inputs_embeds", inputs)
        keep("llm_attention_mask", kwargs["attention_mask"])
        lengths = spec["class_token_counts"]
        width = max(lengths)
        prefix_length = inputs.shape[1] - width
        logits = output.logits[:, prefix_length - 1:prefix_length + width - 1].float()
        if logits.shape[:2] != (2, width):
            raise ValueError("Positions de classe inattendues")
        keep("class_logits", logits)
        candidates = [torch.tensor(ids, device=logits.device, dtype=torch.long)
                      for ids in spec["class_token_ids"]]
        ids = torch.nn.utils.rnn.pad_sequence(candidates, batch_first=True,
                                             padding_value=model.tokenizer.pad_token_id)
        mask = torch.arange(width, device=logits.device)[None, :] < torch.tensor(lengths, device=logits.device)[:, None]
        keep("class_target_logits", logits.gather(-1, ids[..., None]).squeeze(-1))
        keep("class_log_normalizer", torch.logsumexp(logits, dim=-1))
        token_lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[..., None]).squeeze(-1)
        keep("class_token_mask", mask)
        keep("class_token_logprobs", token_lp.masked_fill(~mask, 0))

    try:
        handles.append(model.encoder.register_forward_hook(component_hook("encoder")))
        handles.append(model.projector.register_forward_hook(component_hook("projector")))
        handles.append(model.llm.register_forward_hook(llm_hook, with_kwargs=True))
        yield arrays, dtypes
    finally:
        for handle in handles:
            handle.remove()


def trace_one(model, series):
    import torch
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate as collate
    from pipe.tslm.preprocessing import model_input
    example = model_input(series)
    if set(example) != {"pre_prompt", "post_prompt", "time_series", "time_series_text"}:
        raise ValueError("Métadonnées ou réponse dans l'entrée de diagnostic")
    prompt_hash = hashlib.sha256(json.dumps({key: value for key, value in example.items()
                                            if key != "time_series"}, sort_keys=True).encode()).hexdigest()
    batch = collate([example], normalize=False)
    with capture_stages(model) as (arrays, dtypes):
        logprobs = model.score_class_logprobs(batch)
        probability = float(torch.softmax(logprobs, dim=-1)[0, 0].item())
    arrays = {"input_series": series.copy(), **arrays, "class_logprob_sums": logprobs.cpu().numpy().copy()}
    arrays["probability_leak"] = np.asarray([probability], dtype=np.float64)
    if "class_token_logprobs" not in arrays:
        raise ValueError("Hooks sans trace de la classe")
    np.testing.assert_allclose(arrays["class_token_logprobs"].astype(np.float64).sum(-1),
                               arrays["class_logprob_sums"][0], rtol=0, atol=1e-12)
    return arrays, {"source_dtypes": dtypes, "prompt_sha256": prompt_hash,
                    "class_logprob_sums": arrays["class_logprob_sums"][0].tolist(),
                    "class_token_logprobs": arrays["class_token_logprobs"].tolist(),
                    "probability_leak": probability}


def compare_stages(left, right):
    if list(left) != list(right):
        raise ValueError("Étages de trace incompatibles")
    comparisons = {name: array_difference(left[name], right[name]) for name in left}
    first = next((name for name, delta in comparisons.items() if not delta["exact_values"]), None)
    return {"first_exact_divergence": first, "stages": comparisons}


def compare_reference(report, reference):
    current, previous = report["provenance"], reference["provenance"]
    for key in ("checkpoint_checksums_sha256", "temporal_sha256", "clip_ids", "scoring_spec", "cache_sha256"):
        if current[key] != previous[key]:
            raise ValueError(f"Référence d'un autre protocole : {key}")
    if current["hostname"] != previous["hostname"] or current["pid"] == previous["pid"]:
        raise ValueError("Comparer deux processus distincts sur le même hôte de référence")
    by_id = {r["clip_id"]: r for r in reference["records"]}
    comparisons = {}
    for row in report["records"]:
        other = by_id[row["clip_id"]]
        if row["input_sha256"] != other["input_sha256"] or row["series"] != other["series"]:
            raise ValueError("Référence sur d'autres entrées numériques")
        comparisons[row["clip_id"]] = {name: score_spread([values[0], other["scores"][name][0]])
                                       for name, values in row["scores"].items()}
    return {"fresh_process": True, "all_scores_within_atol": all(
                item["within_atol"] for clip in comparisons.values() for item in clip.values()),
            "scores": comparisons}


def run_traces(args, inputs, records, provenance):
    import torch
    from pipe.tslm import model as model_module, predict as predict_module
    from pipe.tslm.predict import Predictor
    from pipe.tslm.preprocessing import decode_wav, model_input
    from pipe.tslm.train import tensor_state_hash
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate as collate
    predictor = Predictor(args.checkpoint, device=args.device)
    if args.base:
        # Expérience séparée ; ne pas maintenir deux décodeurs de 9 Go en mémoire.
        del predictor.model
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
        predictor.model = model_module.AcousticQwenSP(args.base, device=args.device)
        temporal = torch.load(args.checkpoint / "temporal.pt", map_location=args.device, weights_only=True)
        predictor.model.encoder.load_state_dict(temporal["encoder_state"], strict=True)
        predictor.model.projector.load_state_dict(temporal["projector_state"], strict=True)
        predictor.model.eval()
    model = predictor.model
    frozen_hash = tensor_state_hash(model.llm)
    expected_frozen = json.loads((args.checkpoint / "training-report.json").read_text())["frozen_llm_sha256_after"]
    if frozen_hash != expected_frozen or model.training:
        raise ValueError("Décodeur ou état eval différent du checkpoint de référence")
    provenance.update({"model_source": "original_base_plus_temporal" if args.base else "bundle",
        "original_base": str(args.base) if args.base else None, "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": args.device, "scoring_spec": model.scoring_spec(),
        "frozen_llm_sha256": frozen_hash, "inference_batch_size": 1,
        "model_training": model.training, "llm_use_cache_config": model.llm.config.use_cache,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32})
    provenance["source_sha256"].update({"model.py": sha256_file(model_module.__file__),
                                       "predict.py": sha256_file(predict_module.__file__)})
    if str(args.device).startswith("cuda"):
        provenance["gpu"] = torch.cuda.get_device_name(args.device)
    for row in records:
        cid = row["clip_id"]
        raw, routes = inputs[cid]
        detailed = cid in (*VALIDATION_CASES, TRAIN_CASE)
        repeats = 3 if detailed else 1
        scores = {name: [] for name in routes}
        started = time.perf_counter()
        # Alterner les chemins : un état qui dérive ne se confond pas avec le chemin.
        for repeat in range(repeats):
            for name in (list(routes) if repeat % 2 == 0 else list(reversed(routes))):
                scores[name].append(float(predictor.score_series(routes[name])))
        row["scores"] = scores
        row["repeatability"] = {name: score_spread(values) for name, values in scores.items()}
        row["path_parity"] = {f"cache_vs_{name}": score_spread([scores["cache"][0], values[0]])
                              for name, values in scores.items() if name != "cache"}
        row["scoring_elapsed_seconds"] = time.perf_counter() - started
        if detailed:
            row["same_array_adapters"] = {
                "predictor_series": scores["direct"][0],
                "model_batch": float(model.score_probability_leak(collate([model_input(routes["direct"])], normalize=False))[0]),
                "predictor_wav": float(predictor.score(raw)),
                "predictor_waveform": float(predictor.score_waveform(decode_wav(raw)))}
            row["adapter_parity"] = score_spread(list(row["same_array_adapters"].values()))
            traces, trace_reports = {}, {}
            for name, series in routes.items():
                traces[name], trace_reports[name] = trace_one(model, series)
                trace_reports[name]["hooks_vs_no_hooks"] = score_spread(
                    [trace_reports[name]["probability_leak"], scores[name][0]])
                # ponytail: garder les petits étages ; logits vocabulaire complets seulement comparés en RAM.
                trace_path = args.output / f"{cid}-{name}.npz"
                np.savez_compressed(trace_path, **{key: value for key, value in traces[name].items() if key != "class_logits"})
                trace_reports[name]["arrays_sha256"] = sha256_file(trace_path)
            row["traces"] = trace_reports
            row["stage_comparisons"] = {f"cache_vs_{name}": compare_stages(traces["cache"], arrays)
                                        for name, arrays in traces.items() if name != "cache"}
        print(json.dumps({"diagnostic_clip": cid, "path_parity": row["path_parity"]}), flush=True)
    if tensor_state_hash(model.llm) != frozen_hash:
        raise ValueError("Le décodeur a changé durant le diagnostic")
    provenance["frozen_llm_unchanged"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "trace"), default="preflight")
    for name in ("checkpoint", "prepared", "data-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, default=ROOT / "manifests")
    parser.add_argument("--timef-version", type=Path)
    parser.add_argument("--all-validation", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference", type=Path, help="report.json d'un processus précédent, jamais réécrit")
    parser.add_argument("--base", type=Path, help="Base Qwen d'origine + mêmes poids temporels, expérience séparée")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Dossier de diagnostic existant, aucun écrasement")
    if args.mode == "preflight" and (args.reference or args.base):
        raise ValueError("reference/base réservés au diagnostic du modèle en mode trace")
    inputs, records, provenance = prepare_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "provenance.json", provenance)
    if args.mode == "trace":
        run_traces(args, inputs, records, provenance)
    report = {"provenance": provenance, "records": records,
              "note": "Les écarts de tableaux ne prouvent pas une cause de divergence du score ; mode trace requis."}
    if args.reference:
        report["reference_comparison"] = compare_reference(report, json.loads(args.reference.read_text()))
    write_json(args.output / "report.json", report)
    print(json.dumps({"output": str(args.output), "mode": args.mode, "n_clips": len(records),
                      "report_sha256": sha256_file(args.output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
