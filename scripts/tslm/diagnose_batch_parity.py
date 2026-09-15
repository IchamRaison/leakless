"""Localiser la dépendance au lot : quatre cas de développement, aucun entraînement."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np

from diagnose_parity import (ROOT, SCORE_ATOL, TRAIN_CASE, array_difference, array_info,
                             score_spread, selected_ids, sha256_file, split_loader, write_json)

CASES = ("cf591447e4f2d", "c00c343da6afa", "ce0594e503562", TRAIN_CASE)
SEED = 20260912


def contexts(ids, target):
    """Reconstituer exactement les voisins du gate, y compris les lots tronqués."""
    if len(ids) != len(set(ids)) or target not in ids:
        raise ValueError("IDs du gate répétés ou cible absente")
    result = [{"name": "batch_1_forward", "requested_batch_size": 1,
               "clip_ids": [target], "target_position": 0}]
    for order_name, ordered in (("forward", ids), ("reversed", list(reversed(ids)))):
        for size in (2, 4):
            position = ordered.index(target)
            start = position // size * size
            chunk = ordered[start:start + size]
            result.append({"name": f"batch_{size}_{order_name}", "requested_batch_size": size,
                           "clip_ids": chunk, "target_position": chunk.index(target)})
    return result


@contextmanager
def per_clip_padding(model, pad_sequence=None):
    """Intervention temporaire : même fonction originale, appelée par clip.

    Aucun remplacement des encodeurs/poids/prompts ; seul le regroupement change.
    `pad_sequence` injectable uniquement pour fixtures CPU sans dépendance Torch.
    """
    if pad_sequence is None:
        from torch.nn.utils.rnn import pad_sequence
    original = model.pad_and_apply_batch
    sentinel = object()
    previous = vars(model).get("pad_and_apply_batch", sentinel)
    def independent(batch):
        if not batch:
            raise ValueError("Lot vide")
        embeddings, masks = [], []
        for sample in batch:
            inputs, attention = original([sample])
            if len(inputs) != 1 or len(attention) != 1 or inputs.shape[:2] != attention.shape:
                raise ValueError("Sortie singleton incompatible")
            valid = attention[0] != 0
            embeddings.append(inputs[0][valid])
            masks.append(attention[0][valid])
        return (pad_sequence(embeddings, batch_first=True, padding_value=0),
                pad_sequence(masks, batch_first=True, padding_value=0))
    model.pad_and_apply_batch = independent
    try:
        yield
    finally:
        if previous is sentinel:
            delattr(model, "pad_and_apply_batch")
        else:
            model.pad_and_apply_batch = previous


@contextmanager
def capture_numeric(model):
    """Hooks transparents avant cast BF16 et sur les embeddings du prompt réel."""
    calls, handles, dtypes = {"encoder": [], "projector": [], "prompt": []}, [], {}
    def keep(tensor):
        # BF16 n'est pas représentable directement par NumPy ; la conversion F32 est exacte.
        return tensor.detach().float().cpu().numpy().copy()
    def hook(name):
        def observe(module, args, output):
            dtypes[name + "_input"] = str(args[0].dtype)
            dtypes[name + "_output"] = str(output.dtype)
            calls[name].append((keep(args[0]), keep(output)))
        return observe
    original = model.pad_and_apply_batch
    sentinel = object()
    previous = vars(model).get("pad_and_apply_batch", sentinel)
    def padding(batch):
        inputs, mask = original(batch)
        dtypes["prompt_embeddings"], dtypes["prompt_mask"] = str(inputs.dtype), str(mask.dtype)
        calls["prompt"].append((keep(inputs), keep(mask)))
        return inputs, mask
    try:
        handles.append(model.encoder.register_forward_hook(hook("encoder")))
        handles.append(model.projector.register_forward_hook(hook("projector")))
        model.pad_and_apply_batch = padding
        yield calls, dtypes
    finally:
        for handle in handles:
            handle.remove()
        if previous is sentinel:
            delattr(model, "pad_and_apply_batch")
        else:
            model.pad_and_apply_batch = previous


def sample_component(calls, side, position, batch_size, expected_shape):
    """Même tranche d'un packing contigu ; refuser les formes non expliquées."""
    combined = np.concatenate([pair[side] for pair in calls], axis=0)
    expected = (expected_shape[0] * batch_size, *expected_shape[1:])
    if combined.shape != expected:
        raise ValueError(f"Packing des séries non reconnu : {combined.shape} != {expected}")
    width = expected_shape[0]
    return combined[position*width:(position+1)*width]


def trace(model, batch, position, series, reference=None):
    import torch
    with capture_numeric(model) as (calls, dtypes):
        logprobs = model.score_class_logprobs(batch)
    if len(calls["prompt"]) != 1:
        raise ValueError("Un seul entrelacement de prompt est attendu par requête")
    arrays = {"input_series": series.copy()}
    for component in ("encoder", "projector"):
        for side, label in ((0, "input"), (1, "output")):
            key = component + "_" + label
            if reference is None:
                if len(batch) != 1 or len(calls[component]) != 1:
                    raise ValueError("La référence doit être un singleton et un appel par composant")
                arrays[key] = calls[component][0][side]
            else:
                arrays[key] = sample_component(calls[component], side, position, len(batch), reference[key].shape)
    inputs, mask = calls["prompt"][0]
    valid = mask[position].astype(bool)
    arrays["prompt_embeddings"] = inputs[position][valid]
    arrays["prompt_mask"] = mask[position][valid]
    arrays["class_logprob_sums"] = logprobs[position].cpu().numpy().copy()
    probability = float(torch.softmax(logprobs[position], dim=-1)[0].item())
    score_spread([probability])
    arrays["probability_leak"] = np.asarray([probability], dtype=np.float64)
    if reference is not None and not np.array_equal(arrays["encoder_input"], reference["encoder_input"]):
        raise ValueError("Les tranches d'entrée encodeur ne sont pas identiques : alignement non démontré")
    return arrays, {"source_dtypes": dtypes,
                    "module_calls": {k: len(v) for k, v in calls.items()},
                    "stages": {name: array_info(a) for name, a in arrays.items()},
                    "probability_leak": probability}


def stage_differences(left, right):
    if list(left) != list(right):
        raise ValueError("Étages comparés différents")
    differences = {name: array_difference(left[name], right[name]) for name in left}
    first = next((k for k, value in differences.items() if not value["exact_values"]), None)
    return {"first_exact_divergence": first, "stages": differences}


def run(args):
    if args.output.exists():
        raise FileExistsError("Sortie existante : aucun écrasement")
    if len(args.code_revision) != 40 or any(c not in "0123456789abcdef" for c in args.code_revision):
        raise ValueError("SHA Git complet requis")
    import torch
    from leakless_acoustic import connector
    from pipe.tslm import model as model_module, predict, preprocessing
    from pipe.tslm.campaign import load_cache
    from pipe.tslm.train import tensor_state_hash
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate as collate
    gate = json.loads(args.gate_report.read_text())
    p = gate["provenance"]
    split = split_loader.load_split(args.manifests)
    ids = selected_ids(split, True)
    if (gate.get("schema") != "pipe-parity-v2" or p["clip_ids"] != ids
            or p["manifest_sha256"] != split.sha256
            or p["preprocessing_version"] != preprocessing.CANONICAL_VERSION
            or p["checkpoint_checksums_sha256"] != sha256_file(args.checkpoint / "checksums.json")
            or p["temporal_sha256"] != sha256_file(args.checkpoint / "temporal.pt")):
        raise ValueError("Gate, modèle ou périmètre de diagnostic incompatibles")
    sources = {"model.py": Path(model_module.__file__), "predict.py": Path(predict.__file__),
               "preprocessing.py": Path(preprocessing.__file__), "connector.py": Path(connector.__file__)}
    if any(p["source_sha256"][name] != sha256_file(path) for name, path in sources.items()):
        raise ValueError("Le code numérique diffère du gate à diagnostiquer")
    caches = {}
    for fold in ("train", "val"):
        if p["cache_sha256"][fold] != sha256_file(args.prepared / f"{fold}.npz"):
            raise ValueError("Cache différent du gate")
        rows = [{"clip_id": c.clip_id, "fold": fold} for c in split.fold(fold)]
        caches.update(load_cache(args.prepared, fold, rows, preprocessing.CANONICAL_VERSION))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    predictor = predict.Predictor(args.checkpoint, device=args.device)
    model = predictor.model
    if model.training or model.scoring_spec() != p["scoring_spec"]:
        raise ValueError("Mode ou scoring différent du gate")
    components = {"encoder": model.encoder, "projector": model.projector, "llm": model.llm}
    before = {name: tensor_state_hash(module) for name, module in components.items()}
    if before != p["state_sha256"]:
        raise ValueError("États numériques différents du gate")
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": args.device, "cudnn": torch.backends.cudnn.version(),
        "cuda_device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    for package in ("transformers", "scikit-learn", "timenet", "opentslm"):
        try:
            runtime[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime[package] = "not-installed-as-distribution"
    if runtime != p["runtime"]:
        raise ValueError("Runtime différent du gate à reproduire")
    provenance = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "code_revision": args.code_revision,
        "gate_report_sha256": sha256_file(args.gate_report), "gate_provenance": p, "runtime": runtime,
        "pid": os.getpid(), "seed": SEED, "cases": list(CASES), "score_atol": SCORE_ATOL,
        "source_sha256": {**{k: sha256_file(v) for k, v in sources.items()},
            "diagnose_batch_parity.py": sha256_file(Path(__file__)),
            "diagnose_parity.py": sha256_file(Path(__file__).with_name("diagnose_parity.py"))},
        "state_sha256_before": before, "optimizer_steps": 0,
        "quality_metrics_calculated": False, "test_audio_or_cache_opened": False}
    records = []
    for cid in CASES:
        reference = None
        for context in contexts(ids, cid):
            chunk, position = context["clip_ids"], context["target_position"]
            def make_batch():
                return collate([preprocessing.model_input(caches[other]) for other in chunk], normalize=False)
            original, original_info = trace(model, make_batch(), position, caches[cid], reference)
            if reference is None:
                reference = original
            with per_clip_padding(model):
                independent, independent_info = trace(model, make_batch(), position, caches[cid], reference)
            restored_scores = model.score_probability_leak(make_batch())
            base = args.output / f"{cid}-{context['name']}"
            np.savez_compressed(str(base) + "-original.npz", **original)
            np.savez_compressed(str(base) + "-per-clip.npz", **independent)
            observed = float(gate["scores"][cid][context["name"]])
            record = {"clip_id": cid, **context, "effective_batch_size": len(chunk),
                "original": original_info, "temporary_per_clip": independent_info,
                "singleton_vs_original": stage_differences(reference, original),
                "singleton_vs_per_clip": stage_differences(reference, independent),
                "original_vs_per_clip": stage_differences(original, independent),
                "gate_reproduction": score_spread([observed, original_info["probability_leak"]]),
                "hooks_restoration_check": score_spread([float(restored_scores[position]), original_info["probability_leak"]]),
                "per_clip_matches_singleton": score_spread([float(reference["probability_leak"][0]),
                                                             independent_info["probability_leak"]]),
                "arrays_sha256": {suffix: sha256_file(Path(str(base) + suffix))
                                  for suffix in ("-original.npz", "-per-clip.npz")}}
            records.append(record)
            print(json.dumps({"clip_id": cid, "context": context["name"],
                "first_divergence": record["singleton_vs_original"]["first_exact_divergence"],
                "gate_reproduction": record["gate_reproduction"],
                "per_clip_matches_singleton": record["per_clip_matches_singleton"]}), flush=True)
    after = {name: tensor_state_hash(module) for name, module in components.items()}
    provenance.update(state_sha256_after=after, weights_unchanged=before == after)
    report = {"schema": "pipe-batch-parity-diagnostic-v1", "provenance": provenance, "records": records,
        "all_original_scores_reproduce_gate": all(r["gate_reproduction"]["within_atol"] for r in records),
        "all_hooks_and_patches_restored": all(r["hooks_restoration_check"]["within_atol"] for r in records),
        "all_per_clip_scores_match_singletons": all(r["per_clip_matches_singleton"]["within_atol"] for r in records),
        "weights_unchanged": before == after}
    write_json(args.output / "report.json", report)
    if before != after:
        raise ValueError("États modifiés durant le diagnostic")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "prepared", "manifests", "gate-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
