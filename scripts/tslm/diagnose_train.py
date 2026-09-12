"""Diagnostic V2 limité aux 598 train : sonde fixe, C1 et supervision optionnelle.

Aucun réglage sur la validation officielle, aucun accès à ses WAV/caches ni au
test. Le receipt de parité PASS et le cache canonique sont obligatoires.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/eval"))
from audit_text import load_expected_audio_md5, verify_audio_bytes
from export_run import sha256_file
from harness import features, metrics, split_loader
from pipe.tslm.campaign import epoch_batches, load_cache, write_json

SEED = 20260912
N_FOLDS = 3
THRESHOLD = 0.5  # Diagnostic fixe, jamais un seuil choisi ou livré au produit.
PARITY_CHECKS = ("canonical_inputs_exact", "bundle_unchanged",
                 "adapter_scores_within_atol", "batch_scores_within_atol",
                 "fresh_process_verified")
SOURCE_PATHS = {name: ROOT / "src/pipe/tslm" / name
                for name in ("model.py", "predict.py", "preprocessing.py")}
SOURCE_PATHS["connector.py"] = ROOT / "scripts/timenet/leakless_acoustic/connector.py"


def verify_parity(report_path, prepared, version, *, source_paths=SOURCE_PATHS):
    """Refuser une autre chaîne numérique ; ne lire que le cache TRAIN."""
    report = json.loads(Path(report_path).read_text())
    p = report.get("provenance", {})
    if (report.get("schema") != "pipe-parity-v2" or report.get("all_checks_pass") is not True
            or any(report.get("checks", {}).get(k) is not True for k in PARITY_CHECKS)
            or p.get("score_atol") != 1e-6 or p.get("preprocessing_version") != version
            or p.get("manifest_sha256") != split_loader.FROZEN_SPLIT_SHA256
            or p.get("cache_sha256", {}).get("train") != sha256_file(Path(prepared) / "train.npz")
            or any(p.get("source_sha256", {}).get(k) != sha256_file(path)
                   for k, path in source_paths.items())):
        raise ValueError("Parité PASS absente, périmée ou différente du cache/code canonique")
    preparation = json.loads((Path(prepared) / "preparation.json").read_text())
    if (preparation.get("preprocessing_version") != version or preparation.get("records") != 806
            or preparation.get("fold_counts") != {"train": 598, "val": 208}
            or preparation.get("canonical_vs_timef_exact") is not True
            or preparation.get("test_audio_or_cache_opened") is not False
            or preparation.get("manifest_sha256") != p["manifest_sha256"]
            or preparation.get("cache_sha256") != p["cache_sha256"]
            or preparation.get("timef_manifest_sha256") != p["timef_manifest_sha256"]
            or any(preparation.get("source_sha256", {}).get(name) != sha256_file(ROOT / "src/pipe/tslm" / name)
                   for name in ("prepare.py", "preprocessing.py"))):
        raise ValueError("Reçu de préparation canonique complet absent ou périmé")
    return p


def train_rows(split):
    rows = [{"clip_id": c.clip_id, "group_id": c.group_id, "fold": "train",
             "label": "leak" if c.label else "no_leak"} for c in split.fold("train")]
    if (len(rows) != 598 or len({r["clip_id"] for r in rows}) != 598
            or len({r["group_id"] for r in rows}) != 102):
        raise ValueError("Les 598 clips / 102 groupes train gelés sont requis")
    groups = {r["group_id"] for r in rows}
    if any(c.fold != "train" and c.group_id in groups for c in split.clips):
        raise ValueError("Groupe train partagé avec un fold officiel réservé")
    return sorted(rows, key=lambda r: r["clip_id"])


def grouped_folds(rows):
    """Stratification des GROUPES, pas optimisation des proportions de clips."""
    labels, seen = {}, set()
    for r in rows:
        cid, group, label = r["clip_id"], r["group_id"], r["label"]
        if (r["fold"] != "train" or cid in seen or label not in ("leak", "no_leak")
                or labels.get(group, label) != label):
            raise ValueError("IDs/groupes/classes incohérents ou données hors train")
        seen.add(cid)
        labels[group] = label
    rng, heldout_groups = random.Random(SEED), [set() for _ in range(N_FOLDS)]
    for label in ("no_leak", "leak"):
        groups = sorted(g for g, value in labels.items() if value == label)
        if len(groups) < N_FOLDS:
            raise ValueError("Au moins trois groupes de chaque classe sont requis")
        rng.shuffle(groups)
        for i, group in enumerate(groups):
            heldout_groups[i % N_FOLDS].add(group)
    return [{"fold_id": i,
             "train_ids": sorted(r["clip_id"] for r in rows if r["group_id"] not in groups),
             "heldout_ids": sorted(r["clip_id"] for r in rows if r["group_id"] in groups),
             "heldout_groups": sorted(groups)} for i, groups in enumerate(heldout_groups)]


def batch_composition(rows):
    """Autopsie mécanique de l'ordre V1 (8 époques, lots 8), sans apprentissage."""
    records = []
    for epoch in range(1, 9):
        for number, indices in enumerate(epoch_batches(len(rows), 8, SEED, epoch), 1):
            chunk = [rows[i] for i in indices]
            counts = Counter(r["group_id"] for r in chunk)
            records.append({"epoch": epoch, "batch": number, "n_clips": len(chunk),
                "n_leak": sum(r["label"] == "leak" for r in chunk),
                "n_groups": len(counts), "max_clips_from_one_group": max(counts.values())})
    return {"purpose": "mechanical_v1_batch_recipe_not_v2_budget_selection",
            "epochs": 8, "batch_size": 8, "seed": SEED, "batches": records,
            "n_single_class_batches": sum(r["n_leak"] in (0, r["n_clips"]) for r in records),
            "group_clip_counts": dict(sorted(Counter(r["group_id"] for r in rows).items()))}


def fit_fixed_logreg(train_x, train_y, heldout_x):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train_x)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=SEED)
        model.fit(scaler.transform(train_x), train_y)
    if model.classes_.tolist() != [0, 1]:
        raise ValueError("Deux classes train attendues par le classifieur")
    scores = model.predict_proba(scaler.transform(heldout_x))[:, 1]
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores non finis ou hors bornes : aucun remplacement")
    return scores


def run_probes(rows, matrices, folds):
    ids = [r["clip_id"] for r in rows]
    positions = {cid: i for i, cid in enumerate(ids)}
    y = np.array([int(r["label"] == "leak") for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    result = {}
    for name, matrix in matrices.items():
        if matrix.shape[0] != len(rows) or not np.isfinite(matrix).all():
            raise ValueError("Matrice diagnostique incompatible ou non finie")
        reports, predictions = [], []
        for fold in folds:
            tr = [positions[cid] for cid in fold["train_ids"]]
            ho = [positions[cid] for cid in fold["heldout_ids"]]
            if (set(tr) & set(ho) or set(tr) | set(ho) != set(range(len(rows)))
                    or set(groups[tr]) & set(groups[ho])
                    or set(y[tr]) != {0, 1} or set(y[ho]) != {0, 1}):
                raise ValueError("Partition hors-groupes invalide")
            scores = fit_fixed_logreg(matrix[tr], y[tr], matrix[ho])
            report = metrics.evaluate(y[ho], scores, groups[ho], THRESHOLD)
            _, gy, gs = metrics.aggregate_clusters(y[ho], scores, groups[ho])
            # Moyenne sur valeurs pleines, pas sur les arrondis d'affichage du harness.
            report.update(fold_id=fold["fold_id"], n_fit_clips=len(tr),
                group_roc_auc_full=metrics.roc_auc(gy, gs),
                clip_roc_auc_full=metrics.roc_auc(y[ho], scores))
            reports.append(report)
            predictions.extend({"clip_id": ids[i], "inner_fold": fold["fold_id"],
                                "probability_leak": float(p)} for i, p in zip(ho, scores))
        if sorted(r["clip_id"] for r in predictions) != sorted(ids):
            raise ValueError("Chaque train doit être tenu à l'écart exactement une fois")
        result[name] = {"n_features": matrix.shape[1], "folds": reports,
            "mean_group_roc_auc": float(np.mean([r["group_roc_auc_full"] for r in reports])),
            "mean_clip_roc_auc": float(np.mean([r["clip_roc_auc_full"] for r in reports])),
            "predictions_by_fold": predictions}
    return result


def token_masks(ids, attention, class_ids, eos_ids):
    """Masques de réponses exactes : classe inclut ';', reste inclut l'EOS."""
    ids, attention = np.asarray(ids), np.asarray(attention)
    if (ids.ndim != 2 or attention.shape != ids.shape or len(class_ids) != len(ids)
            or not np.isin(attention, [0, 1]).all() or not eos_ids):
        raise ValueError("Tokens/masques incompatibles")
    classes = np.zeros(ids.shape, dtype=bool)
    counts = []
    for i, wanted in enumerate(class_ids):
        n = int(attention[i].sum())
        if (not wanted or n <= len(wanted) or not np.all(attention[i, :n] == 1)
                or np.any(attention[i, n:] != 0)
                or ids[i, :len(wanted)].tolist() != wanted
                or ids[i, n-len(eos_ids):n].tolist() != eos_ids):
            raise ValueError("Frontière classe/description/EOS ou padding cible invalide")
        classes[i, :len(wanted)] = True
        counts.append({"class_tokens": len(wanted), "description_tokens": n-len(wanted)-len(eos_ids),
                       "eos_tokens": len(eos_ids), "padding_tokens": ids.shape[1]-n})
    return classes, attention.astype(bool) & ~classes, counts


@contextmanager
def capture_loss_forward(llm):
    """Observer les véritables labels/logits de compute_loss sans refaire le collator."""
    original, calls = llm.forward, []
    def wrapped(*args, **kwargs):
        output = original(*args, **kwargs)
        calls.append((kwargs, output))
        return output
    llm.forward = wrapped
    try:
        yield calls
    finally:
        llm.forward = original


def supervision_audit(checkpoint, rows, cached, parity):
    """Autopsie des poids V1 déjà entraînés : PAS une mesure hors-groupes."""
    import torch
    import torch.nn.functional as functional
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate as collate
    from pipe.tslm.predict import Predictor
    from pipe.tslm.preprocessing import model_input, target_text
    from pipe.tslm.train import debug_rows, tensor_state_hash
    for filename, key in (("checksums.json", "checkpoint_checksums_sha256"),
                          ("temporal.pt", "temporal_sha256")):
        if sha256_file(checkpoint / filename) != parity[key]:
            raise ValueError("Checkpoint d'autopsie différent du checkpoint de parité")
    predictor = Predictor(checkpoint)
    model = predictor.model
    spec = model.scoring_spec()
    if spec != parity["scoring_spec"]:
        raise ValueError("Scoring d'autopsie différent du receipt de parité")
    selected = debug_rows(rows, 8)
    samples = [{**model_input(cached[r["clip_id"]]),
                "answer": target_text(r["label"], cached[r["clip_id"]])} for r in selected]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # eval retire le dropout : gradients mécaniques, pas simulation d'une étape stochastique.
    model.eval()
    modules = {"encoder": model.encoder, "projector": model.projector, "llm": model.llm}
    before = {name: tensor_state_hash(module) for name, module in modules.items()}
    if any(p.requires_grad for p in model.llm.parameters()):
        raise ValueError("Qwen doit être gelé avant l'autopsie")
    batch = collate(samples, normalize=False)
    with torch.enable_grad(), capture_loss_forward(model.llm) as calls:
        loss = model.compute_loss(batch)
    if len(calls) != 1 or not torch.isfinite(loss):
        raise ValueError("Une loss complète finie et un seul forward sont requis")
    kwargs, output = calls[0]
    answers = model.tokenizer([s["answer"] + model.get_eos_token() for s in samples],
        return_tensors="pt", padding=True, add_special_tokens=False)
    class_ids = [spec["class_token_ids"][0 if r["label"] == "leak" else 1] for r in selected]
    masks = token_masks(answers.input_ids.cpu().numpy(), answers.attention_mask.cpu().numpy(),
        class_ids, model.tokenizer.encode(model.get_eos_token(), add_special_tokens=False))
    class_mask, rest_mask, counts = masks
    labels = kwargs["labels"]
    prefix = labels.shape[1] - answers.input_ids.shape[1]
    expected = answers.input_ids.to(labels.device).masked_fill(
        answers.attention_mask.to(labels.device) == 0, -100)
    if (prefix < 1 or (labels[:, :prefix] != -100).any()
            or not torch.equal(labels[:, prefix:], expected)
            or not torch.equal(kwargs["attention_mask"][:, prefix:], answers.attention_mask.to(labels.device))
            or not torch.all(kwargs["attention_mask"][:, :prefix] == 1)):
        raise ValueError("Labels réels ou masque du prompt différents de la supervision attendue")
    targets = labels[:, 1:]
    logits = output.logits[:, :-1].float()
    losses = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                      reduction="none", ignore_index=-100).reshape(targets.shape)
    valid_count = (targets != -100).sum()
    terms = {}
    for name, mask in (("class", class_mask), ("description_including_eos", rest_mask)):
        padded = torch.zeros_like(labels, dtype=torch.bool)
        padded[:, prefix:] = torch.as_tensor(mask, device=labels.device)
        terms[name] = losses.masked_select(padded[:, 1:]).sum() / valid_count
    reconstructed = sum(terms.values())
    if not torch.allclose(loss.float(), reconstructed, atol=1e-6, rtol=1e-6):
        raise ValueError("La décomposition ne reconstruit pas la véritable loss")
    parameters = [(component, p) for component in ("encoder", "projector")
                  for p in modules[component].parameters() if p.requires_grad]
    gradients = {}
    for name, term in {**terms, "full": loss}.items():
        grads = torch.autograd.grad(term, [p for _, p in parameters], retain_graph=True, allow_unused=True)
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            raise ValueError(f"Gradient absent/non fini : {name}")
        gradients[name] = {component: float(torch.sqrt(sum(g.float().square().sum()
            for (owner, _), g in zip(parameters, grads) if owner == component)))
            for component in ("encoder", "projector")}
    after = {name: tensor_state_hash(module) for name, module in modules.items()}
    if before != after or any(p.grad is not None for p in model.llm.parameters()):
        raise ValueError("Poids/buffers modifiés ou gradient Qwen apparu pendant l'autopsie")
    return {"purpose": "mechanical_trained_v1_autopsy_not_heldout_generalization",
        "fold": "train", "mode": "eval_with_gradients_no_optimizer",
        "input_preprocessing_version": parity["preprocessing_version"],
        "checkpoint_preprocessing_version": predictor.metadata["preprocessing_version"],
        "checkpoint_checksums_sha256": parity["checkpoint_checksums_sha256"],
        "temporal_sha256": parity["temporal_sha256"], "scoring_spec": spec,
        "samples": [{"clip_id": r["clip_id"], "group_id": r["group_id"], **c}
                    for r, c in zip(selected, counts)],
        "class_fraction_of_supervised_tokens": float(class_mask.sum() / (class_mask | rest_mask).sum()),
        "loss_full": float(loss.detach()), "loss_reconstructed": float(reconstructed.detach()),
        "loss_contributions_same_total_token_denominator": {k: float(v.detach()) for k, v in terms.items()},
        "gradient_norms": gradients, "state_sha256_before": before, "state_sha256_after": after,
        "weights_unchanged": True, "optimizer_steps": 0}


def diagnose(args):
    if args.output.exists():
        raise FileExistsError("Dossier de diagnostic existant : aucun écrasement")
    if len(args.code_revision) != 40 or any(c not in "0123456789abcdef" for c in args.code_revision):
        raise ValueError("--code-revision exige un SHA Git complet")
    from pipe.tslm.preprocessing import CANONICAL_VERSION, decode_wav, preprocess_audio
    parity = verify_parity(args.parity_report, args.prepared, CANONICAL_VERSION)
    split = split_loader.load_split(args.manifests)
    rows = train_rows(split)
    expected_ids = sorted(c.clip_id for c in split.fold("val")) + ["c0128b879694e"]
    if parity.get("clip_ids") != expected_ids or "c0128b879694e" not in {r["clip_id"] for r in rows}:
        raise ValueError("Receipt de parité sans couverture des 208 val et du témoin train fixé")
    expected_md5 = load_expected_audio_md5(args.manifests)
    cached = load_cache(args.prepared, "train", rows, CANONICAL_VERSION)
    features_c1, audio_sha = [], {}
    for r in rows:
        cid = r["clip_id"]
        path = (args.data_root / split.path_of(cid)).resolve()
        if not path.is_relative_to(args.data_root.resolve()):
            raise ValueError("Chemin audio hors data-root")
        raw = path.read_bytes()
        verify_audio_bytes(raw, cid, expected_md5)
        waveform = decode_wav(raw)
        canonical = preprocess_audio(waveform, 8000, version=CANONICAL_VERSION)
        if canonical.dtype != cached[cid].dtype or not np.array_equal(canonical, cached[cid]):
            raise ValueError(f"Cache train non identique à l'entrée canonique : {cid}")
        features_c1.append(features.c1_envelope(waveform))
        audio_sha[cid] = sha256_file(path)
    folds = grouped_folds(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"schema": "pipe-inner-train-folds-v1", "seed": SEED, "n_folds": N_FOLDS,
        "source_fold": "train", "n_clips": len(rows), "split_sha256": split.sha256,
        "preprocessing_version": CANONICAL_VERSION, "cache_train_sha256": parity["cache_sha256"]["train"],
        "parity_report_sha256": sha256_file(args.parity_report),
        "algorithm": "sorted_groups_per_label(no_leak,leak);Random(seed).shuffle;round_robin_3",
        "folds": folds}
    write_json(args.output / "folds.json", protocol)  # Figer AVANT le premier fit.
    metadata = {"schema": "pipe-train-diagnostic-v1", "code_revision": args.code_revision,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_fold": "train",
        "n_probe_configurations": 1, "n_fixed_c1_configurations": 1, "n_total_fits": 6,
        "recipe": {"C": 1.0, "max_iter": 5000, "solver": "lbfgs", "class_weight": None,
                   "scaler": "StandardScaler_fit_inner_train_only", "seed": SEED},
        "threshold": THRESHOLD, "threshold_rule": "fixed_diagnostic_only_not_selected_not_served",
        "C1_note": "official_features_fixed_C1_not_validation_tuned_historical_C1",
        "probe_note": "canonical_TimeNet_4x64_flat_row_major_including_12_zero_padding_values",
        "folds_sha256": sha256_file(args.output / "folds.json"), "parity_provenance": parity,
        "parity_report_sha256": sha256_file(args.parity_report), "audio_sha256_train_only": audio_sha,
        "preparation_receipt_sha256": sha256_file(args.prepared / "preparation.json"),
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in
            [Path(__file__), ROOT / "scripts/eval/harness/features.py", ROOT / "scripts/eval/harness/metrics.py",
             ROOT / "src/pipe/tslm/campaign.py", ROOT / "src/pipe/tslm/train.py"]},
        "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn")},
        "official_validation_or_test_audio_read": False,
        "official_validation_or_test_scores_or_metrics_computed": False,
        "supervision_audit_requested": args.checkpoint is not None}
    write_json(args.output / "metadata.json", metadata)
    write_json(args.output / "batch-composition.json", batch_composition(rows))
    matrices = {"TimeNet256": np.stack([cached[r["clip_id"]].reshape(-1) for r in rows]),
                "C1_fixed": np.stack(features_c1)}
    report = run_probes(rows, matrices, folds)
    write_json(args.output / "probes.json", report)
    if args.checkpoint is not None:
        write_json(args.output / "supervision.json", supervision_audit(args.checkpoint, rows, cached, parity))
    print(json.dumps({"completed": True, "output": str(args.output), "source_fold": "train",
                      "n_clips": len(rows), "supervision_audit": args.checkpoint is not None}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "parity-report", "data-root", "manifests", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--checkpoint", type=Path,
        help="Option GPU : autopsie sans step de ce bundle V1, sur huit groupes train fixés")
    diagnose(parser.parse_args())


if __name__ == "__main__":
    main()
