"""Campagne préinscrite : audit → train → reload/validation → confirmation, processus séparés."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import numpy as np
from pipe.temporal_model import C1Detector, acoustic_features, decode_pcm, digest, v2_c1
from harness import metrics, split_loader
from audit_text import load_expected_audio_md5, verify_audio_bytes

PREPARATION_SHA = "ba193cb22418b32291df535f9b48668fea7de8ae166c558a0c7f075e929162a7"


def write(path, obj):
    with Path(path).open("x") as file:
        json.dump(obj, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def partition(record, config):
    if record["topology"] == config["confirmation_topology"]:
        return "confirmation"
    if record["topology"] != config["train_topology"]:
        raise ValueError("Topologie inconnue")
    return "validation" if record["condition"] in config["validation_conditions"] else "train"


def audit(args):
    if not args.authorize_repurpose:
        raise ValueError("Réaffectation explicitement autorisée requise")
    config = json.loads(args.config.read_text())
    if digest(args.external / "checksums.json") != PREPARATION_SHA:
        raise ValueError("Préparation Aghashahi différente de l'archive auditée")
    checksums = json.loads((args.external / "checksums.json").read_text())
    for name, sha in checksums.items():
        path = (args.external / name).resolve()
        if not path.is_relative_to(args.external.resolve()) or digest(path) != sha:
            raise ValueError("Intégrité externe incorrecte")
    records = json.loads((args.external / "recordings.json").read_text())
    primary = sorted([r for r in records if r["subset"] == "primary"], key=lambda r: r["recording_id"])
    if len(primary) != 120 or len({r["recording_id"] for r in primary}) != 120:
        raise ValueError("120 enregistrements uniques requis")
    groups = {}
    for record in primary:
        record["partition"] = partition(record, config)
        signal = np.load(args.external / record["array_path"], allow_pickle=False)
        record["full_scale_samples"] = int(((signal <= -1) | (signal >= 32767 / 32768)).sum())
        group = record["condition_group_id"]
        groups.setdefault(group, []).append(record)
    if len(groups) != 60 or any(len(g) != 2 or {r["sensor"] for r in g} != {"H1", "H2"}
            or len({r["partition"] for r in g}) != 1 or len({r["label"] for r in g}) != 1 for g in groups.values()):
        raise ValueError("Groupes/capteurs incohérents")
    counts = {}
    for fold, size, negatives in (("train", 40, 8), ("validation", 20, 4), ("confirmation", 60, 12)):
        selected = [r for r in primary if r["partition"] == fold]
        labels = Counter(r["label"] for r in selected)
        if len(selected) != size or labels != {"no_leak": negatives, "leak": size - negatives}:
            raise ValueError("Population différente du protocole")
        counts[fold] = {"recordings": size, "condition_groups": size // 2, "labels": dict(labels)}
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "registration.json", {"config": config, "config_sha256": digest(args.config),
        "external": str(args.external.resolve()), "external_checksums_sha256": PREPARATION_SHA,
        "source_revision": args.revision, "created_at": datetime.now(timezone.utc).isoformat(),
        "authorization": "Icham explicit repurposing; no WhatsApp; GPU endpoint",
        "previous_external_role_superseded_for_this_campaign_only": True,
        "counts": counts, "records": primary, "event_onsets_available": False,
        "limitations": ["condition groups heuristic, not proven independent sessions",
                        "confirmation is another topology of same laboratory, not another site"]})
    print(json.dumps(counts), flush=True)


def sequences(registration, fold, detector):
    rows = [r for r in registration["records"] if r["partition"] == fold]
    root = Path(registration["external"])
    values = []
    for row in rows:
        path = (root / row["array_path"]).resolve()
        if not path.is_relative_to(root) or digest(path) != row["array_sha256"]:
            raise ValueError("Source de séquence différente du gel")
        x = np.load(path, allow_pickle=False)
        if x.dtype != np.dtype("<f8") or x.shape != (240000,):
            raise ValueError("Signal préparé incompatible")
        # Protocole expérimental inclusif : aucune exclusion guidée par les scores.
        # Le service reste strict et s'abstient sur la pleine échelle.
        values.append(detector.sequence_inputs(x, allow_full_scale=True))
    return rows, np.asarray(values), np.asarray([int(r["label"] == "leak") for r in rows])


def train(args, registration):
    import torch
    from pipe.sequence_model import SequenceModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if not torch.cuda.is_available():
        raise RuntimeError("Cette campagne exige le GPU CUDA autorisé")
    config = registration["config"]
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    rng = np.random.default_rng(config["seed"])
    bundle = args.output / "bundle"
    bundle.mkdir(exist_ok=False)
    split = split_loader.load_split(ROOT / "manifests")
    clips = sorted(split.fold("train"), key=lambda c: c.clip_id)
    if len(clips) != 598 or len({c.group_id for c in clips}) != 102:
        raise ValueError("Train Zenodo divergent")
    md5 = load_expected_audio_md5(ROOT / "manifests")
    X = []
    for clip in clips:
        path = (args.data_root / split.path_of(clip.clip_id)).resolve()
        if not path.is_relative_to(args.data_root.resolve()):
            raise ValueError("Chemin audio hors racine")
        raw = path.read_bytes()
        verify_audio_bytes(raw, clip.clip_id, md5)
        X.append(acoustic_features(decode_pcm(raw)))
    X, y = np.asarray(X), np.asarray([c.label for c in clips])
    if args.reuse_c1:
        prior = C1Detector(args.reuse_c1, args.reuse_c1_sha256)
        scaler, classifier = prior.scaler, prior.classifier
        shutil.copy2(args.reuse_c1 / "c1.pkl", bundle / "c1.pkl")
        c1_sha = digest(bundle / "c1.pkl")
    else:
        scaler, classifier = v2_c1.fit_final(X, y, config["c1_C"])
        c1_sha = v2_c1.save_checkpoint(bundle / "c1.pkl", scaler, classifier)
    loaded = v2_c1.load_checkpoint(bundle / "c1.pkl", expected_sha256=c1_sha, trusted=True)
    delta = float(np.max(np.abs(v2_c1.predict_probability(*loaded, X) - v2_c1.predict_probability(scaler, classifier, X))))
    if delta != 0:
        raise ValueError("Reload C1 divergent")
    # Bundle provisoire distinct, uniquement pour réutiliser le même lecteur de signaux.
    write(bundle / "bundle.json", {"config": config, "files": {"c1.pkl": c1_sha}})
    detector = C1Detector(bundle, digest(bundle / "bundle.json"))
    rows, raw_X, y = sequences(registration, "train", detector)
    mean, scale = raw_X.reshape(-1, 10).mean(0), raw_X.reshape(-1, 10).std(0)
    scale[scale == 0] = 1
    np.savez(bundle / "normalization.npz", mean=mean, scale=scale)
    x = torch.tensor((raw_X - mean) / scale, dtype=torch.float32, device="cuda")
    target = torch.tensor(y, dtype=torch.float32, device="cuda")
    model = SequenceModel(config["hidden_size"]).cuda()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    weight = torch.tensor(float((y == 0).sum() / (y == 1).sum()), device="cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    started, step = time.perf_counter(), 0
    with (args.output / "training.jsonl").open("x") as log:
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            order = rng.permutation(len(x))
            for start in range(0, len(x), config["batch_size"]):
                idx = torch.tensor(order[start:start + config["batch_size"]], device="cuda")
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x[idx])
                loss = loss_fn(logits[:, -1], target[idx])
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                if not torch.isfinite(loss) or any(p.grad is None for p in model.parameters()):
                    raise ValueError("Loss/gradient invalide")
                optimizer.step()
                step += 1
            model.eval()
            with torch.inference_mode():
                logits, _ = model(x)
                nll = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, -1], target)
                correct = int(((logits[:, -1] >= 0) == target.bool()).sum())
            record = {"epoch": epoch, "steps": step, "train_nll": float(nll), "correct": correct,
                      "n_sequences": len(x), "gradient_norm_last": float(norm)}
            log.write(json.dumps(record) + "\n"); log.flush()
            if epoch % 20 == 0:
                print(json.dumps(record), flush=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()) or not any(
            not torch.equal(initial[k], v) for k, v in model.state_dict().items()):
        raise ValueError("Poids non finis ou inchangés")
    torch.save(model.cpu().state_dict(), bundle / "lstm.pt")
    orderless = make_pipeline(StandardScaler(), LogisticRegression(C=config["orderless_C"],
        class_weight="balanced", max_iter=5000, random_state=config["seed"]))
    orderless.fit(raw_X.mean(1), y)
    with (bundle / "orderless.pkl").open("xb") as file:
        pickle.dump(orderless, file)
    with torch.inference_mode():
        logits, _ = model(x.cpu())
        reference = torch.sigmoid(logits[:, -1]).tolist()
    write(args.output / "training-result.json", {"steps": step, "epochs": config["epochs"],
        "elapsed_seconds": time.perf_counter() - started, "c1_train_clips": 598,
        "sequence_train_records": len(rows), "c1_reload_max_diff": delta,
        "train_reference_ids": [r["recording_id"] for r in rows], "train_reference_scores": reference,
        "torch": torch.__version__, "numpy": np.__version__, "device": torch.cuda.get_device_name(),
        "registration_sha256": digest(args.output / "registration.json"),
        "reused_c1_from": str(args.reuse_c1) if args.reuse_c1 else None})
    metadata = {"schema": "pipe.temporal-bundle.v1", "config": config,
        "registration_sha256": digest(args.output / "registration.json"),
        "files": {name: digest(bundle / name) for name in ("c1.pkl", "lstm.pt", "normalization.npz", "orderless.pkl")},
        "default_detector": "c1", "event_detector_validated": False, "sequence_supervision": "terminal_only"}
    # Le fichier provisoire reste conservé ; le manifeste final est celui à épingler.
    (bundle / "bundle.json").rename(bundle / "c1-only.json")
    write(bundle / "bundle.json", metadata)
    write(args.output / "bundle-receipt.json", {"bundle_sha256": digest(bundle / "bundle.json")})


def evaluate(args, registration):
    import torch
    from pipe.sequence_model import load_sequence, sequence_probability
    torch.set_num_threads(2)
    receipt = json.loads((args.output / "bundle-receipt.json").read_text())
    detector = C1Detector(args.output / "bundle", receipt["bundle_sha256"])
    model, mean, scale = load_sequence(detector)
    with (detector.directory / "orderless.pkl").open("rb") as file:
        orderless = pickle.load(file)  # Artefact propre dont l'empreinte vient d'être vérifiée.
    # Reload CPU/GPU comparé sur train seulement, avant toute lecture de réserve.
    rows, X, _ = sequences(registration, "train", detector)
    reference = json.loads((args.output / "training-result.json").read_text())
    actual = [sequence_probability(model, x, mean, scale) for x in X]
    if [r["recording_id"] for r in rows] != reference["train_reference_ids"]:
        raise ValueError("Mapping du contrôle reload divergent")
    reload_delta = float(np.max(np.abs(np.asarray(actual) - reference["train_reference_scores"])))
    if reload_delta > 1e-5:
        raise ValueError("Reload LSTM divergent")
    fold = args.phase
    validation = None
    if fold == "confirmation":
        validation = json.loads((args.output / "validation" / "report.json").read_text())
        if validation["bundle_sha256"] != receipt["bundle_sha256"]:
            raise ValueError("La validation doit porter sur le même checkpoint")
    rows, X, y = sequences(registration, fold, detector)
    scores = {"c1_median": np.median(X[:, :, 9], axis=1), "c1_ewma": X[:, 0, 9].copy(),
        "orderless": orderless.predict_proba(X.mean(1))[:, 1],
        "lstm": np.asarray([sequence_probability(model, x, mean, scale) for x in X])}
    alpha = registration["config"]["ewma_alpha"]
    for t in range(1, 30):
        scores["c1_ewma"] = alpha * X[:, t, 9] + (1 - alpha) * scores["c1_ewma"]
    groups = np.asarray([r["condition_group_id"] for r in rows])
    reports = {name: metrics.evaluate(y, s, groups, .5) for name, s in scores.items()}
    auc = {name: metrics.roc_auc(*metrics.aggregate_clusters(y, s, groups)[1:]) for name, s in scores.items()}
    baseline = (validation["selected_baseline"] if validation else max(
        ("c1_median", "c1_ewma", "orderless"), key=lambda name: auc[name]))
    _, gy, ls = metrics.aggregate_clusters(y, scores["lstm"], groups)
    _, _, bs = metrics.aggregate_clusters(y, scores[baseline], groups)
    rng = np.random.default_rng(registration["config"]["seed"])
    deltas = []
    for _ in range(registration["config"]["bootstrap_draws"]):
        # Bootstrap apparié stratifié : mêmes conditions pour les deux modèles.
        idx = np.r_[rng.choice(np.flatnonzero(gy == 0), (gy == 0).sum(), replace=True),
                    rng.choice(np.flatnonzero(gy == 1), (gy == 1).sum(), replace=True)]
        deltas.append(metrics.roc_auc(gy[idx], ls[idx]) - metrics.roc_auc(gy[idx], bs[idx]))
    gain = auc["lstm"] - auc[baseline]
    ci = np.quantile(deltas, [.025, .975]).tolist()
    minimum = registration["config"]["minimum_auc_gain"]
    eligible = gain >= minimum if validation is None else validation["eligible"] and gain >= minimum and ci[0] > 0
    report = {"partition": fold, "bundle_sha256": receipt["bundle_sha256"], "reload_max_diff": reload_delta,
        "metrics": reports, "group_auc": auc, "selected_baseline": baseline, "lstm_auc_gain": gain,
        "paired_gain_95_interval": ci, "eligible": bool(eligible), "n_recordings": len(rows),
        "n_condition_groups": len(gy), "n_nonleak_groups": int((gy == 0).sum()),
        "classification_sequence_only": True, "event_metrics": None, "n_candidate_systems": 4,
        "full_scale_recordings_included": sum(r["full_scale_samples"] > 0 for r in rows),
        "service_quality_policy_applied": False,
        "official_zenodo_val_test_used": False, "independent_site_confirmation": False}
    destination = args.output / fold
    destination.mkdir(exist_ok=False)
    write(destination / "report.json", report)
    with (destination / "predictions.csv").open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["recording_id", *scores])
        for i, row in enumerate(rows):
            writer.writerow([row["recording_id"], *(float(s[i]) for s in scores.values())])
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["audit", "train", "validation", "confirmation"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/temporal/c1_lstm.json")
    parser.add_argument("--revision")
    parser.add_argument("--reuse-c1", type=Path)
    parser.add_argument("--reuse-c1-sha256")
    parser.add_argument("--authorize-repurpose", action="store_true")
    args = parser.parse_args()
    if args.phase == "audit":
        audit(args)
    else:
        registration = json.loads((args.output / "registration.json").read_text())
        if digest(args.config) != registration["config_sha256"]:
            raise ValueError("Configuration modifiée après préinscription")
        if args.phase == "train":
            train(args, registration)
        else:
            evaluate(args, registration)
