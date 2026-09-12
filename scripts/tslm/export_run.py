"""Exporter T0/T1/T2/T3 sans métriques, puis lancer le contrôleur officiel."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
sys.path.insert(0, str(ROOT / "scripts/temporal"))
from harness import split_loader  # noqa: E402
from stress import SEED_SCHEME, STRESS_SEED, TRANSFORMS, clip_rng  # noqa: E402

HARNESS_COMMIT = "6dfdf63580bc15ce6a1cf0817b9da3569553aca1"
STRESS_SHA256 = "7c37e001d270655d2e54ba2087e8b8e476da8cc42948f92bd83c669773ac6350"
# Même valeur gelée que prepare.MANIFEST_HASHES ; pas d'import du pipeline
# TimeF/libarchive pour cette vérification de conformité avant chargement GPU.
FROZEN_AUDIT_SHA256 = "1a3bd3c18ad6d886d42ecc85ba3a5cceaa45a9084102fc112b53e66782e29d61"


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_predictions(path, expected_ids, probabilities):
    """Compléter notre squelette uniquement, sans arrondi à dix décimales."""
    if len(expected_ids) != len(set(expected_ids)) or set(probabilities) != set(expected_ids):
        raise ValueError("Couverture ou unicité des clip_id incorrecte")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Score non fini ou hors [0,1] ; aucun remplacement autorisé")
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["clip_id", "probability_leak"])
        for clip_id in sorted(expected_ids):
            writer.writerow([clip_id, format(probabilities[clip_id], ".17g")])


def frozen_provenance(checkpoint, reload_report, predictor):
    """Vérifications avant toute lecture/inférence test ; aucun critère de qualité."""
    metadata = predictor.metadata
    count = metadata.get("n_configs_compared")
    if type(count) is not int or not 1 <= count <= 3:
        raise ValueError("n_configs_compared doit décrire les 1 à 3 configurations réellement comparées")
    if not metadata.get("training_commit") or not metadata.get("config_hash"):
        raise ValueError("Provenance entraînement/configuration absente")
    if metadata.get("test_labels_not_used_for_tuning") is not True:
        raise ValueError("Déclaration explicite d'absence de réglage sur le test absente du checkpoint")
    checksums = json.loads((checkpoint / "checksums.json").read_text())
    required = {"scoring_spec.json", "reload-expected.json", "reload-example.wav",
                "training-report.json", "metadata.json", "temporal.pt"}
    if not required.issubset(checksums):
        raise ValueError("Bundle V1 incomplet : scoring/reload/provenance non figés")
    specification = json.loads((checkpoint / "scoring_spec.json").read_text())
    if specification != predictor.model.scoring_spec():
        raise ValueError("Méthode de scoring différente du checkpoint figé")
    reload = json.loads(reload_report.read_text())
    digest = sha256_file(checkpoint / "checksums.json")
    if (reload.get("fresh_process_reload") is not True
            or reload.get("score_match") is not True
            or reload.get("deterministic_output_match") is not True
            or reload.get("input_paths_match") is not True
            or reload.get("checkpoint_checksums_sha256") != digest):
        raise ValueError("Reload frais validé absent ou effectué sur un autre checkpoint")
    return {"training_commit": metadata["training_commit"],
            "config_hash": metadata["config_hash"], "n_configs_compared": count,
            "checkpoint_checksums_sha256": digest,
            "temporal_sha256": checksums["temporal.pt"],
            "scoring_spec_sha256": checksums["scoring_spec.json"],
            "scoring_spec": specification, "reload_report_sha256": sha256_file(reload_report)}


def export(args):
    # Refuser avant même de charger plusieurs Go de poids ; ne jamais réinitialiser un run.
    if args.output.exists():
        raise FileExistsError(f"Dossier de run déjà présent, refus d'écraser : {args.output}")
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                              for c in args.run_id):
        raise ValueError("run_id doit être non vide et limité à A-Z a-z 0-9 . _ -")
    if len(args.code_revision) != 40 or any(c not in "0123456789abcdef" for c in args.code_revision):
        raise ValueError("code-revision doit être le SHA Git complet du code transféré (40 caractères hex)")
    audit_sha256 = sha256_file(args.manifests / "split_v2_audit.csv")
    if audit_sha256 != FROZEN_AUDIT_SHA256:
        raise ValueError("Mapping clip_id vers WAV modifié : SHA256 de split_v2_audit.csv divergent")
    split = split_loader.load_split(args.manifests)
    ids = sorted(c.clip_id for c in split.clips if c.fold in ("val", "test"))
    if len(ids) != 402 or len(set(ids)) != 402:
        raise ValueError("Le split gelé doit fournir exactement 402 clips val/test uniques")
    if sha256_file(ROOT / "scripts/temporal/stress.py") != STRESS_SHA256:
        raise ValueError("Transformation officielle modifiée : reprendre la révision épinglée")
    from pipe.tslm.predict import Predictor
    from pipe.tslm.preprocessing import decode_wav

    checkpoint = args.checkpoint.resolve()
    predictor = Predictor(checkpoint, device=args.device)
    provenance = frozen_provenance(checkpoint, args.reload_report, predictor)
    args.output.mkdir(parents=True, exist_ok=False)
    checker = [sys.executable, str(ROOT / "scripts/eval/check_run.py"),
               "--manifests", str(args.manifests.resolve())]
    subprocess.run([*checker, "--template", str(args.output)], check=True)
    # Les chemins restent dans l'adaptateur, jamais dans les entrées du modèle.
    data_root = args.data_root.resolve()
    probabilities = {}
    for index, clip_id in enumerate(ids, 1):
        path = (data_root / split.path_of(clip_id)).resolve()
        if not path.is_relative_to(data_root):
            raise ValueError("Chemin audio hors data-root")
        raw = path.read_bytes()
        if args.transform == "T0":
            probability = predictor.score(raw)
        else:
            waveform = decode_wav(raw)
            transformed = TRANSFORMS[args.transform]["fn"](waveform, clip_rng(args.transform, clip_id))
            probability = predictor.score_waveform(transformed, 8000)
        probability = float(probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"Score invalide pour {clip_id}, export arrêté sans valeur de secours")
        probabilities[clip_id] = probability
        if index % 20 == 0 or index == len(ids):
            print(f"{args.transform} : {index}/{len(ids)} clips prédits", flush=True)
    write_predictions(args.output / "predictions.csv", ids, probabilities)
    source_paths = ["scripts/tslm/export_run.py", "scripts/eval/check_run.py",
                    "scripts/eval/harness/contract.py", "scripts/eval/harness/split_loader.py",
                    "scripts/temporal/stress.py", "src/pipe/tslm/model.py",
                    "src/pipe/tslm/predict.py", "src/pipe/tslm/preprocessing.py",
                    "scripts/timenet/leakless_acoustic/connector.py"]
    metadata = {"run_id": args.run_id, "model_name": "AcousticQwenSP / Qwen3.5-4B V1",
                "checkpoint": str(checkpoint), **provenance,
                "split_filename": split_loader.FROZEN_SPLIT_NAME, "split_sha256": split.sha256,
                "split_audit_filename": "split_v2_audit.csv", "split_audit_sha256": audit_sha256,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "threshold_rule": "aucun seuil appliqué — probabilités brutes non calibrées ; "
                                  "sommes des log-probabilités des continuations complètes ; "
                                  "softmax à deux classes ; aucune normalisation par longueur",
                "test_labels_not_used_for_tuning": True,
                "export_commit": args.code_revision,
                "harness_source_commit": HARNESS_COMMIT, "transform": args.transform,
                "stress_seed": STRESS_SEED, "stress_seed_scheme": SEED_SCHEME,
                "numpy_version": np.__version__,
                "source_sha256": {name: sha256_file(ROOT / name) for name in source_paths}}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    subprocess.run([*checker, "--run", str(args.output), "--inspect"], check=True)
    if set(p.name for p in args.output.iterdir()) != {"metadata.json", "predictions.csv"}:
        raise ValueError("Le run doit contenir uniquement metadata.json et predictions.csv")
    print(f"Run exporté : {args.output} ; aucune métrique finale calculée.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reload-report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-revision", required=True,
                        help="SHA Git complet réellement transféré ; aucun .git requis sur la machine GPU")
    parser.add_argument("--transform", choices=tuple(TRANSFORMS), default="T0")
    parser.add_argument("--manifests", type=Path, default=ROOT / "manifests")
    parser.add_argument("--device", default="cuda")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
