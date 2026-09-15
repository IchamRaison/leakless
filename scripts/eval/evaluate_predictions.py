#!/usr/bin/env python3
"""L'unique moteur d'évaluation du projet. Tout passe par ici, TSLM compris.

Entrée : un ou plusieurs dossiers de run au format du contrat
(`metadata.json` + `predictions.csv`). Les contrôles C0-C3 et le TSLM de Hicham
ont exactement la même forme : il n'y a pas de chemin privilégié.

Parcours historique par défaut :

  - les folds, les étiquettes et les clusters viennent de `split_v2.csv`, jamais
    du fichier de prédictions ; le contrat refuse une colonne `fold` ou `label` ;
  - le seuil est choisi sur la VALIDATION, au niveau cluster, avec la règle gelée
    de `harness/metrics.py`. Une valeur de seuil fournie par le run est ignorée
    pour le calcul et seulement reportée pour information ;
  - les intervalles rééchantillonnent des CLUSTERS, jamais des clips ;
  - les comparaisons sont appariées : un seul tirage de clusters sert aux deux
    modèles comparés, et les deux doivent couvrir la même population de clips.

Usage :
  python3 scripts/eval/evaluate_predictions.py --runs <dir> [<dir> ...] \\
      [--manifests manifests] [--compare A:B ...] [--out metrics.json]

Opt-in externe : --external-manifest PATH --external-manifest-sha256 SHA
  --threshold-evidence RUN_ID=PATH (répété pour chaque run).
Les reçus existants de run_v2_campaign fournissent les seuils validation déjà
figés ; aucun seuil n'est ajusté ici sur l'externe ni sur ses stress T1/T2/T3.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from harness import contract, metrics, split_loader  # noqa: E402

EXTERNAL_THRESHOLD_RULE = "argmax_validation_cluster_macro_f1;median_cluster_score;smallest_threshold_on_exact_tie"


@dataclass(frozen=True)
class ExternalThreshold:
    """Preuve liée à un run précis, produite par load_external_threshold."""
    value: float
    run_id: str
    evidence_sha256: str
    checkpoint_key: tuple[str, str]
    run_files_sha256: dict
    provenance: dict


def _sha256(path):
    return split_loader.sha256_of(Path(path))


def _is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _run_hashes(run):
    return {name: _sha256(run.source / name) for name in ("metadata.json", "predictions.csv")}


def load_external_threshold(run, path) -> ExternalThreshold:
    """Vérifie les reçus TSLM/C1 existants, sans ouvrir labels ni checkpoint.

    Producteur : metadata.transform et threshold_provenance_sha256 obligatoires.
    TSLM : model_identity identique au reçu et checksums/temporal/config plats
    cohérents. C1 : checkpoint_sha256 identique au reçu. Le producteur vérifie
    les poids matériels avant inférence ; cet évaluateur lie les artefacts.
    """
    expected = run.metadata.get("threshold_provenance_sha256")
    path = Path(path)
    if not _is_digest(expected) or path.stat().st_size > 1_000_000:
        raise ValueError("SHA du reçu de seuil complet et artefact borné requis")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Reçu de seuil différent de l'empreinte déclarée par le run")
    receipt = json.loads(content)
    if not isinstance(receipt, dict):
        raise ValueError("Objet de preuve de seuil requis")
    common = {"fit_fold", "threshold", "threshold_repr", "rule", "split_sha256",
              "validation_predictions_sha256", "threshold_method_sha256"}
    value = receipt.get("threshold")
    if (receipt.get("fit_fold") != "val" or type(value) not in (int, float)
            or not math.isfinite(value) or receipt.get("threshold_repr") != repr(float(value))
            or receipt.get("rule") != EXTERNAL_THRESHOLD_RULE
            or receipt.get("split_sha256") != split_loader.FROZEN_SPLIT_SHA256
            or not _is_digest(receipt.get("validation_predictions_sha256"))
            or receipt.get("threshold_method_sha256") != _sha256(metrics.__file__)
            or run.metadata.get("transform") not in ("T0", "T1", "T2", "T3")
            or ("threshold" in run.metadata and run.metadata["threshold"] != value)):
        raise ValueError("Seuil validation, méthode, précision ou transformation incompatible")
    if receipt.get("schema_version") == "pipe-threshold-evidence-v1":
        identity = receipt.get("model_identity")
        identity_fields = {"checkpoint_checksums_sha256", "temporal_sha256", "config_hash",
                           "model_version", "preprocessing_version", "scoring_spec_sha256", "source_sha256"}
        if (set(receipt) != common | {"schema_version", "model_identity"}
                or not isinstance(identity, dict) or set(identity) != identity_fields
                or any(not _is_digest(identity[key]) for key in (
                    "checkpoint_checksums_sha256", "temporal_sha256", "config_hash", "scoring_spec_sha256"))
                or any(not isinstance(identity[key], str) or not identity[key]
                       for key in ("model_version", "preprocessing_version"))
                or not isinstance(identity["source_sha256"], dict) or not identity["source_sha256"]
                or any(not _is_digest(v) for v in identity["source_sha256"].values())
                or run.metadata.get("model_identity") != identity
                or any(run.metadata.get(key) != identity[key] for key in (
                    "checkpoint_checksums_sha256", "temporal_sha256", "config_hash"))):
            raise ValueError("Reçu de seuil TSLM lié à un autre checkpoint, score ou preprocessing")
        key = ("tslm", identity["checkpoint_checksums_sha256"])
        schema = receipt["schema_version"]
    elif receipt.get("schema") == "pipe-v2-c1-threshold-v1":
        if (set(receipt) != common | {"schema", "checkpoint_sha256"}
                or not _is_digest(receipt.get("checkpoint_sha256"))
                or run.metadata.get("checkpoint_sha256") != receipt["checkpoint_sha256"]):
            raise ValueError("Reçu de seuil C1 lié à un autre checkpoint")
        key = ("c1", receipt["checkpoint_sha256"])
        schema = receipt["schema"]
    else:
        raise ValueError("Schéma de seuil externe inconnu : reçus TSLM/C1 existants requis")
    hashes = _run_hashes(run)
    provenance = {"schema": schema, "fit_fold": "val", "threshold": float(value),
        "threshold_repr": repr(float(value)), "rule": receipt["rule"],
        "threshold_provenance_sha256": expected, "validation_split_sha256": receipt["split_sha256"],
        "validation_predictions_sha256": receipt["validation_predictions_sha256"],
        "threshold_method_sha256": receipt["threshold_method_sha256"],
        "checkpoint_kind": key[0], "checkpoint_sha256": key[1], "run_files_sha256": hashes}
    return ExternalThreshold(float(value), run.run_id, expected, key, hashes, provenance)


def _external_threshold(split, run, proof):
    if (not split.clips or any(c.fold != "external" for c in split.clips)
            or set(run.probabilities) != {c.clip_id for c in split.clips}
            or not isinstance(proof, ExternalThreshold) or proof.run_id != run.run_id
            or proof.run_files_sha256 != _run_hashes(run)):
        raise ValueError("Scope externe exact et preuve de seuil liée au run inchangé requis")
    # Revérifier le manifeste effectivement chargé et le contrat strict ; aucun
    # assouplissement du loader historique, aucune cible issue des prédictions.
    verified = contract.load_run(run.source, split, folds=("external",),
                                 external_manifest_name=split.manifest_path.name,
                                 external_manifest_sha256=split.sha256)
    if (verified.run_id != run.run_id or verified.metadata != run.metadata
            or verified.probabilities != run.probabilities):
        raise ValueError("Objet de run différent des prédictions et métadonnées figées")
    return proof.value


def _same_checkpoint_thresholds(proofs):
    by_checkpoint = {}
    for proof in proofs:
        if proof.checkpoint_key in by_checkpoint and by_checkpoint[proof.checkpoint_key] != proof.evidence_sha256:
            raise ValueError("Un même checkpoint doit conserver le même reçu de seuil sur T0/T1/T2/T3")
        by_checkpoint[proof.checkpoint_key] = proof.evidence_sha256


def vectors(split, run, fold: str):
    """(y, score, groups) alignés par clip_id, dans un ordre déterministe."""
    clips = sorted(split.fold(fold), key=lambda c: c.clip_id)
    y = np.array([c.label for c in clips])
    g = np.array([c.group_id for c in clips])
    s = np.array([run.probabilities[c.clip_id] for c in clips])
    ids = [c.clip_id for c in clips]
    return y, s, g, ids


def evaluate_run(split, run, *, external_threshold: ExternalThreshold | None = None) -> dict:
    if external_threshold is not None:
        threshold = _external_threshold(split, run, external_threshold)
        folds = ("external",)
    else:
        if any(c.fold == "external" for c in split.clips):
            raise ValueError("Un reçu de seuil figé est obligatoire pour l'externe")
        yv, sv, gv, _ = vectors(split, run, "val")
        threshold = metrics.pick_threshold(yv, sv, gv)      # validation SEULEMENT
        folds = ("val", "test")

    out = {
        "run_id": run.run_id,
        "model_name": run.model_name,
        "checkpoint": run.metadata.get("checkpoint"),
        "training_commit": run.metadata.get("training_commit"),
        "config_hash": run.metadata.get("config_hash"),
        "timestamp": run.metadata.get("timestamp"),
        "threshold_rule": run.metadata.get("threshold_rule"),
        "threshold_recomputed_on_val": round(float(threshold), 6),
        "threshold_declared_by_run": run.metadata.get("threshold"),
        "contract_checks": [{"name": c.name, "passed": c.passed, "coverage": c.coverage}
                            for c in run.checks],
        "folds": {},
    }
    if external_threshold is not None:
        out["threshold_recomputed_on_val"] = None
        out["fixed_threshold_provenance"] = external_threshold.provenance
        out["transform"] = run.metadata["transform"]
    for fold in folds:
        y, s, g, _ = vectors(split, run, fold)
        block = metrics.evaluate(y, s, g, threshold)
        block["bootstrap_ci95"] = metrics.bootstrap_ci(y, s, g, threshold)
        out["folds"][fold] = block
    return out


def compare(split, runs: dict, a: str, b: str, fold: str = "test", *, thresholds: dict | None = None) -> dict:
    ra, rb = runs[a], runs[b]
    external = fold == "external" or any(c.fold == "external" for c in split.clips)
    if external or thresholds is not None:
        if fold != "external" or thresholds is None or a not in thresholds or b not in thresholds:
            raise ValueError("Comparaison externe : fold explicite et deux preuves de seuil requises")
        ta = _external_threshold(split, ra, thresholds[a])
        tb = _external_threshold(split, rb, thresholds[b])
        _same_checkpoint_thresholds([thresholds[a], thresholds[b]])
    ya, sa, ga, ida = vectors(split, ra, fold)
    yb, sb, gb, idb = vectors(split, rb, fold)
    if ida != idb:
        raise ValueError(
            f"populations de clips différentes entre {a} et {b} — comparaison refusée")
    if not external:
        yv_a, sv_a, gv_a, _ = vectors(split, ra, "val")
        yv_b, sv_b, gv_b, _ = vectors(split, rb, "val")
        ta = metrics.pick_threshold(yv_a, sv_a, gv_a)
        tb = metrics.pick_threshold(yv_b, sv_b, gv_b)
    d = metrics.paired_bootstrap_delta(ya, ga, sa, sb, ta, tb)
    d.update({"fold": fold, "a": a, "b": b, "n_clips": int(len(ya)),
              "avertissement": "aucune significativité statistique n'est revendiquée : "
                               f"{d['n_clusters']} groupes heuristiques seulement"})
    if external:
        d["fixed_threshold_provenance"] = {a: thresholds[a].provenance, b: thresholds[b].provenance}
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="paires « A:B » ; par défaut toutes les paires")
    ap.add_argument("--out")
    ap.add_argument("--external-manifest", type=Path)
    ap.add_argument("--external-manifest-sha256")
    ap.add_argument("--threshold-evidence", action="append", default=[], metavar="RUN_ID=PATH",
                    help="Mode externe seulement : reçu validation existant pour chaque run, sans nouveau fit")
    args = ap.parse_args()

    external = args.external_manifest is not None or args.external_manifest_sha256 is not None or bool(args.threshold_evidence)
    threshold_paths, thresholds = {}, {}
    if external:
        if args.external_manifest is None or not _is_digest(args.external_manifest_sha256) or not args.threshold_evidence:
            ap.error("Mode externe : manifeste, SHA-256 épinglé et reçus de seuil obligatoires ensemble")
        if args.out and (Path(args.out).exists() or Path(args.out).is_symlink()):
            ap.error("Rapport externe déjà présent : aucun écrasement")
        from external_manifest import load_external_manifest
        split = load_external_manifest(args.external_manifest)
        if split.sha256 != args.external_manifest_sha256:
            raise ValueError("Manifeste externe différent du SHA explicitement épinglé")
        for item in args.threshold_evidence:
            run_id, separator, path = item.partition("=")
            if not separator or not run_id or not path or run_id in threshold_paths:
                ap.error("Un reçu RUN_ID=PATH unique par run est requis")
            threshold_paths[run_id] = Path(path)
    else:
        split = split_loader.load_split(args.manifests)
    runs, reports = {}, {}
    for d in args.runs:
        run = (contract.load_run(d, split, folds=("external",),
                 external_manifest_name=args.external_manifest.name,
                 external_manifest_sha256=args.external_manifest_sha256)
               if external else contract.load_run(d, split))
        print(run.report())
        if external:
            if run.run_id in runs or run.run_id not in threshold_paths:
                raise ValueError("Run externe répété ou sans reçu de seuil")
            thresholds[run.run_id] = load_external_threshold(run, threshold_paths[run.run_id])
        runs[run.run_id] = run
        if not external:
            reports[run.run_id] = evaluate_run(split, run)
    if external:
        if set(threshold_paths) != set(runs):
            raise ValueError("Les reçus de seuil doivent couvrir exactement les runs demandés")
        _same_checkpoint_thresholds(thresholds.values())
        for run in runs.values():
            reports[run.run_id] = evaluate_run(split, run, external_threshold=thresholds[run.run_id])

    pairs = ([tuple(p.split(":", 1)) for p in args.compare] if args.compare
             else list(itertools.combinations(sorted(runs), 2)))
    comparisons = {}
    for a, b in pairs:
        if a not in runs or b not in runs:
            sys.exit(f"run inconnu dans la comparaison {a}:{b}")
        comparisons[f"{a}_vs_{b}"] = (compare(split, runs, a, b, fold="external", thresholds=thresholds)
                                       if external else compare(split, runs, a, b))

    result = {
        "split": {"filename": args.external_manifest.name if external else split_loader.FROZEN_SPLIT_NAME, "sha256": split.sha256,
                  "n_clips": len(split.clips),
                  "n_clusters": len({c.group_id for c in split.clips})},
        "aggregation_rule": metrics.AGGREGATION,
        "bootstrap": {"draws": metrics.BOOTSTRAP_DRAWS, "seed": metrics.BOOTSTRAP_SEED,
                      "unit": "cluster de dépendance, jamais le clip"},
        "runs": reports,
        "paired_comparisons": comparisons,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "x" if external else "w") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
            fh.write("\n")


if __name__ == "__main__":
    main()
