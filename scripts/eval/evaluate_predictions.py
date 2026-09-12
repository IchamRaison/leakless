#!/usr/bin/env python3
"""L'unique moteur d'évaluation du projet. Tout passe par ici, TSLM compris.

Entrée : un ou plusieurs dossiers de run au format du contrat
(`metadata.json` + `predictions.csv`). Les contrôles C0-C3 et le TSLM de Hicham
ont exactement la même forme : il n'y a pas de chemin privilégié.

Règles appliquées sans exception :

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
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from harness import contract, metrics, split_loader  # noqa: E402


def vectors(split, run, fold: str):
    """(y, score, groups) alignés par clip_id, dans un ordre déterministe."""
    clips = sorted(split.fold(fold), key=lambda c: c.clip_id)
    y = np.array([c.label for c in clips])
    g = np.array([c.group_id for c in clips])
    s = np.array([run.probabilities[c.clip_id] for c in clips])
    ids = [c.clip_id for c in clips]
    return y, s, g, ids


# Champs de provenance recopiés tels quels depuis metadata.json. Le moteur ne
# les recalcule jamais : un commit lu ici est celui que le run a déclaré.
PROVENANCE_FIELDS = ("model_definition_commit", "training_worktree_dirty",
                     "execution_commit", "execution_worktree_dirty", "model_fingerprint",
                     "fit_data", "control_level", "stress_transform", "base_run_id",
                     "retrained", "retrained_meaning")


def evaluate_run(split, run) -> dict:
    yv, sv, gv, _ = vectors(split, run, "val")
    threshold = metrics.pick_threshold(yv, sv, gv)      # validation SEULEMENT

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
        "provenance": {k: run.metadata[k] for k in PROVENANCE_FIELDS if k in run.metadata},
        "contract_checks": [{"name": c.name, "passed": c.passed, "coverage": c.coverage}
                            for c in run.checks],
        "folds": {},
    }
    for fold in ("val", "test"):
        y, s, g, _ = vectors(split, run, fold)
        block = metrics.evaluate(y, s, g, threshold)
        block["bootstrap_ci95"] = metrics.bootstrap_ci(y, s, g, threshold)
        out["folds"][fold] = block
    return out


def compare(split, runs: dict, a: str, b: str, fold: str = "test") -> dict:
    ra, rb = runs[a], runs[b]
    ya, sa, ga, ida = vectors(split, ra, fold)
    yb, sb, gb, idb = vectors(split, rb, fold)
    if ida != idb:
        raise ValueError(
            f"populations de clips différentes entre {a} et {b} — comparaison refusée")
    yv_a, sv_a, gv_a, _ = vectors(split, ra, "val")
    yv_b, sv_b, gv_b, _ = vectors(split, rb, "val")
    ta = metrics.pick_threshold(yv_a, sv_a, gv_a)
    tb = metrics.pick_threshold(yv_b, sv_b, gv_b)
    d = metrics.paired_bootstrap_delta(ya, ga, sa, sb, ta, tb)
    d.update({"fold": fold, "a": a, "b": b, "n_clips": int(len(ya)),
              "avertissement": "aucune significativité statistique n'est revendiquée : "
                               f"{d['n_clusters']} clusters indépendants seulement"})
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="paires « A:B » ; par défaut toutes les paires")
    ap.add_argument("--out")
    args = ap.parse_args()

    split = split_loader.load_split(args.manifests)
    runs, reports = {}, {}
    for d in args.runs:
        run = contract.load_run(d, split)
        print(run.report())
        runs[run.run_id] = run
        reports[run.run_id] = evaluate_run(split, run)

    pairs = ([tuple(p.split(":", 1)) for p in args.compare] if args.compare
             else list(itertools.combinations(sorted(runs), 2)))
    comparisons = {}
    for a, b in pairs:
        if a not in runs or b not in runs:
            sys.exit(f"run inconnu dans la comparaison {a}:{b}")
        comparisons[f"{a}_vs_{b}"] = compare(split, runs, a, b)

    result = {
        "split": {"filename": split_loader.FROZEN_SPLIT_NAME, "sha256": split.sha256,
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
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
            fh.write("\n")


if __name__ == "__main__":
    main()
