#!/usr/bin/env python3
"""Contrôle RMS seul, sur le signal BRUT, évalué sur les folds gelés de `split_v2`.

Pourquoi ce contrôle existe : les clips leak sont ~11 dB plus forts que les no-leak,
et c'est un artefact du protocole d'acquisition, pas une signature de fuite. Un modèle
peut obtenir un score élevé en mesurant le volume. Ce script mesure exactement ce que
le volume seul rapporte, sur les mêmes folds que tout le reste.

> ### Le signal est lu BRUT, sans normalisation d'amplitude.
> Les entrées du modèle et de la baseline sont normalisées ; ce contrôle ne l'est pas.
> Le calculer sur de l'audio normalisé mesurerait une information supprimée par
> construction et produirait un contrôle artificiellement faible qui flatterait tout
> ce à quoi on le compare. `docs/EVAL_PROTOCOL.md` §7bis-A.

Le seuil est choisi sur la VALIDATION, pondéré par groupe. Le test n'est touché
qu'une fois, pour être rapporté.

Usage :
  python3 scripts/eval/rms_control.py --data-root <dossier des WAV, hors dépôt> \\
      [--manifests manifests] [--out <json hors dépôt>]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from group_metrics import full_report, pick_threshold  # noqa: E402


def rms_dbfs(path: Path) -> float:
    """Niveau RMS en dBFS du signal brut. Aucune normalisation."""
    with wave.open(str(path)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    a = a - a.mean()
    return float(20 * np.log10(np.sqrt(np.mean(a**2)) / 32768 + 1e-12))


def load(manifests: Path, data_root: Path):
    with open(manifests / "split_v2.csv") as fh:
        split = list(csv.DictReader(fh))
    # split_v2_audit.csv ne sert QU'À résoudre clip_id -> chemin. Aucune autre colonne
    # n'est lue : ni pression, ni débit, ni device, ni matériau.
    with open(manifests / "split_v2_audit.csv") as fh:
        paths = {r["clip_id"]: r["path"] for r in csv.DictReader(fh)}
    rows = []
    for r in split:
        rows.append({
            "clip_id": r["clip_id"],
            "y": 1 if r["label"] == "leak" else 0,
            "group": r["group_id"],
            "fold": r["fold"],
            "rms": rms_dbfs(data_root / paths[r["clip_id"]]),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = load(Path(args.manifests), Path(args.data_root))
    by = {f: [r for r in rows if r["fold"] == f] for f in ("train", "val", "test")}

    def arrays(f):
        d = by[f]
        return (np.array([r["y"] for r in d]),
                np.array([r["rms"] for r in d]),
                np.array([r["group"] for r in d]))

    # Le "modèle" est le niveau brut lui-même : un seul scalaire, rien à ajuster sur
    # le train au-delà du choix du seuil. Le seuil vient de la validation.
    yv, sv, gv = arrays("val")
    threshold = pick_threshold(yv, sv, gv)

    report = {
        "controle": "RMS seul (signal brut, non normalisé)",
        "split": "manifests/split_v2.csv",
        "seuil_choisi_sur": "validation, pondéré par groupe",
        "avertissement": "ce contrôle n'est pas un modèle : il mesure le volume. "
                         "Un résultat qui ne le dépasse pas décrit un volumètre.",
        "folds": {},
    }
    for f in ("train", "val", "test"):
        y, s, g = arrays(f)
        report["folds"][f] = full_report(f, y, s, g, threshold)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
            fh.write("\n")


if __name__ == "__main__":
    main()
