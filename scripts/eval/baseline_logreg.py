#!/usr/bin/env python3
"""Baseline acoustique simple : petit vecteur de descripteurs + régression logistique.

Entrées : forme d'onde **normalisée en amplitude**, exactement comme le jeu TimeF.
Le niveau sonore brut est donc retiré par construction — c'est le confound mesuré
(`docs/EVAL_PROTOCOL.md` §7bis-A), et c'est ce qui rend la comparaison avec le
contrôle RMS informative.

Descripteurs, tous calculés sur le signal normalisé, tous sans unité :
  - centroïde spectral, écart-type spectral, roll-off 85 %, platitude spectrale
  - énergie relative dans 5 bandes (0-250, 250-500, 500-1000, 1000-2000, 2000-4000 Hz)
  - taux de passages par zéro
  - facteur de crête (crête / RMS), invariant à la normalisation RMS

Aucune métadonnée de nom de fichier n'entre ici : ni pression, ni débit, ni device,
ni matériau, ni région, ni chemin. `split_v2_audit.csv` ne sert qu'à résoudre
clip_id -> chemin du WAV.

Ajustement : sur le train uniquement. Sélection de C et du seuil : sur la validation
uniquement. Le test est lu une seule fois, pour être rapporté.

Usage :
  python3 scripts/eval/baseline_logreg.py --data-root <dossier des WAV, hors dépôt> \\
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from group_metrics import full_report, pick_threshold  # noqa: E402

SR = 8000
BANDS = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000)]
FEATURE_NAMES = (["centroide_hz", "ecart_type_spectral_hz", "rolloff85_hz", "platitude"]
                 + [f"bande_{lo}_{hi}" for lo, hi in BANDS]
                 + ["taux_passages_zero", "facteur_crete"])


def features(path: Path) -> np.ndarray:
    """Descripteurs du clip, calculés APRÈS normalisation d'amplitude."""
    with wave.open(str(path)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    x = np.zeros(SR)
    x[: min(SR, len(a))] = a[:SR]
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x**2)))
    if rms > 0:
        x = x / rms          # <- la normalisation d'amplitude, identique pour toute classe

    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    total = spec.sum() + 1e-12
    p = spec / total

    centroid = float((freqs * p).sum())
    spread = float(np.sqrt(((freqs - centroid) ** 2 * p).sum()))
    cum = np.cumsum(p)
    rolloff = float(freqs[int(np.searchsorted(cum, 0.85))])
    # Platitude spectrale : moyenne géométrique / moyenne arithmétique. Proche de 1
    # pour un bruit large bande, proche de 0 pour un signal tonal.
    flatness = float(np.exp(np.mean(np.log(spec + 1e-12))) / (np.mean(spec) + 1e-12))
    bands = [float(p[(freqs >= lo) & (freqs < hi)].sum()) for lo, hi in BANDS]
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    crest = float(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-12))
    return np.array([centroid, spread, rolloff, flatness, *bands, zcr, crest])


def load(manifests: Path, data_root: Path):
    with open(manifests / "split_v2.csv") as fh:
        split = list(csv.DictReader(fh))
    with open(manifests / "split_v2_audit.csv") as fh:
        paths = {r["clip_id"]: r["path"] for r in csv.DictReader(fh)}
    rows = []
    for r in split:
        rows.append({"y": 1 if r["label"] == "leak" else 0, "group": r["group_id"],
                     "fold": r["fold"], "x": features(data_root / paths[r["clip_id"]])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = load(Path(args.manifests), Path(args.data_root))

    def fold(f):
        d = [r for r in rows if r["fold"] == f]
        return (np.vstack([r["x"] for r in d]), np.array([r["y"] for r in d]),
                np.array([r["group"] for r in d]))

    Xtr, ytr, gtr = fold("train")
    Xva, yva, gva = fold("val")
    Xte, yte, gte = fold("test")

    scaler = StandardScaler().fit(Xtr)        # ajusté sur le TRAIN seulement
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    # Sélection de C sur la validation, au niveau GROUPE : sinon la plus grosse
    # session du fold choisirait l'hyperparamètre à elle seule.
    from group_metrics import _aggregate_groups, macro_f1
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=2000).fit(Xtr_s, ytr)
        sv = clf.predict_proba(Xva_s)[:, 1]
        t = pick_threshold(yva, sv, gva)
        _, gy, gs = _aggregate_groups(yva, sv, gva)
        f1 = macro_f1(gy, (gs >= t).astype(int))
        if best is None or f1 > best[0]:
            best = (f1, C, clf, t)
    _, C, clf, threshold = best

    report = {
        "baseline": "descripteurs spectraux + régression logistique",
        "entrees": "forme d'onde normalisée en amplitude (le niveau brut est retiré)",
        "descripteurs": FEATURE_NAMES,
        "split": "manifests/split_v2.csv",
        "ajustement": "train ; C et seuil choisis sur la validation, pondérés par groupe ; "
                      "le test n'est lu qu'une fois",
        "C_retenu": C,
        "folds": {},
        "poids": dict(zip(FEATURE_NAMES, [round(float(w), 4) for w in clf.coef_[0]])),
    }
    for name, X, y, g in (("train", Xtr_s, ytr, gtr), ("val", Xva_s, yva, gva),
                          ("test", Xte_s, yte, gte)):
        report["folds"][name] = full_report(name, y, clf.predict_proba(X)[:, 1], g, threshold)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
            fh.write("\n")


if __name__ == "__main__":
    main()
