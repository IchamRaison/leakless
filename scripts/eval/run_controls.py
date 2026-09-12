#!/usr/bin/env python3
"""Produit les runs de prédiction des contrôles C0 à C3.

Chaque contrôle sort dans **le même format que le TSLM de Hicham** :
`metadata.json` + `predictions.csv`. Il n'y a donc pas de chemin d'évaluation
privilégié — le TSLM n'est pas un cas particulier, c'est un run de plus.

Protocole, identique pour les quatre :
  1. descripteurs gelés (`harness/features.py`) ;
  2. standardisation ajustée sur le TRAIN seulement ;
  3. régression logistique, `C` choisi sur la VALIDATION au niveau cluster ;
  4. probabilités écrites pour tous les folds ; le test n'intervient dans aucun choix.

C0 utilise un unique descripteur, le RMS brut. La régression logistique n'y est
qu'une transformation monotone : l'AUC est donc rigoureusement celle du RMS nu,
et le run gagne en échange des probabilités calibrées (score de Brier lisible).

Usage :
  python3 scripts/eval/run_controls.py --data-root <WAV hors dépôt> \\
      --runs-dir <dossier de runs hors dépôt> [--controls C0 C1 C2 C3] [--shuffle-labels]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from harness import contract, features, metrics, split_loader  # noqa: E402

C_GRID = (0.01, 0.1, 1.0, 10.0)   # gelé
SHUFFLE_SEED = 20260912           # gelé


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def build_matrix(split, data_root: Path, control: str):
    fn = features.LADDER[control]["fn"]
    ids, X, y, g, folds = [], [], [], [], []
    for c in split.clips:
        raw = split_loader.read_wav_raw(data_root, split.path_of(c.clip_id))
        ids.append(c.clip_id)
        X.append(fn(raw))
        y.append(c.label)
        g.append(c.group_id)
        folds.append(c.fold)
    return (np.array(ids), np.vstack(X), np.array(y), np.array(g), np.array(folds))


def fit_control(control: str, ids, X, y, g, folds, *, shuffle_labels: bool):
    tr, va = folds == "train", folds == "val"

    y_fit = y.copy()
    if shuffle_labels:
        # Contrôle négatif : on casse la relation étiquette/signal en permutant
        # les étiquettes PAR CLUSTER (un cluster garde une étiquette unique, sinon
        # l'agrégation refuse). Le résultat attendu est le hasard.
        rng = np.random.default_rng(SHUFFLE_SEED)
        uniq = np.unique(g)
        lab = {gg: int(y[g == gg][0]) for gg in uniq}
        shuffled = rng.permutation(list(lab.values()))
        mapping = dict(zip(uniq, shuffled))
        y_fit = np.array([mapping[gg] for gg in g])

    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X)

    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], y_fit[tr])
        pv = clf.predict_proba(Xs[va])[:, 1]
        t = metrics.pick_threshold(y_fit[va], pv, g[va])
        _, gy, gs = metrics.aggregate_clusters(y_fit[va], pv, g[va])
        f1 = metrics.macro_f1(gy, (gs >= t).astype(int))
        if best is None or f1 > best[0]:
            best = (f1, C, clf, t)
    _, C, clf, threshold = best
    probs = clf.predict_proba(Xs)[:, 1]
    return {cid: float(p) for cid, p in zip(ids, probs)}, C, threshold, clf, y_fit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--runs-dir", required=True, help="dossier de runs, HORS dépôt")
    ap.add_argument("--controls", nargs="+", default=["C0", "C1", "C2", "C3"])
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="contrôle négatif : étiquettes permutées par cluster, "
                         "on attend un résultat proche du hasard")
    args = ap.parse_args()

    split = split_loader.load_split(args.manifests)
    commit = git_commit()
    summary = []

    for control in args.controls:
        if control not in features.LADDER:
            sys.exit(f"contrôle inconnu : {control}")
        ids, X, y, g, folds = build_matrix(split, Path(args.data_root), control)
        probs, C, threshold, clf, y_fit = fit_control(
            control, ids, X, y, g, folds, shuffle_labels=args.shuffle_labels)

        suffix = "-shuffled" if args.shuffle_labels else ""
        run_id = f"{control.lower()}{suffix}"
        spec = features.LADDER[control]
        run_dir = contract.write_run(
            Path(args.runs_dir) / run_id,
            run_id=run_id,
            model_name=f"{control} — {spec['description']}" + (" [ÉTIQUETTES PERMUTÉES]" if suffix else ""),
            checkpoint=f"logreg(C={C})",
            training_commit=commit,
            split=split,
            threshold_rule="argmax du macro-F1 cluster-level sur la validation ; "
                           "le test n'intervient dans aucun choix",
            probabilities=probs,
            extra={
                "control_level": control,
                "audio": spec["audio"],
                "features": list(spec["names"]),
                "feature_weights": dict(zip(spec["names"],
                                            [round(float(w), 6) for w in clf.coef_[0]])),
                "selected_C": C,
                "threshold": round(float(threshold), 6),
                "negative_control_shuffled_labels": bool(suffix),
            },
        )
        summary.append({"control": control, "run_dir": str(run_dir), "C": C,
                        "threshold": round(float(threshold), 6),
                        "n_features": len(spec["names"])})
        print(f"{run_id:<16} -> {run_dir}")

    print(json.dumps({"runs": summary, "split_sha256": split.sha256,
                      "commit": commit}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
