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
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "temporal"))
from harness import contract, features, metrics, split_loader  # noqa: E402
from stress import TRANSFORMS, clip_rng  # noqa: E402

C_GRID = (0.01, 0.1, 1.0, 10.0)   # gelé
SHUFFLE_SEED = 20260912           # gelé


# Trois notions de commit, à ne pas confondre :
#   model_definition_commit  où les descripteurs ont été figés (features.LADDER)
#   training_commit          HEAD au moment de l'AJUSTEMENT, capturé juste avant
#                            le fit et transporté tel quel dans chaque run qui en
#                            dérive ; jamais recalculé ensuite, ni par le rapport
#   execution_commit         HEAD au moment où le run est écrit
# Ici l'ajustement et l'écriture ont lieu dans le même processus : les deux
# derniers coïncident. Ils restent séparés parce qu'un checkpoint de TSLM, lui,
# peut être entraîné à un commit et exécuté sous stress à un autre.
STRESS_THRESHOLD_RULE = (
    "seuil recalculé sur le fold de validation de ce run transformé (argmax du "
    "macro-F1 cluster-level) ; les métriques dépendantes du seuil ne sont pas "
    "utilisées pour les conclusions de sensibilité temporelle")
RETRAINED_MEANING = "not retrained on stressed data"


REPO_DIR = Path(__file__).resolve().parent


def git_state() -> tuple[str, bool]:
    """(HEAD, worktree modifié ?) du dépôt qui contient CE script.

    Git est interrogé depuis le dossier du script, jamais depuis le répertoire
    courant : lancé ailleurs, il relèverait un autre dépôt ou aucun. Les fichiers
    non suivis sont ignorés.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, cwd=REPO_DIR,
                              text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True,
                                    cwd=REPO_DIR).stdout.strip())
        return head, dirty
    except Exception:
        return "unknown", True


def model_fingerprint(control: str, C: float, scaler, clf) -> str:
    """Empreinte du modèle ajusté : paramètres exacts, octet pour octet.

    Deux runs portant la même empreinte ont été prédits par le même scaler et la
    même régression logistique. C'est ce qui permet de vérifier, sans checkpoint
    sérialisé, qu'un run de stress utilise le modèle du run T0.
    """
    h = hashlib.sha256()
    h.update(f"{control}\x1f{C!r}\x1f".encode("utf-8"))
    for arr in (scaler.mean_, scaler.scale_, clf.coef_, clf.intercept_):
        h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
    return h.hexdigest()


def stress_run_fields(control: str, tname: str, C: float, base_run_id: str,
                      provenance: dict) -> dict:
    """Métadonnées d'un run de stress. Isolé pour être testé sans données."""
    defc = features.LADDER[control]["definition_commit"]
    return {
        "model_name": f"{control} sous {tname} — {TRANSFORMS[tname]['description']}",
        "checkpoint": (f"{contract.NO_SERIALIZED_CHECKPOINT} : définition figée {defc[:7]}, "
                       f"logreg(C={C}) réajustée de façon déterministe sur T0/train, "
                       f"jamais ajustée sur {tname}"),
        "threshold_rule": STRESS_THRESHOLD_RULE,
        "extra": {
            "control_level": control,
            "stress_transform": tname,
            "base_run_id": base_run_id,
            "retrained": False,
            "retrained_meaning": RETRAINED_MEANING,
            "fit_data": "T0/train",
            "hyperparameter_selected_on": "T0/val",
            "stress_applied_at": "évaluation uniquement",
            "selected_C": C,
            **provenance,
        },
    }


def build_matrix(split, data_root: Path, control: str, transform: str = "T0"):
    """Descripteurs de tous les clips, éventuellement après un stress temporel.

    `transform` s'applique au signal BRUT, avant l'extraction. Aucun modèle n'est
    ajusté sur ces descripteurs : on mesure comment le modèle ajusté sur T0/train
    réagit à une organisation temporelle perturbée.
    """
    fn = features.LADDER[control]["fn"]
    tfn = TRANSFORMS[transform]["fn"]
    ids, X, y, g, folds = [], [], [], [], []
    for c in split.clips:
        raw = split_loader.read_wav_raw(data_root, split.path_of(c.clip_id))
        if transform != "T0":
            raw = tfn(raw, clip_rng(transform, c.clip_id))
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
    return ({cid: float(p) for cid, p in zip(ids, probs)}, C, threshold, clf, y_fit, scaler)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--runs-dir", required=True, help="dossier de runs, HORS dépôt")
    ap.add_argument("--controls", nargs="+", default=["C0", "C1", "C2", "C3"])
    ap.add_argument("--stress-transforms", nargs="*", default=[],
                    help="applique T1/T2/T3 à l'audio et y applique le modèle ajusté "
                         "sur T0/train dans ce même processus ; rien n'est ajusté sur "
                         "les données transformées")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="autorise un worktree modifié ; le run le déclare alors "
                         "(training_worktree_dirty=true) et son commit ne le reproduit pas")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="contrôle négatif : étiquettes permutées par cluster, "
                         "on attend un résultat proche du hasard")
    args = ap.parse_args()

    split = split_loader.load_split(args.manifests)
    head, dirty = git_state()
    if head == "unknown" and not args.allow_dirty:
        sys.exit("commit introuvable : aucun training_commit ne pourrait être enregistré. "
                 "Lancer depuis un clone Git du dépôt, ou passer --allow-dirty.")
    if dirty and not args.allow_dirty:
        sys.exit("worktree modifié : le commit enregistré ne reproduirait pas ce run. "
                 "Commiter d'abord, ou passer --allow-dirty (le run le déclarera).")
    summary = []

    for control in args.controls:
        if control not in features.LADDER:
            sys.exit(f"contrôle inconnu : {control}")
        ids, X, y, g, folds = build_matrix(split, Path(args.data_root), control)
        # training_commit capturé ICI, juste avant l'ajustement, puis transporté.
        training_commit, training_dirty = git_state()
        probs, C, threshold, clf, y_fit, scaler = fit_control(
            control, ids, X, y, g, folds, shuffle_labels=args.shuffle_labels)
        fingerprint = model_fingerprint(control, C, scaler, clf)

        suffix = "-shuffled" if args.shuffle_labels else ""
        run_id = f"{control.lower()}{suffix}"
        spec = features.LADDER[control]
        execution_commit, execution_dirty = git_state()
        provenance = {
            "model_definition_commit": spec["definition_commit"],
            "training_worktree_dirty": training_dirty,
            "execution_commit": execution_commit,
            "execution_worktree_dirty": execution_dirty,
            "model_fingerprint": fingerprint,
        }
        run_dir = contract.write_run(
            Path(args.runs_dir) / run_id,
            run_id=run_id,
            model_name=f"{control} — {spec['description']}" + (" [ÉTIQUETTES PERMUTÉES]" if suffix else ""),
            checkpoint=(f"{contract.NO_SERIALIZED_CHECKPOINT} : logreg(C={C}) ajustée de façon "
                        f"déterministe sur T0/train"),
            training_commit=training_commit,
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
                "fit_data": "T0/train",
                "hyperparameter_selected_on": "T0/val",
                **provenance,
            },
        )
        summary.append({"control": control, "run_dir": str(run_dir), "C": C,
                        "threshold": round(float(threshold), 6),
                        "n_features": len(spec["names"])})
        print(f"{run_id:<16} -> {run_dir}")

        # Stress temporel : le scaler et la régression logistique ajustés ci-dessus
        # sur T0/train, dans ce processus, sont appliqués aux descripteurs
        # recalculés sur l'audio transformé. Rien n'est ajusté sur T1/T2/T3.
        for tname in args.stress_transforms:
            if tname == "T0":
                continue
            ids_t, Xt, _, _, _ = build_matrix(split, Path(args.data_root), control, tname)
            assert list(ids_t) == list(ids), "l'ordre des clips a changé sous transformation"
            probs_t = {cid: float(p) for cid, p in
                       zip(ids_t, clf.predict_proba(scaler.transform(Xt))[:, 1])}
            assert model_fingerprint(control, C, scaler, clf) == fingerprint
            rid = f"{run_id}-{tname}"
            stress_exec, stress_exec_dirty = git_state()
            fields = stress_run_fields(control, tname, C, run_id, {
                **provenance,
                "execution_commit": stress_exec,
                "execution_worktree_dirty": stress_exec_dirty,
            })
            d = contract.write_run(
                Path(args.runs_dir) / rid, run_id=rid,
                model_name=fields["model_name"], checkpoint=fields["checkpoint"],
                training_commit=training_commit, split=split,
                threshold_rule=fields["threshold_rule"],
                probabilities=probs_t,
                extra={**fields["extra"], "audio": spec["audio"],
                       "features": list(spec["names"])})
            summary.append({"control": control, "transform": tname, "run_dir": str(d)})
            print(f"{rid:<16} -> {d}")

    print(json.dumps({"runs": summary, "split_sha256": split.sha256,
                      "execution_commit": git_state()[0]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
