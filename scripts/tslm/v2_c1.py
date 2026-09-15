"""C1 phase 5 : grille officielle, CV train groupée et checkpoint complet.

Helpers sans CLI ni accès aux données. Avant tout appel réel, l'orchestrateur
doit vérifier le receipt de parité PASS, les 598 IDs / 102 groupes train, le SHA
du folds.json diagnostique et la préinscription. X suit exactement l'ordre de
rows et provient de features.c1_envelope, sans modifier sa définition.

compare attend folds_json['folds'] ; il ne crée pas de nouvelles partitions.
Le seuil 0.5 n'est que diagnostique : le seuil final sera choisi séparément sur
la validation officielle, jamais ici. Aucun fit ni seuil sur un holdout externe.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import pickle
import re
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from harness import features, metrics  # noqa: E402

SEED = 20260912
DIAGNOSTIC_THRESHOLD = 0.5
SCHEMA = "pipe-v2-c1-checkpoint-v1"


def _official_grid():
    # Import tardif : les validations structurelles n'exigent pas sklearn.
    from run_controls import C_GRID
    return C_GRID


def _matrix(X):
    X = np.asarray(X, dtype=np.float64)
    if (X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != len(features.C1_NAMES)
            or not np.isfinite(X).all()):
        raise ValueError("Matrice C1 non vide, finie et à neuf descripteurs requise")
    return X


def _partition_indices(rows, X, folds):
    """Tout valider avant le premier fit, y compris la couverture des 3 folds."""
    if len(rows) != X.shape[0] or len(folds) != 3:
        raise ValueError("Matrice alignée aux rows et exactement trois folds requis")
    positions, group_labels = {}, {}
    for i, row in enumerate(rows):
        cid, group, label = row["clip_id"], row["group_id"], row["label"]
        if (not isinstance(cid, str) or not cid or cid in positions
                or not isinstance(group, str) or not group or row["fold"] != "train"
                or label not in ("leak", "no_leak")
                or group_labels.get(group, label) != label):
            raise ValueError("IDs, groupes ou classes incohérents, ou données hors train")
        positions[cid], group_labels[group] = i, label
    y = np.array([int(row["label"] == "leak") for row in rows])
    groups = np.array([row["group_id"] for row in rows])
    all_ids, seen_heldout, partitions, fold_ids = set(positions), [], [], []
    for fold in folds:
        train_ids, heldout_ids = fold["train_ids"], fold["heldout_ids"]
        train_set, heldout_set = set(train_ids), set(heldout_ids)
        if (len(train_set) != len(train_ids) or len(heldout_set) != len(heldout_ids)
                or train_set & heldout_set or train_set | heldout_set != all_ids):
            raise ValueError("Partition train/heldout incomplète, inconnue ou dupliquée")
        tr, ho = np.array([positions[cid] for cid in train_ids], dtype=int), np.array(
            [positions[cid] for cid in heldout_ids], dtype=int)
        if (set(groups[tr]) & set(groups[ho]) or set(y[tr]) != {0, 1}
                or set(y[ho]) != {0, 1}
                or sorted(fold["heldout_groups"]) != sorted(set(groups[ho]))):
            raise ValueError("Groupes partagés, classes absentes ou heldout_groups incorrects")
        partitions.append((fold["fold_id"], tr, ho))
        fold_ids.append(fold["fold_id"])
        seen_heldout.extend(heldout_ids)
    if sorted(fold_ids) != [0, 1, 2] or sorted(seen_heldout) != sorted(all_ids):
        raise ValueError("Chaque clip train doit être tenu à l'écart exactement une fois")
    return y, groups, partitions


def fit_final(X, y, C):
    """Ajuste uniquement scaler + LogReg au C déjà choisi ; aucun choix interne.

    Ce même helper sert aux inner-train de compare. Lors du refit final, le
    l'orchestrateur doit fournir uniquement les 598 train autorisés, jamais train + val.
    """
    X, y = _matrix(X), np.asarray(y)
    if y.ndim != 1 or len(y) != len(X) or set(y.tolist()) != {0, 1}:
        raise ValueError("Étiquettes binaires alignées, avec les deux classes, requises")
    if isinstance(C, bool) or C not in _official_grid():
        raise ValueError("C doit appartenir à la grille officielle gelée")
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    classifier = LogisticRegression(C=float(C), max_iter=5000, solver="lbfgs",
                                    random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(scaler.transform(X), y)
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError("Ordre des classes inattendu")
    return scaler, classifier


def predict_probability(scaler, classifier, X):
    """Probabilité brute de la classe 1, sans seuil ni remplacement numérique."""
    X = _matrix(X)
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError("La deuxième colonne doit être la classe fuite = 1")
    scores = np.asarray(classifier.predict_proba(scaler.transform(X)))[:, 1]
    if (scores.shape != (len(X),) or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Scores non finis ou hors bornes : aucun remplacement")
    return scores


def compare(rows, X, folds):
    """Quatre C × trois folds ; moyenne non pondérée des AUC groupe exactes."""
    X = _matrix(X)
    y, groups, partitions = _partition_indices(rows, X, folds)
    candidates = []
    for C in sorted(_official_grid()):
        reports = []
        for fold_id, tr, ho in partitions:
            scaler, classifier = fit_final(X[tr], y[tr], C)
            scores = predict_probability(scaler, classifier, X[ho])
            report = metrics.evaluate(y[ho], scores, groups[ho], DIAGNOSTIC_THRESHOLD)
            _, group_y, group_scores = metrics.aggregate_clusters(y[ho], scores, groups[ho])
            report.update(fold_id=fold_id, n_fit_clips=len(tr),
                          group_roc_auc_full=metrics.roc_auc(group_y, group_scores),
                          clip_roc_auc_full=metrics.roc_auc(y[ho], scores))
            reports.append(report)
        mean_group = float(np.mean([r["group_roc_auc_full"] for r in reports]))
        if not np.isfinite(mean_group):
            raise ValueError("AUC groupe non définie : aucune sélection de remplacement")
        candidates.append({"C": float(C), "folds": reports, "mean_group_roc_auc": mean_group,
            "mean_clip_roc_auc": float(np.mean([r["clip_roc_auc_full"] for r in reports]))})
    selected = min(candidates, key=lambda item: (-item["mean_group_roc_auc"], item["C"]))
    return {"selected_C": selected["C"], "candidates": candidates,
            "n_configs_compared": len(candidates), "n_fits": len(candidates) * len(partitions),
            "selection_rule": "max_mean_inner_fold_group_roc_auc_then_smallest_C",
            "threshold": DIAGNOSTIC_THRESHOLD,
            "threshold_rule": "fixed_diagnostic_only_not_selected_not_served", "seed": SEED}


def _runtime():
    return {"numpy": np.__version__, "scikit-learn": importlib.metadata.version("scikit-learn")}


def _validate_models(scaler, classifier):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.validation import check_is_fitted
    if type(scaler) is not StandardScaler or type(classifier) is not LogisticRegression:
        raise ValueError("Checkpoint C1 : scaler / classifieur inattendu")
    check_is_fitted(scaler)
    check_is_fitted(classifier)
    if (scaler.n_features_in_ != len(features.C1_NAMES)
            or classifier.n_features_in_ != len(features.C1_NAMES)
            or classifier.classes_.tolist() != [0, 1]
            or classifier.C not in _official_grid() or classifier.solver != "lbfgs"
            or classifier.max_iter != 5000 or classifier.random_state != SEED
            or classifier.class_weight is not None or not scaler.with_mean or not scaler.with_std):
        raise ValueError("Checkpoint incompatible avec la recette C1 gelée")
    for values in (scaler.mean_, scaler.var_, scaler.scale_, classifier.coef_, classifier.intercept_):
        if not np.isfinite(values).all():
            raise ValueError("Paramètres du checkpoint non finis")


def save_checkpoint(path, scaler, classifier):
    """Écrit nos objets complets sans écrasement et retourne leur SHA-256.

    L'orchestrateur conserve ce SHA dans sa provenance gelée. Ni coefficients arrondis
    ni reconstruction approximative du scaler. Le pickle n'est pas portable
    entre versions de sklearn / NumPy ; les versions exactes sont enregistrées.
    """
    _validate_models(scaler, classifier)
    payload = {"schema": SCHEMA, "runtime": _runtime(), "seed": SEED,
               "feature_names": list(features.C1_NAMES),
               "feature_source_sha256": hashlib.sha256(Path(features.__file__).read_bytes()).hexdigest(),
               "scaler": scaler, "classifier": classifier}
    content = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
    return hashlib.sha256(content).hexdigest()


def load_checkpoint(path, *, expected_sha256, trusted=False):
    """Nos propres artefacts uniquement : SHA vérifié AVANT désérialisation.

    Un hash n'est pas une preuve de sûreté d'un pickle tiers. trusted=True doit
    être explicite et le SHA attendu doit provenir de notre gel indépendant.
    """
    if trusted is not True:
        raise ValueError("Le chargement pickle exige un artefact propre de confiance explicite")
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("Empreinte SHA-256 complète épinglée requise")
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Empreinte du checkpoint différente du gel : pickle non chargé")
    payload = pickle.loads(content)
    if (not isinstance(payload, dict) or payload.get("schema") != SCHEMA
            or payload.get("runtime") != _runtime() or payload.get("seed") != SEED
            or payload.get("feature_names") != list(features.C1_NAMES)
            or payload.get("feature_source_sha256") != hashlib.sha256(
                Path(features.__file__).read_bytes()).hexdigest()):
        raise ValueError("Recette, versions ou descripteurs C1 différents du checkpoint")
    scaler, classifier = payload["scaler"], payload["classifier"]
    _validate_models(scaler, classifier)
    return scaler, classifier
