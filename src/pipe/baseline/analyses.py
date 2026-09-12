"""Analyses exploratoires sur développement, jamais une évaluation du test final."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from .donnees import exiger, lire_json
from .modele import metriques


def lire_options(chemin):
    options = lire_json(chemin)
    exiger(set(options) == {"thresholds", "minimum_coverage", "cv_folds", "bootstrap_resamples", "bootstrap_min_groups", "amplitude_feature"}, "Options d'analyse incorrectes")
    seuils = options["thresholds"]
    exiger(isinstance(seuils, list) and 0 < len(seuils) <= 100, "Grille de seuils vide ou trop grande")
    exiger(all(type(seuil) in (int, float) and np.isfinite(seuil) and 0.5 <= seuil <= 1 for seuil in seuils), "Seuils attendus entre 0,5 et 1")
    exiger(seuils == sorted(set(seuils)) and seuils[0] == 0.5, "Grille triée, unique et commençant à 0,5 requise")
    couverture = options["minimum_coverage"]
    exiger(type(couverture) in (int, float) and 0 < couverture <= 1, "Couverture minimale attendue dans ]0,1]")
    for cle, minimum, maximum in (("cv_folds", 2, 10), ("bootstrap_resamples", 100, 2000), ("bootstrap_min_groups", 5, 1000)):
        exiger(type(options[cle]) is int and minimum <= options[cle] <= maximum, f"{cle} hors limites")
    exiger(options["amplitude_feature"] is None or options["amplitude_feature"] == "clip_log_rms", "Seule clip_log_rms, fournie par Nevil, peut servir à l'ablation d'amplitude")
    return options


def metriques_selectives(cibles, predites):
    """Le rappel global compte les abstentions sur fuite comme non-détections."""
    cibles, predites = np.asarray(cibles), np.asarray(predites)
    exiger(cibles.ndim == predites.ndim == 1 and len(cibles) == len(predites) > 0, "Dimensions des sorties sélectives incohérentes")
    exiger(set(cibles) <= {0, 1} and set(predites) <= {-1, 0, 1}, "Classes sélectives invalides")
    retenues = predites != -1
    nombre_fuites = int(sum(cibles == 1))
    matrice = [[int(sum((cibles == classe) & (predites == prediction))) for prediction in (0, 1, -1)] for classe in (0, 1)]
    return {"n_total": len(cibles), "n_accepted": int(sum(retenues)), "n_abstained": int(sum(~retenues)),
            "coverage": float(np.mean(retenues)), "accuracy_all_abstentions_as_errors": float(np.mean(predites == cibles)),
            "leak_recall_all": float(sum((cibles == 1) & (predites == 1)) / nombre_fuites) if nombre_fuites else None,
            "retained_accuracy": float(np.mean(predites[retenues] == cibles[retenues])) if retenues.any() else None,
            "accepted_only": metriques(cibles[retenues], predites[retenues]) if retenues.any() else None,
            "true_label_order": ["no_leak", "leak"], "prediction_order": ["no_leak", "leak", "abstention"],
            "confusion_matrix_with_abstentions": matrice}


def selectionner_seuil(cibles, predites, scores_max, seuils, couverture_minimale):
    cibles, predites, scores_max = np.asarray(cibles), np.asarray(predites), np.asarray(scores_max)
    exiger(scores_max.shape == cibles.shape and np.isfinite(scores_max).all(), "Scores de sélection invalides")
    courbe = []
    for seuil in seuils:
        decisions = np.where(scores_max >= seuil, predites, -1)
        courbe.append({"threshold": seuil, **metriques_selectives(cibles, decisions)})
    candidates = [point for point in courbe if point["coverage"] >= couverture_minimale and point["retained_accuracy"] is not None]
    exiger(bool(candidates), "Aucun seuil ne satisfait la couverture minimale")
    choisi = max(candidates, key=lambda point: (point["accepted_only"]["macro_f1"], point["coverage"], -point["threshold"]))
    return {"threshold": choisi["threshold"], "selection_scope": "validation_only", "score_type": "raw",
            "objective": "max_retained_macro_f1_subject_to_minimum_coverage", "minimum_coverage": couverture_minimale,
            "curve": courbe, "independent_evaluation": False,
            "warning": "Seuil choisi sur ces mêmes sorties de validation : performance descriptive, optimiste pour généraliser."}


def validation_croisee(modele, matrice, cibles, groupes, nombre_plis):
    groupes = np.asarray(groupes)
    uniques = np.unique(groupes)
    if len(uniques) < nombre_plis:
        return {"status": "insufficient_groups", "n_groups": len(uniques), "requested_folds": nombre_plis}
    rapports = []
    for numero, (train, validation) in enumerate(GroupKFold(n_splits=nombre_plis).split(matrice, cibles, groupes), 1):
        exiger(not set(groupes[train]) & set(groupes[validation]), "Groupes croisés dans la CV")
        commun = {"fold": numero, "n_train": len(train), "n_validation": len(validation),
                  "n_train_groups": len(set(groupes[train])), "n_validation_groups": len(set(groupes[validation]))}
        if set(cibles[train]) != {0, 1} or set(cibles[validation]) != {0, 1}:
            rapports.append({**commun, "status": "missing_class"})
            continue
        estimateur = clone(modele)
        estimateur.fit(matrice[train], cibles[train])
        rapports.append({**commun, "status": "ok", "metrics": metriques(cibles[validation], estimateur.predict(matrice[validation]))})
    if any(rapport["status"] != "ok" for rapport in rapports):
        return {"status": "incomplete_class_support", "folds": rapports, "summary": None}
    scores = [rapport["metrics"]["macro_f1"] for rapport in rapports]
    return {"status": "ok", "scope": "development_only_without_abstention", "n_groups": len(uniques), "folds": rapports,
            "macro_f1_mean": float(np.mean(scores)), "macro_f1_std_across_folds": float(np.std(scores)),
            "warning": "Dispersion des plis, pas un intervalle de confiance ; ne remplace pas le test final de Nevil."}


def bootstrap_groupes(cibles, predites, groupes, repetitions, minimum_groupes, seed):
    groupes = np.asarray(groupes)
    uniques = np.unique(groupes)
    commun = {"target": "validation_macro_f1_without_abstention", "n_groups": len(uniques), "minimum_groups": minimum_groupes,
              "n_resamples": repetitions, "seed": seed, "confidence_level": 0.95, "resampling_unit": "event_group_id"}
    if len(uniques) < minimum_groupes or set(cibles) != {0, 1}:
        return {**commun, "status": "insufficient_group_or_class_support", "interval": None}
    generateur = np.random.default_rng(seed)
    indices = [np.flatnonzero(groupes == groupe) for groupe in uniques]
    scores, degeneres = [], 0
    for _ in range(repetitions):
        tirages = generateur.choice(len(uniques), size=len(uniques), replace=True)
        selection = np.concatenate([indices[indice] for indice in tirages])
        if set(cibles[selection]) != {0, 1}:
            degeneres += 1
        scores.append(float(f1_score(cibles[selection], predites[selection], labels=[0, 1], average="macro", zero_division=0)))
    if degeneres:
        return {**commun, "status": "unstable_class_support", "n_degenerate_resamples": degeneres, "interval": None,
                "warning": "Pas d'intervalle publié : des rééchantillonnages perdent une classe, aucun filtrage silencieux."}
    return {**commun, "status": "ok", "interval": np.quantile(scores, [0.025, 0.975]).tolist(),
            "point_estimate": float(f1_score(cibles, predites, labels=[0, 1], average="macro", zero_division=0)),
            "warning": "Intervalle percentile conditionnel au modèle et aux groupes déclarés ; seuil minimal heuristique, pas preuve d'indépendance."}


def analyser(modele, matrice, cibles, partitions, groupes, noms, options, seed):
    validation = partitions == "validation"
    train = partitions == "train"
    predites = modele.predict(matrice[validation])
    selection = selectionner_seuil(cibles[validation], predites, modele.predict_proba(matrice[validation]).max(axis=1), options["thresholds"], options["minimum_coverage"])
    feature = options["amplitude_feature"]
    ablation = {"status": "unavailable", "reason": "clip_log_rms non demandée ou absente ; aucun DSP ni substitution de bande."}
    if feature is not None and feature in noms:
        colonne = noms.index(feature)
        estimateur = clone(modele)
        estimateur.fit(matrice[train][:, [colonne]], cibles[train])
        ablation = {"status": "ok", "feature": feature, "scope": "validation_only_without_abstention",
                    "metrics": metriques(cibles[validation], estimateur.predict(matrice[validation][:, [colonne]])),
                    "warning": "Comparaison d'une feature déclarée, pas preuve causale ni test de perturbation de gain."}
    return {"scope": "development_exploration", "benchmark_eligible": False, "abstention": selection,
            "group_cv": validation_croisee(modele, matrice, cibles, groupes, options["cv_folds"]),
            "group_bootstrap": bootstrap_groupes(cibles[validation], predites, np.asarray(groupes)[validation], options["bootstrap_resamples"], options["bootstrap_min_groups"], seed),
            "amplitude_ablation": ablation}
