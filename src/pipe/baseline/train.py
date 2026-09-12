"""Pilote de développement : aucun entraînement ou réglage sur test final."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .analyses import analyser, lire_options, metriques_selectives
from .donnees import charger_developpement, ecrire_json, empreinte, exiger, lire_json, verifier_features, verifier_texte
from .modele import entrainer, exemple_inference, metriques, predict_baseline, sauvegarder, versions


def charger_configuration(chemin):
    configuration = lire_json(chemin)
    attendus = {"data_path", "split_path", "split_sha256", "output_dir", "seed", "random_forest", "feature_names", "feature_version", "preprocessing_version", "execution_mode"}
    exiger(set(configuration) == attendus, "Clés de configuration incorrectes")
    for cle in ("data_path", "split_path", "output_dir", "feature_version", "preprocessing_version"):
        verifier_texte(configuration[cle], cle)
    exiger(configuration["execution_mode"] in ("live", "development_fixture"), "execution_mode de configuration invalide")
    exiger(type(configuration["seed"]) is int and 0 <= configuration["seed"] < 2 ** 32, "Seed entière requise")
    verifier_features(configuration["feature_names"])
    parametres = configuration["random_forest"]
    exiger(isinstance(parametres, dict) and set(parametres) == {"n_estimators", "max_depth", "min_samples_leaf", "class_weight"}, "Paramètres Random Forest incorrects")
    for cle in ("n_estimators", "min_samples_leaf"):
        exiger(type(parametres[cle]) is int and parametres[cle] > 0, f"{cle} doit être un entier positif")
    exiger(parametres["n_estimators"] <= 1000, "V0 limitée à 1000 arbres")
    profondeur = parametres["max_depth"]
    exiger(profondeur is None or (type(profondeur) is int and profondeur > 0), "max_depth invalide")
    exiger(parametres["class_weight"] in (None, "balanced", "balanced_subsample"), "class_weight invalide")
    return configuration


def exporter(predictions, dossier):
    identifiants = [prediction["sample_id"] for prediction in predictions]
    exiger(bool(predictions) and len(set(identifiants)) == len(identifiants), "Exports vides ou sample_id répétés")
    with (dossier / "predictions.jsonl").open("w") as fichier:
        for prediction in predictions:
            fichier.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
    colonnes = ["sample_id", "input_sha256", "prediction", "score_no_leak", "score_leak", "model_version", "preprocessing_version", "score_type", "execution_mode", "abstained", "abstention_reason"]
    with (dossier / "predictions.csv").open("w", newline="") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=colonnes)
        ecrivain.writeheader()
        for prediction in predictions:
            ligne = {cle: prediction[cle] for cle in colonnes if not cle.startswith("score_")}
            ligne.update(score_type=prediction["score_type"], score_no_leak=prediction["class_scores"]["no_leak"], score_leak=prediction["class_scores"]["leak"])
            ecrivain.writerow(ligne)


def executer(chemin_configuration, chemin_analyses=None):
    chemin_configuration = Path(chemin_configuration).resolve()
    configuration = charger_configuration(chemin_configuration)
    options = lire_options(chemin_analyses) if chemin_analyses else None
    base = chemin_configuration.parent
    donnees, matrice, cibles, partitions = charger_developpement(base / configuration["data_path"], base / configuration["split_path"], configuration)
    provenance_sha256 = None
    if donnees["dataset_version"].startswith("split_v2_binary_with_noise_v0:"):
        from .nevil import verifier_developpement
        verifier_developpement(donnees, base / configuration["split_path"], Path(__file__).resolve().parents[3] / "manifests")
        provenance = lire_json(base / "source_provenance.json")
        exiger(provenance["features_sha256"] == empreinte(base / configuration["data_path"]),
               "Features modifiées après préparation")
        exiger(provenance["projected_split_sha256"] == configuration["split_sha256"],
               "Projection modifiée après préparation")
        provenance_sha256 = empreinte(base / "source_provenance.json")
    dossier = (base / configuration["output_dir"]).resolve()
    exiger(not dossier.exists(), "Le dossier de sortie existe déjà : choisir un nouveau run")
    modele = entrainer(matrice, cibles, partitions, configuration)
    masque = partitions == "validation"
    analyses = None
    if options is not None:
        affectations = lire_json(base / configuration["split_path"])["assignments"]
        groupes_par_id = {ligne["sample_id"]: ligne["event_group_id"] for ligne in affectations}
        groupes = [groupes_par_id[ligne["sample_id"]] for ligne in donnees["samples"]]
        analyses = analyser(modele, matrice, cibles, partitions, groupes, donnees["feature_names"], options, configuration["seed"])
    majoritaire = int(np.bincount(cibles[partitions == "train"], minlength=2).argmax())
    try:
        depot = Path(__file__).resolve().parents[3]
        commit = subprocess.check_output(["git", "-C", str(depot), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        modifie = bool(subprocess.check_output(["git", "-C", str(depot), "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, modifie = None, None
    sources = {fichier.name: empreinte(fichier) for fichier in Path(__file__).parent.glob("*.py")}
    hachage_code = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    instant = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    metadonnees = {cle: donnees[cle] for cle in ("feature_names", "feature_version", "preprocessing_version", "dataset_version", "execution_mode")}
    metadonnees.update(model_version=f"rf-{instant}", environment=versions(), config=configuration,
                       config_sha256=empreinte(chemin_configuration), split_sha256=configuration["split_sha256"],
                       data_sha256=empreinte(base / configuration["data_path"]), commit=commit, working_tree_dirty=modifie,
                       source_sha256=hachage_code, source_files=sources, n_train=int(sum(partitions == "train")),
                       n_validation=int(sum(masque)), final_test_evaluated=False, training_pid=os.getpid(),
                       source_provenance_sha256=provenance_sha256)
    if analyses is not None:
        metadonnees.update(abstention_threshold=analyses["abstention"]["threshold"], abstention_selection_scope="validation_only",
                           analysis_options=options, analysis_config_sha256=empreinte(chemin_analyses))
    artefact = {"model": modele, "metadata": metadonnees}
    predictions = [predict_baseline(artefact, exemple_inference(donnees, ligne)) for ligne, retenue in zip(donnees["samples"], masque) if retenue]
    # Les métriques doivent décrire exactement les sorties remises à Nevil.
    predictions_numeriques = np.asarray([{"no_leak": 0, "leak": 1, None: -1}[prediction["prediction"]] for prediction in predictions])
    hachage = sauvegarder(modele, metadonnees, dossier)
    exporter(predictions, dossier)
    rapport = {"execution_mode": donnees["execution_mode"], "benchmark_eligible": False,
               "scope": "validation_only", "final_test_evaluated": False, "model_sha256": hachage,
               "baseline": metriques(cibles[masque], predictions_numeriques) if not np.any(predictions_numeriques == -1) else None,
               "majority_sanity_check": metriques(cibles[masque], np.full(sum(masque), majoritaire))}
    if analyses is not None:
        rapport.update(analyses=analyses, selective=metriques_selectives(cibles[masque], predictions_numeriques))
    ecrire_json(dossier / "metrics.json", rapport)
    ecrire_json(dossier / "run.json", {"status": "complete", "execution_mode": donnees["execution_mode"], "files": {fichier.name: empreinte(fichier) for fichier in sorted(dossier.iterdir()) if fichier.is_file()}})
    return dossier


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--config", required=True)
    arguments.add_argument("--analyses", help="Options exploratoires, désactivées par défaut")
    options = arguments.parse_args()
    try:
        dossier = executer(options.config, options.analyses)
    except (ValueError, OSError, KeyError, TypeError) as erreur:
        arguments.exit(2, f"Baseline refusée : {erreur}\n")
    print(f"Run de développement sauvegardé : {dossier}")


if __name__ == "__main__":
    main()
