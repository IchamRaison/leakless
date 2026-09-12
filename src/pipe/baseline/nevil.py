"""Préparer le développement puis remettre les scores au harnais de Nevil.

Le cache provient de pipe.tslm.prepare (TimeNet). Aucun nouveau split ni DSP.
Le test n'est ouvert pour inférence qu'après le gel du modèle et de la configuration.
"""
import argparse
import csv
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scripts.eval.harness.contract import load_run
from scripts.eval.harness.split_loader import load_split
from pipe.tslm.prepare import MANIFEST_HASHES, PROTOCOL, safe_member_path
from pipe.tslm.preprocessing import VERSION

from .audio import FEATURE_NAMES, Predictor, agreger, exemple_audio, sources_partagees, version_features
from .donnees import ecrire_json, empreinte, exiger, lire_json
from .integrite import hashes_audio, verifier_audio, lire_provenance


def charger_split(dossier):
    dossier = Path(dossier)
    for nom, attendu in MANIFEST_HASHES.items():
        exiger(empreinte(dossier / nom) == attendu, f"Manifeste gelé modifié : {nom}")
    split = load_split(dossier)
    groupes = {}
    for clip in split.clips:
        exiger(clip.group_id not in groupes or groupes[clip.group_id] == clip.fold,
               "Intersection des groupes entre splits")
        groupes[clip.group_id] = clip.fold
    exiger(Counter(c.fold for c in split.clips) == {"train": 598, "val": 208, "test": 194},
           "Effectifs différents du split v2")
    return split


def lire_cache(dossier, fold, split):
    with np.load(Path(dossier) / f"{fold}.npz", allow_pickle=False) as cache:
        exiger(set(cache.files) == {"ids", "series", "preprocessing_version"}, "Cache inattendu")
        exiger(cache["preprocessing_version"].item() == VERSION, "Version TimeF incompatible")
        identifiants = cache["ids"].tolist()
        series = cache["series"].copy()
    attendus = {c.clip_id for c in split.fold(fold)}
    exiger(len(identifiants) == len(set(identifiants)) and set(identifiants) == attendus,
           "IDs/fold du cache incompatibles")
    exiger(series.shape == (len(identifiants), 4, 64), "Dimensions du cache incompatibles")
    return dict(zip(identifiants, series))


def preparer(manifestes, prepare, racine_audio, sortie):
    sortie, prepare, racine_audio = Path(sortie), Path(prepare), Path(racine_audio)
    exiger(not sortie.exists(), "Dossier de préparation déjà présent")
    split = charger_split(manifestes)
    hashes = hashes_audio(manifestes)
    exiger(set(hashes) == {c.clip_id for c in split.clips}, "Couverture des empreintes audio différente du split")
    rapport = lire_json(prepare / "preparation.json")
    exiger(rapport["protocol"] == PROTOCOL and rapport["manifest_sha256"] == split.sha256
           and rapport["preprocessing_version"] == VERSION and rapport["records"] == 1000,
           "Provenance TimeNet incompatible")
    affectations = [{"sample_id": c.clip_id, "event_group_id": c.group_id,
                     "split": "validation" if c.fold == "val" else c.fold} for c in split.clips]
    lignes, erreur_max = [], 0.0
    # Le manifeste complet sert à l'audit ; seules les features train/val sont lues.
    for fold in ("train", "val"):
        cache = lire_cache(prepare, fold, split)
        for clip in split.fold(fold):
            contenu = safe_member_path(split.path_of(clip.clip_id), racine_audio).read_bytes()
            verifier_audio(contenu, clip.clip_id, hashes)
            exemple = exemple_audio(contenu, clip.clip_id)
            depuis_timef = np.asarray(agreger(cache[clip.clip_id]))
            direct = np.asarray(exemple["features"])
            np.testing.assert_allclose(depuis_timef, direct, rtol=1e-6, atol=1e-6)
            erreur_max = max(erreur_max, float(np.max(np.abs(depuis_timef - direct))))
            # Valeurs du cache TimeNet pour l'entraînement, et parité WAV vérifiée.
            lignes.append({"sample_id": clip.clip_id, "input_sha256": exemple["input_sha256"],
                           "features": depuis_timef.tolist(), "label": clip.label})
    sortie.mkdir(parents=True)
    ecrire_json(sortie / "split.json", {"schema_version": "baseline-split-v0", "assignments": affectations})
    ecrire_json(sortie / "features.json", {"schema_version": "baseline-features-v0", "execution_mode": "live",
                "dataset_version": PROTOCOL + ":" + split.sha256, "feature_names": FEATURE_NAMES,
                "feature_version": version_features(), "preprocessing_version": VERSION, "samples": lignes})
    ecrire_json(sortie / "config.json", {"data_path": "features.json", "split_path": "split.json",
                "split_sha256": empreinte(sortie / "split.json"), "output_dir": "../rf-v1-001",
                "seed": 42, "random_forest": {"n_estimators": 100, "max_depth": None,
                "min_samples_leaf": 2, "class_weight": None}, "feature_names": FEATURE_NAMES,
                "feature_version": version_features(), "preprocessing_version": VERSION, "execution_mode": "live"})
    provenance = {"protocol": PROTOCOL, "frozen_manifest_sha256": split.sha256,
                  "features_sha256": empreinte(sortie / "features.json"),
                  "projected_split_sha256": empreinte(sortie / "split.json"),
                  "audit_sha256": MANIFEST_HASHES["split_v2_audit.csv"], "shared_sources_sha256": sources_partagees(),
                  "preparation_sha256": empreinte(prepare / "preparation.json"),
                  "cache_sha256": {fold: empreinte(prepare / f"{fold}.npz") for fold in ("train", "val")},
                  "max_timef_wav_feature_difference": erreur_max, "tolerance": {"rtol": 1e-6, "atol": 1e-6},
                  "n_train": 598, "n_validation": 208, "test_features_loaded": False,
                  "test_metrics_previously_published_by_team": True, "test_used_for_tuning": False,
                  "benchmark_eligible": False}
    ecrire_json(sortie / "source_provenance.json", provenance)
    return provenance


def verifier_developpement(donnees, chemin_split, manifestes):
    """Rattacher labels et affectations de développement aux sources avant tout fit."""
    split = charger_split(manifestes)
    attendus = [{"sample_id": c.clip_id, "event_group_id": c.group_id,
                 "split": "validation" if c.fold == "val" else c.fold} for c in split.clips]
    recus = lire_json(chemin_split)["assignments"]
    exiger(sorted(recus, key=lambda x: x["sample_id"]) == sorted(attendus, key=lambda x: x["sample_id"]),
           "La projection ne correspond pas au split v2 gelé")
    par_id = split.by_id()
    exiger({l["sample_id"] for l in donnees["samples"]} == {c.clip_id for c in split.clips if c.fold in ("train", "val")},
           "Couverture développement différente du manifeste")
    for ligne in donnees["samples"]:
        exiger(ligne["label"] == par_id[ligne["sample_id"]].label, "Label développement différent du manifeste gelé")
    exiger(donnees["dataset_version"] == PROTOCOL + ":" + split.sha256,
           "Identité du dataset différente du manifeste")


def livrer(manifestes, prepare, racine_audio, developpement, modele, sha256, sortie):
    """Processus neuf, pas de fit : contrôle du modèle gelé et exports sans scoring test."""
    modele, sortie, developpement = Path(modele), Path(sortie), Path(developpement)
    exiger(not sortie.exists(), "Livraison existante, aucun écrasement")
    split = charger_split(manifestes)
    predicteur = Predictor(modele, sha256, artefact_de_confiance=True)
    meta = predicteur.artefact["metadata"]
    exiger(meta["training_pid"] != os.getpid(), "Rechargement exigé dans un nouveau processus")
    exiger(meta["config_sha256"] == empreinte(developpement / "config.json")
           and meta["data_sha256"] == empreinte(developpement / "features.json")
           and meta["split_sha256"] == empreinte(developpement / "split.json"), "Développement modifié après gel")
    provenance = lire_provenance(developpement / "source_provenance.json", meta["source_provenance_sha256"])
    exiger(provenance["frozen_manifest_sha256"] == split.sha256
           and provenance["projected_split_sha256"] == meta["split_sha256"], "Provenance split incohérente")
    for fold, attendu in provenance["cache_sha256"].items():
        exiger(empreinte(Path(prepare) / f"{fold}.npz") == attendu, "Cache modifié après préparation")
    from .modele import predict_baseline
    from .donnees import lire_json as lire
    from .modele import exemple_inference
    donnees = lire(developpement / "features.json")
    verifier_developpement(donnees, developpement / "split.json", manifestes)
    exemples = {l["sample_id"]: exemple_inference(donnees, l) for l in donnees["samples"]}
    import json
    lignes_reference = list(map(json.loads, (modele / "predictions.jsonl").read_text().splitlines()))
    reference = {p["sample_id"]: p for p in lignes_reference}
    exiger(len(reference) == len(lignes_reference), "Prédictions validation répétées")
    attendus_val = {c.clip_id for c in split.fold("val")}
    exiger(set(reference) == attendus_val, "Couverture validation sauvegardée incorrecte")
    for identifiant in sorted(attendus_val):
        recharge = predict_baseline(predicteur.artefact, exemples[identifiant])
        for cle in ("prediction", "class_scores", "abstained", "input_sha256"):
            exiger(recharge[cle] == reference[identifiant][cle], "Prédictions différentes après rechargement")
    resultats = []
    hashes = hashes_audio(manifestes)
    exiger(set(hashes) == {c.clip_id for c in split.clips}, "Couverture des empreintes audio différente du split")
    for clip in (*split.fold("val"), *split.fold("test")):
        contenu = safe_member_path(split.path_of(clip.clip_id), Path(racine_audio)).read_bytes()
        verifier_audio(contenu, clip.clip_id, hashes)
        resultat = predicteur.predict(contenu, sample_id=clip.clip_id).model_dump(mode="json")
        if clip.fold == "val":
            for cle in ("prediction", "class_scores", "input_sha256"):
                exiger(resultat[cle] == reference[clip.clip_id][cle], "Divergence TimeNet/inférence WAV sur validation")
        resultats.append(resultat)
    exiger(empreinte(modele / "model.joblib") == sha256, "Modèle modifié pendant l'inférence")
    sortie.mkdir(parents=True)
    with (sortie / "predictions.csv").open("x", newline="") as fichier:
        csv_sortie = csv.writer(fichier, lineterminator="\n")
        csv_sortie.writerow(["clip_id", "probability_leak"])
        csv_sortie.writerows((r["sample_id"], repr(r["class_scores"]["leak"])) for r in resultats)
    with (sortie / "predictions.jsonl").open("x") as fichier:
        for resultat in resultats:
            fichier.write(json.dumps(resultat, ensure_ascii=False, allow_nan=False) + "\n")
    ecrire_json(sortie / "metadata.json", {"run_id": meta["model_version"], "model_name": "RandomForest-band20",
                "checkpoint": "../model.joblib", "model_sha256": sha256, "training_commit": meta["commit"],
                "training_source_sha256": meta["source_sha256"], "training_working_tree_dirty": meta["working_tree_dirty"],
                "config_hash": meta["config_sha256"], "split_filename": "split_v2.csv", "split_sha256": split.sha256,
                "timestamp": datetime.now(timezone.utc).isoformat(), "protocol": PROTOCOL,
                "threshold_rule": "Nevil: argmax macro-F1 cluster validation, median aggregation; demo RF argmax fixed",
                "test_labels_not_used_for_tuning": True, "test_metrics_previously_published_by_team": True,
                "score_type": "raw", "benchmark_eligible": False, "final_test_evaluated": False})
    contrat = load_run(sortie, split)
    rapport = {"fresh_process": True, "training_pid": meta["training_pid"], "reload_pid": os.getpid(),
               "validation_predictions_identical": len(reference),
               "wav_timef_validation_scores_exact": len(reference), "export_count": len(resultats),
               "validation_count": 208, "test_inference_count": 194, "test_scored": False,
               "model_sha256": sha256, "contract_checks": [vars(c) for c in contrat.checks]}
    ecrire_json(sortie / "reload_and_contract.json", rapport)
    return rapport


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    commandes = arguments.add_subparsers(dest="action", required=True)
    for nom in ("preparer", "livrer"):
        commande = commandes.add_parser(nom)
        commande.add_argument("--manifestes", type=Path, required=True)
        commande.add_argument("--prepare", type=Path, required=True)
        commande.add_argument("--audio", type=Path, required=True)
        commande.add_argument("--sortie", type=Path, required=True)
        if nom == "livrer":
            commande.add_argument("--developpement", type=Path, required=True)
            commande.add_argument("--modele", type=Path, required=True)
            commande.add_argument("--sha256", required=True)
    options = arguments.parse_args()
    if options.action == "preparer":
        resultat = preparer(options.manifestes, options.prepare, options.audio, options.sortie)
    else:
        resultat = livrer(options.manifestes, options.prepare, options.audio, options.developpement,
                         options.modele, options.sha256, options.sortie)
    import json
    print(json.dumps(resultat, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
