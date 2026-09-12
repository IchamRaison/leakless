"""Adaptateur JSON provisoire : Nevil reste propriétaire des features et splits."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


def exiger(condition, message):
    if not condition:
        raise ValueError(message)


def empreinte(chemin):
    return hashlib.sha256(Path(chemin).read_bytes()).hexdigest()


def lire_json(chemin):
    def objet_unique(paires):
        objet = {}
        for cle, valeur in paires:
            exiger(cle not in objet, f"Clé JSON répétée : {cle}")
            objet[cle] = valeur
        return objet

    return json.loads(Path(chemin).read_text(), object_pairs_hook=objet_unique)


def ecrire_json(chemin, objet, *, exclusif=False):
    texte = json.dumps(objet, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with Path(chemin).open("x" if exclusif else "w") as fichier:
        fichier.write(texte)


def verifier_texte(valeur, nom):
    exiger(isinstance(valeur, str) and valeur.strip() == valeur and bool(valeur), f"{nom} doit être un texte non vide")


def verifier_hash(valeur):
    exiger(isinstance(valeur, str) and re.fullmatch(r"[0-9a-f]{64}", valeur), "SHA256 invalide")


def verifier_features(noms):
    exiger(isinstance(noms, list) and bool(noms), "feature_names vide ou invalide")
    for nom in noms:
        verifier_texte(nom, "feature_names")
        termes = set(re.sub(r"([a-z])([A-Z])", r"\1_\2", nom).lower().replace("-", "_").split("_"))
        interdits = {"label", "labels", "target", "class", "id", "filename", "file", "path", "device", "session", "group", "material", "region", "pressure", "velocity", "source", "leak", "truth", "split"}
        exiger(not termes.intersection(interdits), f"Feature interdite : {nom}")
    exiger(len(set(noms)) == len(noms), "feature_names contient des doublons")


def verifier_matrice(valeurs, largeur):
    try:
        brut = np.asarray(valeurs)
    except (ValueError, TypeError) as erreur:
        raise ValueError("Dimensions des features incohérentes") from erreur
    exiger(brut.ndim == 2 and brut.shape[0] > 0 and brut.shape[1] == largeur, "Dimensions des features incohérentes")
    exiger(brut.dtype.kind in "fiu", "Features numériques attendues, sans booléen ni texte")
    exiger(not any(isinstance(valeur, (bool, np.bool_)) for valeur in np.asarray(valeurs, dtype=object).flat), "Features booléennes interdites")
    with np.errstate(over="ignore", invalid="ignore"):
        matrice = brut.astype(np.float32)
    exiger(bool(np.isfinite(matrice).all()), "NaN/Inf ou dépassement float32 : consulter Nevil, aucune imputation automatique")
    exiger(not bool(np.any((matrice == 0) & (brut != 0))), "Sous-dépassement float32 : consulter Nevil")
    return matrice


def charger_developpement(chemin_donnees, chemin_split, configuration):
    """Lire uniquement les features de développement, auditer aussi le manifeste scellé."""
    exiger(Path(chemin_donnees).is_file() and Path(chemin_split).is_file(), "Données/split absents : demander les fichiers de développement à Nevil")
    verifier_hash(configuration["split_sha256"])
    exiger(empreinte(chemin_split) == configuration["split_sha256"], "Le hash du split ne correspond pas à celui fourni")
    manifeste = lire_json(chemin_split)
    exiger(manifeste.get("schema_version") == "baseline-split-v0", "Version du manifeste non supportée")
    affectations = manifeste.get("assignments")
    exiger(isinstance(affectations, list) and bool(affectations), "Manifeste vide")
    par_id, par_groupe = {}, {}
    for ligne in affectations:
        exiger(set(ligne) == {"sample_id", "event_group_id", "split"}, "Le manifeste doit contenir uniquement IDs, groupes et splits")
        identifiant, groupe, partition = ligne["sample_id"], ligne["event_group_id"], ligne["split"]
        verifier_texte(identifiant, "sample_id")
        verifier_texte(groupe, "event_group_id")
        exiger(partition in {"train", "validation", "test", "quarantine"}, "Split inconnu")
        exiger(identifiant not in par_id, "sample_id répété dans le split")
        exiger(groupe not in par_groupe or par_groupe[groupe] == partition, "Intersection des groupes entre splits")
        par_groupe[groupe] = partition
        par_id[identifiant] = ligne

    donnees = lire_json(chemin_donnees)
    attendus = {"schema_version", "execution_mode", "dataset_version", "feature_version", "preprocessing_version", "feature_names", "samples"}
    exiger(set(donnees) == attendus, "Champs du fichier de développement invalides")
    exiger(donnees["schema_version"] == "baseline-features-v0", "Version des données non supportée")
    exiger(donnees["execution_mode"] in {"live", "development_fixture"}, "execution_mode non supporté")
    exiger(donnees["execution_mode"] == configuration["execution_mode"], "Mode synthétique/réel incohérent")
    for cle in ("dataset_version", "feature_version", "preprocessing_version"):
        verifier_texte(donnees[cle], cle)
    verifier_features(donnees["feature_names"])
    exiger(donnees["feature_names"] == configuration["feature_names"], "Ordre ou noms des features incompatibles")
    for cle in ("feature_version", "preprocessing_version"):
        exiger(donnees[cle] == configuration[cle], f"{cle} incompatible")
    lignes = donnees["samples"]
    exiger(isinstance(lignes, list) and bool(lignes), "Données absentes : demander la fixture à Nevil")
    vus, hashes, partitions = set(), {}, []
    for ligne in lignes:
        exiger(set(ligne) == {"sample_id", "input_sha256", "features", "label"}, "Champs d'un échantillon invalides")
        identifiant = ligne["sample_id"]
        verifier_texte(identifiant, "sample_id")
        exiger(identifiant in par_id, "sample_id absent du split fourni")
        partition = par_id[identifiant]["split"]
        exiger(partition in {"train", "validation"}, "Features/labels test ou quarantine interdits dans le pilote de développement")
        exiger(identifiant not in vus, "sample_id répété dans les données")
        vus.add(identifiant)
        exiger(type(ligne["label"]) is int and ligne["label"] in (0, 1), "Labels binaires entiers 0/1 requis")
        verifier_hash(ligne["input_sha256"])
        hachage = ligne["input_sha256"]
        exiger(hachage not in hashes or hashes[hachage] == partition, "Même signal déclaré dans plusieurs splits")
        hashes[hachage] = partition
        partitions.append(partition)
    attendus = {identifiant for identifiant, ligne in par_id.items() if ligne["split"] in {"train", "validation"}}
    exiger(vus == attendus, "Échantillons de développement manquants : aucune omission silencieuse")
    matrice = verifier_matrice([ligne["features"] for ligne in lignes], len(donnees["feature_names"]))
    cibles = np.array([ligne["label"] for ligne in lignes], dtype=np.int64)
    partitions = np.asarray(partitions)
    exiger(set(cibles[partitions == "train"]) == {0, 1}, "Les deux classes doivent être présentes dans train")
    exiger(bool(np.any(partitions == "validation")), "Validation absente")
    return donnees, matrice, cibles, partitions
